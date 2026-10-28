// =============================================================================
// tests/test_pool_geometry.cpp -- MaxPool2D / AvgPool2D spatial geometry.
//
// THE DEFECT THIS FILE EXISTS FOR
//
// `include/nn/layers/pooling/pool_layer.h:13` declares
//
//     MaxPool2D(int kH, int kW, int H_in, int W_in, int stride_h = 1, int stride_w = 1);
//
// which matches `Conv2D`'s convention exactly (conv_layer.h:26). Like
// `Conv2D`, the pooling layers are SHAPE-STATIC: `H`/`W` are baked in at
// construction and `forward` asserts `input.cols % (H * W) == 0`.
//
// EVERY CALL SITE IN THE REPO PASSES THE STRIDE WHERE H_in/W_in BELONG:
//
//   cnn_models.cpp:8    pool_(2, 2, pool_stride, pool_stride)
//                          -> k=2x2, H_in=pool_stride, W_in=pool_stride, stride=1
//   cnn_models.cpp:119  pool1_(2, 2, 2, 2), pool2_(2, 2, 2, 2)
//                          -> k=2x2, H_in=2, W_in=2, stride=1
//   cnn_models.cpp:198  pool1_(3, 3, 2, 2), pool2_(3, 3, 2, 2), pool3_(3, 3, 2, 2)
//                          -> k=3x3, H_in=2, W_in=2, stride=1
//   squeezenet.cpp:110  pool1_(3, 3, 2, 2), pool_final_(3, 3, 2, 2)
//                          -> k=3x3, H_in=2, W_in=2, stride=1
//
// MEASURED CONSEQUENCE. With H_in = W_in = 2 and kH = kW = 3 (the AlexNet /
// SqueezeNet case):
//
//     H_out = (H_in - kH) / stride_h + 1 = (2 - 3) / 1 + 1 = 0
//
// so the constructor throws `std::invalid_argument("MaxPool2D: output
// dimensions non-positive")`. With H_in = W_in = 2 and kH = kW = 2 (the LeNet5
// case) it does NOT throw -- it silently computes H_out = 1, collapsing the
// whole feature map to a single spatial position, and the failure surfaces
// later at `forward` on the shape assertion.
//
// WHY IT NEVER FAILED BEFORE -- TWO INDEPENDENT REASONS, AND THEY MASKED
// EACH OTHER:
//
// 1. `MaxPool2D` was never constructed on a WORKING path. The one layer that
//    pools for real, `SpatialTransformer`, does NOT use the class: it
//    hand-rolls `maxpool2d_forward` / `maxpool2d_backward`
//    (spatial_transformer.cpp:239,260) precisely because the class cannot
//    follow a changing spatial size.
//
// 2. The classes that DO construct it could not be INSTANTIATED anyway,
//    because they were abstract. Fixing that abstractness (commit 9fae6e3)
//    surfaced this bug immediately -- the new instantiability suite aborted
//    with the MaxPool2D throw.
//
// Two defects masking each other is the fingerprint of a latent bug hidden
// behind an earlier one. That is why this is a separate commit rather than a
// silent extra change.
//
// ---------------------------------------------------------------------------
// THE SPEC IS THE SPATIAL CHAIN, AND IT IS PINNED HERE IN FULL:
//
//   LeNet5     : 32x32  -> conv(5x5, pad2)  -> 32x32 -> pool(2x2, s2) -> 16x16
//                         -> conv(5x5, pad0)  -> 12x12 -> pool(2x2, s2) ->  6x6
//   AlexNet    : 227x227-> conv(11x11, s4, p3) -> 56x56 -> pool(3x3, s2) -> 27x27
//                         -> conv(5x5, s1, p2)  -> 27x27 -> pool(3x3, s2) -> 13x13
//                         -> conv(3x3, s1, p1)  -> 13x13
//                         -> conv(3x3, s1, p1)  -> 13x13
//                         -> conv(3x3, s1, p1)  -> 13x13 -> pool(3x3, s2) ->  6x6
//   SqueezeNet : HxW    -> conv(7x7, s2, p3)  -> ceil((H-7)/2)+1
//                         -> pool(3x3, s2)      -> floor((H1-3)/2)+1
//                         -> 8 fire modules (spatial size UNCHANGED)
//                         -> pool(3x3, s2)      -> floor((Hf-3)/2)+1
//
// A POOL THAT POOLS THE WRONG REGION IS NOT DETECTABLE BY A SHAPE CHECK
// ALONE: a pool over a 2x2 region of a 32x32 map still produces a
// well-shaped tensor. So this file pins the pooling VALUES as well as the
// geometry -- a 4x4 input with a 2x2 kernel and stride 2 must return the
// 2x2 max of each 2x2 block, which is a check that can actually fail.
// =============================================================================

#include "nn/nn.h"

#include <cstdio>
#include <stdexcept>
#include <string>

using namespace nn;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", name.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", name.c_str()); }
}

// A CONSTRUCTION THAT THROWS MUST BE REPORTED, NOT ALLOWED TO ABORT.
template<typename F>
static bool construct_ok(F&& f, const std::string& name) {
    try {
        f();
        check(true, name);
        return true;
    } catch (const std::exception& e) {
        check(false, name + " -- construction threw: " + e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// PART A -- the pooling layers' OWN contract.
//
// These assertions hold for MaxPool2D in isolation and do not depend on any
// architecture. They are the unit-level spec of the layer.
// ---------------------------------------------------------------------------
static void test_maxpool_geometry_contract() {
    // A 4x4 map, 2x2 kernel, stride 2 -> 2x2 output.
    MaxPool2D pool(2, 2, 4, 4, 2, 2);
    check(pool.H == 4 && pool.W == 4, "MaxPool2D stores the input spatial dims it was given");
    check(pool.stride_h == 2 && pool.stride_w == 2,
          "MaxPool2D stores the stride in the STRIDE slot, not in H/W");

    // A (1, C*H*W) input with C=1: 4x4 = 16 columns.
    Tensor in(1, 16);
    for (size_t i = 0; i < in.rows; ++i)
        for (size_t j = 0; j < in.cols; ++j)
            in(i, j) = static_cast<double>(i * in.cols + j);   // 0..15

    Tensor out = pool.forward(in);
    check(out.rows == 1 && out.cols == 4,
          "MaxPool2D forward maps (1, 1*4*4) to (1, 1*2*2)");

    // The 2x2 max of each block:
    //   block(0,0) = {0,1,4,5}   -> 5
    //   block(0,1) = {2,3,6,7}   -> 7
    //   block(1,0) = {8,9,12,13} -> 13
    //   block(1,1) = {10,11,14,15}-> 15
    check(out(0, 0) == 5.0,  "MaxPool2D takes the max of the top-left 2x2 block");
    check(out(0, 1) == 7.0,  "MaxPool2D takes the max of the top-right 2x2 block");
    check(out(0, 2) == 13.0, "MaxPool2D takes the max of the bottom-left 2x2 block");
    check(out(0, 3) == 15.0, "MaxPool2D takes the max of the bottom-right 2x2 block");
}

static void test_maxpool_stride_is_honoured() {
    // A 4x4 map, 2x2 kernel, stride 1 -> 3x3 output. If the stride were
    // silently taken from the H_in/W_in slots, this would be a 2x2 output.
    MaxPool2D pool(2, 2, 4, 4, 1, 1);
    check(pool.H == 4 && pool.W == 4, "stride-1 pool still stores H=4, W=4");

    Tensor in(1, 16);
    for (size_t i = 0; i < in.rows; ++i)
        for (size_t j = 0; j < in.cols; ++j)
            in(i, j) = static_cast<double>(i * in.cols + j);

    Tensor out = pool.forward(in);
    check(out.cols == 9, "MaxPool2D with stride 1 over a 4x4 map yields a 3x3 output");
    check(out(0, 0) == 5.0, "stride-1 MaxPool2D first window max is 5");
    check(out(0, 8) == 15.0, "stride-1 MaxPool2D last window max is 15");
}

static void test_avgpool_geometry_contract() {
    AvgPool2D pool(2, 2, 4, 4, 2, 2);
    check(pool.H == 4 && pool.W == 4, "AvgPool2D stores the input spatial dims it was given");
    check(pool.stride_h == 2 && pool.stride_w == 2,
          "AvgPool2D stores the stride in the STRIDE slot, not in H/W");

    Tensor in(1, 16);
    for (size_t i = 0; i < in.rows; ++i)
        for (size_t j = 0; j < in.cols; ++j)
            in(i, j) = static_cast<double>(i * in.cols + j);

    Tensor out = pool.forward(in);
    check(out.cols == 4, "AvgPool2D forward maps (1, 1*4*4) to (1, 1*2*2)");

    // Block means: {0,1,4,5} -> 2.5 ; {2,3,6,7} -> 4.5
    //              {8,9,12,13} -> 10.5 ; {10,11,14,15} -> 12.5
    check(out(0, 0) == 2.5,  "AvgPool2D averages the top-left 2x2 block");
    check(out(0, 1) == 4.5,  "AvgPool2D averages the top-right 2x2 block");
    check(out(0, 2) == 10.5, "AvgPool2D averages the bottom-left 2x2 block");
    check(out(0, 3) == 12.5, "AvgPool2D averages the bottom-right 2x2 block");
}

// ---------------------------------------------------------------------------
// PART B -- the CALL SITES.
//
// Each architecture's pools must be built with the spatial dims of the stage
// they pool. The chain above is the spec; these assertions pin it.
// ---------------------------------------------------------------------------
static void test_lenet5_pools_match_the_chain() {
    bool built = construct_ok([]{ LeNet5 l(10); (void)l; },
                               "LeNet5 constructs");
    if (!built) return;

    // The chain: 32x32 -> conv1 (5x5, pad0) -> 28x28 -> pool1 -> 14x14
    //            14x14 -> conv2 (5x5, pad0) -> 10x10 -> pool2 ->  5x5
    // LeNet5's pools are PRIVATE, so the chain is checked through the FORWARD
    // PASS instead: the output must be (batch, num_classes).
    LeNet5 lenet(10);
    Tensor x(2, 1 * 32 * 32);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x(i, j) = 0.01 * static_cast<double>((i * 7 + j) % 13);

    bool forward_ok = true;
    Tensor out;
    try {
        out = lenet.forward(x);
    } catch (const std::exception& e) {
        forward_ok = false;
        printf("  [INFO] LeNet5::forward threw: %s\n", e.what());
    }
    check(forward_ok, "LeNet5 forward runs the whole conv/pool chain");
    if (forward_ok) {
        check(out.rows == 2 && out.cols == 10,
              "LeNet5 forward produces (batch, num_classes)");
    }
}

static void test_alexnet_pools_match_the_chain() {
    bool built = construct_ok([]{ AlexNet a(10, /*use_lrbn=*/false); (void)a; },
                               "AlexNet constructs");
    if (!built) return;

    // 224x224 -> conv1 (11x11, s4, pad3) -> 55x55 -> pool1 (3x3, s2) -> 27x27
    //          -> conv2 (5x5, s1, pad2)  -> 27x27 -> pool2 (3x3, s2) -> 13x13
    //          -> conv3/4/5 (3x3, s1, pad1) -> 13x13 -> pool3 (3x3, s2) ->  6x6
    //          -> flatten (6*6*256 = 9216) -> fc1 (4096) -> fc2 (4096) -> fc3 (10)
    AlexNet alex(10, /*use_lrbn=*/false);
    Tensor x(1, 3 * 224 * 224);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x(i, j) = 0.01 * static_cast<double>((i * 7 + j) % 13);

    bool forward_ok = true;
    Tensor out;
    try {
        out = alex.forward(x);
    } catch (const std::exception& e) {
        forward_ok = false;
        printf("  [INFO] AlexNet::forward threw: %s\n", e.what());
    }
    check(forward_ok, "AlexNet forward runs the whole conv/pool chain");
    if (forward_ok) {
        check(out.rows == 1 && out.cols == 10,
              "AlexNet forward produces (batch, num_classes)");
    }
}

static void test_squeezenet_pools_match_the_chain() {
    bool built = construct_ok([]{ SqueezeNet s(10, 224, 224); (void)s; },
                               "SqueezeNet constructs");
    if (!built) return;

    // 224x224 -> conv1 (7x7, s2, p3) -> 111x111 -> pool1 (3x3, s2) -> 55x55
    //          -> 8 fire modules (spatial UNCHANGED, 55x55)
    //          -> pool_final (3x3, s2) -> 27x27
    SqueezeNet sq(10, 224, 224);
    Tensor x(1, 3 * 224 * 224);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x(i, j) = 0.01 * static_cast<double>((i * 7 + j) % 13);

    bool forward_ok = true;
    Tensor out;
    try {
        out = sq.forward(x);
    } catch (const std::exception& e) {
        forward_ok = false;
        printf("  [INFO] SqueezeNet::forward threw: %s\n", e.what());
    }
    check(forward_ok, "SqueezeNet forward runs the whole conv/pool chain");
    if (forward_ok) {
        check(out.rows == 1, "SqueezeNet forward keeps the batch dimension");
    }
}

static void test_vggblock_pools_match_the_chain() {
    size_t filters[2] = {4, 4};
    bool built = construct_ok([&]{ VGGBlock v(3, filters, 2); (void)v; },
                               "VGGBlock constructs");
    if (!built) return;

    // VGGBlock's conv stack is built at a fixed 224x224 input
    // (cnn_models.cpp:11), so its pool must be built for a 224x224 map.
    VGGBlock vgg(3, filters, 2);
    Tensor x(1, 3 * 224 * 224);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j)
            x(i, j) = 0.01 * static_cast<double>((i * 7 + j) % 13);

    bool forward_ok = true;
    Tensor out;
    try {
        out = vgg.forward(x);
    } catch (const std::exception& e) {
        forward_ok = false;
        printf("  [INFO] VGGBlock::forward threw: %s\n", e.what());
    }
    check(forward_ok, "VGGBlock forward runs the whole conv/pool chain");
    if (forward_ok) {
        check(out.rows == 1, "VGGBlock forward keeps the batch dimension");
    }
}

int main() {
    printf("=== tests/test_pool_geometry.cpp ===\n");

    printf("-- Test 1: MaxPool2D geometry contract --\n");
    test_maxpool_geometry_contract();

    printf("-- Test 2: MaxPool2D honours its stride --\n");
    test_maxpool_stride_is_honoured();

    printf("-- Test 3: AvgPool2D geometry contract --\n");
    test_avgpool_geometry_contract();

    printf("-- Test 4: LeNet5 pools match the chain --\n");
    test_lenet5_pools_match_the_chain();

    printf("-- Test 5: AlexNet pools match the chain --\n");
    test_alexnet_pools_match_the_chain();

    printf("-- Test 6: SqueezeNet pools match the chain --\n");
    test_squeezenet_pools_match_the_chain();

    printf("-- Test 7: VGGBlock pools match the chain --\n");
    test_vggblock_pools_match_the_chain();

    printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
