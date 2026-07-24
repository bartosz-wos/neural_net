#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "nn/utils/training_history.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static void check(const string& name, bool condition) {
    cout << "  [" << (condition ? "PASS" : "FAIL") << "] " << name << '\n';
    condition ? ++passed : ++failed;
}

template <typename Exception, typename Fn>
static bool throws_as(Fn&& fn) {
    try { fn(); } catch (const Exception&) { return true; } catch (...) {}
    return false;
}

static bool near(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

static string read_text(const string& path) {
    ifstream in(path);
    return string((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
}

static void test_empty_and_recording() {
    cout << "\n== empty state and recording ==\n";
    TrainingHistory h;
    check("starts empty", h.empty());
    check("starts size zero", h.size() == 0);
    check("records view starts empty", h.records().empty());
    check("latest on empty throws", throws_as<runtime_error>([&] { (void)h.latest(); }));
    check("at on empty throws", throws_as<out_of_range>([&] { (void)h.at(0); }));

    h.record(0, 1.25, 1.5, 0.01);
    h.record(1, 0.75, 0.8, 0.005);
    check("two records stored", h.size() == 2);
    check("first epoch preserved", h.at(0).epoch == 0);
    check("first train loss preserved", near(h.at(0).train_loss, 1.25));
    check("second val loss preserved", near(h.at(1).val_loss, 0.8));
    check("latest is final epoch", h.latest().epoch == 1);
    check("learning rate preserved", near(h.latest().learning_rate, 0.005));
}

static void test_validation_is_transactional() {
    cout << "\n== validation ==\n";
    TrainingHistory h;
    h.record(3, 1.0, 2.0, 0.1);
    const size_t before = h.size();
    check("duplicate epoch rejected", throws_as<invalid_argument>([&] { h.record(3, 0.9, 1.9, 0.1); }));
    check("decreasing epoch rejected", throws_as<invalid_argument>([&] { h.record(2, 0.9, 1.9, 0.1); }));
    check("NaN train loss rejected", throws_as<invalid_argument>([&] { h.record(4, numeric_limits<double>::quiet_NaN(), 1.0, 0.1); }));
    check("infinite val loss rejected", throws_as<invalid_argument>([&] { h.record(4, 1.0, numeric_limits<double>::infinity(), 0.1); }));
    check("negative learning rate rejected", throws_as<invalid_argument>([&] { h.record(4, 1.0, 1.0, -0.1); }));
    check("infinite learning rate rejected", throws_as<invalid_argument>([&] { h.record(4, 1.0, 1.0, numeric_limits<double>::infinity()); }));
    check("invalid records do not mutate history", h.size() == before);
}

static void test_best_epoch_queries() {
    cout << "\n== best epoch queries ==\n";
    TrainingHistory h;
    h.record(2, 0.8, 0.6, 0.1);
    h.record(4, 0.4, 0.7, 0.05);
    h.record(7, 0.4, 0.5, 0.01);

    check("minimum train loss keeps earliest tie", h.best_epoch(TrainingMetric::TRAIN_LOSS, HistoryMode::MINIMIZE).epoch == 4);
    check("minimum val loss", h.best_epoch(TrainingMetric::VAL_LOSS, HistoryMode::MINIMIZE).epoch == 7);
    check("maximum train loss", h.best_epoch(TrainingMetric::TRAIN_LOSS, HistoryMode::MAXIMIZE).epoch == 2);
    check("maximum val loss", h.best_epoch(TrainingMetric::VAL_LOSS, HistoryMode::MAXIMIZE).epoch == 4);
    check("best query on empty throws", throws_as<runtime_error>([] {
        TrainingHistory empty;
        (void)empty.best_epoch(TrainingMetric::VAL_LOSS);
    }));
}

static void test_callback_and_clear() {
    cout << "\n== callback and clear ==\n";
    TrainingHistory h;
    auto cb = h.callback(0.025);
    cb(0, 1.0, 1.2);
    cb(1, 0.7, 0.9);
    check("callback records epochs", h.size() == 2);
    check("callback captures learning rate", near(h.latest().learning_rate, 0.025));
    h.clear();
    check("clear empties records", h.empty());
    cb(0, 0.5, 0.6);
    check("callback remains usable after clear", h.size() == 1 && h.latest().epoch == 0);
}

static void test_csv_export() {
    cout << "\n== CSV export ==\n";
    TrainingHistory h;
    h.record(0, 1.0 / 3.0, 0.25, 0.01);
    h.record(2, 0.125, 0.2, 0.001);
    const string expected =
        "epoch,train_loss,val_loss,learning_rate\n"
        "0,0.33333333333333331,0.25,0.01\n"
        "2,0.125,0.20000000000000001,0.001\n";
    check("CSV string has stable schema and precision", h.to_csv() == expected);

    TrainingHistory empty;
    check("empty CSV contains header", empty.to_csv() == "epoch,train_loss,val_loss,learning_rate\n");

    const string path = "/tmp/neural_net_training_history.csv";
    h.save_csv(path);
    check("saved CSV matches in-memory export", read_text(path) == expected);
    remove(path.c_str());
    check("unwritable CSV path throws", throws_as<runtime_error>([&] {
        h.save_csv("/definitely/missing/directory/history.csv");
    }));
}

int main() {
    cout << "=== TrainingHistory Tests ===\n";
    test_empty_and_recording();
    test_validation_is_transactional();
    test_best_epoch_queries();
    test_callback_and_clear();
    test_csv_export();
    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
