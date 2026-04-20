#include "nn/utils/tokenizer.h"
#include "nn/layers/embedding.h"
#include "nn/layers/rnn.h"
// removed (via nn.h)
#include "nn/nn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

static const char* corpus = R"(
the quick brown fox jumps over the lazy dog
the cat sat on the mat
a dog is a mans best friend
machine learning is transforming the world
neural networks learn from data
deep learning enables artificial intelligence
natural language processing powers chatbots
computer vision recognizes objects
the quick red fox jumped over the lazy cat
a neural network has many layers
training models requires lots of data
 embeddings capture semantic meaning
word vectors represent words as numbers
transformers revolutionized sequence modeling
attention mechanisms focus on relevant parts
the weather is nice today
she sells seashells by the seashore
peter piper picked a peck of pickled peppers
how much wood would a woodchuck chuck
the early bird catches the worm
better late than never
practice makes perfect
time flies like an arrow
fruit flies like a banana
to be or not to be that is the question
all that glitters is not gold
a journey of a thousand miles begins with a single step
an apple a day keeps the doctor away
the pen is mightier than the sword
knowledge is power
learning never exhausts the mind
)";

std::string load_corpus() {
    std::string result = corpus;
    std::string cleaned;
    for (char c : result) {
        if (c == '\n' || c == '\r') cleaned += ' ';
        else cleaned += c;
    }
    return cleaned;
}

void prepare_data(const Tokenizer& tok, int seq_len,
                  std::vector<std::vector<int>>& X_ids,
                  std::vector<int>& y_ids) {
    std::istringstream iss(corpus);
    std::string word;
    std::vector<int> all_ids;
    while (iss >> word) {
        auto ids = tok.encode(word);
        all_ids.insert(all_ids.end(), ids.begin(), ids.end());
    }
    all_ids.push_back(tok.eos_id());

    for (size_t i = 0; i + seq_len < all_ids.size(); ++i) {
        std::vector<int> seq;
        for (int t = 0; t < seq_len; ++t)
            seq.push_back(all_ids[i + t]);
        X_ids.push_back(seq);
        y_ids.push_back(all_ids[i + seq_len]);
    }
}

Tensor seq_to_tensor(const std::vector<double>& seq) {
    return Tensor(std::vector<std::vector<double>>{seq});
}

Tensor onehot_label(int target_id, int vocab_size) {
    std::vector<double> row(vocab_size, 0.0);
    row[target_id] = 1.0;
    return Tensor(std::vector<std::vector<double>>{row});
}

int sample_token(const Tensor& logits, double temp = 1.0) {
    double max_val = logits[0][0];
    for (size_t i = 1; i < logits.cols; ++i)
        if (logits[0][i] > max_val) max_val = logits[0][i];

    double sum = 0.0;
    std::vector<double> probs(logits.cols);
    for (size_t i = 0; i < logits.cols; ++i) {
        probs[i] = std::exp((logits[0][i] - max_val) / temp);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double r = dis(gen);
    double cum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return (int)i;
    }
    return (int)logits.cols - 1;
}

int main() {
    std::cout << "=== Tokenization + Embedding Demo ===\n\n";

    std::cout << "[1] Training BPE tokenizer...\n";
    Tokenizer tok;
    tok.train(load_corpus(), 150);

    std::cout << "Tokenizer vocab: " << tok.vocab_size()
              << " (256 base + " << (tok.vocab_size()-256) << " merges)\n";

    std::cout << "\n[2] Tokenization examples:\n";
    const char* examples[] = { "neural", "learning", "network", "transformers", "embeddings" };
    for (auto word : examples) {
        auto ids = tok.encode(word);
        std::cout << "  \"" << word << "\" => [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i + 1 < ids.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    std::cout << "\n[3] Preparing next-token training data...\n";
    const int seq_len = 5;
    std::vector<std::vector<int>> X_ids;
    std::vector<int> y_ids;
    prepare_data(tok, seq_len, X_ids, y_ids);
    std::cout << "  " << X_ids.size() << " sequences, vocab=" << tok.vocab_size() << "\n";

    const int embed_dim = 32;
    const int hidden_size = 64;
    const int vocab = tok.vocab_size();
    const int epochs = 300;

    std::cout << "\n[4] Building model: Embed(" << vocab << "," << embed_dim
              << ") -> RNN(" << embed_dim << "," << hidden_size << "," << seq_len
              << ") -> Dense(" << hidden_size << "," << vocab << ")\n";

    Model model;
    model.add_layer(new Embedding(vocab, embed_dim));
    model.add_layer(new SimpleRNN(embed_dim, hidden_size, seq_len));
    model.add_layer(new Dense(hidden_size, vocab));

    std::cout << "\n[5] Training (SGD, LR=0.1, " << epochs << " epochs)...\n";

    std::vector<std::vector<double>> X_rows;
    for (const auto& ids : X_ids) {
        std::vector<double> row;
        for (int id : ids) row.push_back((double)id);
        X_rows.push_back(row);
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X_rows.size(); ++i) {
            Tensor sample = seq_to_tensor(X_rows[i]);
            Tensor label = onehot_label(y_ids[i], vocab);

            Tensor logits = model.forward(sample);
            Tensor probs = Softmax()(logits);
            double loss = Softmax::cross_entropy_loss(logits, label);
            total_loss += loss;

            Tensor grad = probs - label;
            model.backward(grad, 0.0);

            for (auto& layer : model.layers) {
                layer->update_weights(0.1);
                layer->zero_grad();
            }
        }
        double avg = total_loss / X_rows.size();
        if ((epoch + 1) % 60 == 0 || epoch == 0)
            std::cout << "  Epoch " << epoch + 1 << "/" << epochs << " - CE loss: " << avg << "\n";
    }

    std::cout << "\n[6] Nearest neighbors in embedding space:\n";
    Embedding* emb = dynamic_cast<Embedding*>(model.layers[0].get());
    Tensor table = emb->get_weights();

    auto cosine_sim = [&](int a, int b) -> double {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int d = 0; d < embed_dim; ++d) {
            dot += table[a][d] * table[b][d];
            na += table[a][d] * table[a][d];
            nb += table[b][d] * table[b][d];
        }
        return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8);
    };

    const char* test_words[] = { "neural", "learning", "dog", "cat" };
    for (auto word : test_words) {
        auto ids = tok.encode(word);
        if (ids.empty()) continue;
        int query_id = ids[0];
        std::vector<std::pair<double, int>> sims;
        for (int i = 0; i < vocab; ++i) {
            if (i == query_id || i == tok.eos_id()) continue;
            sims.emplace_back(cosine_sim(query_id, i), i);
        }
        std::sort(sims.begin(), sims.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cout << "  \"" << word << "\" (id=" << query_id << ") -> ";
        for (size_t k = 0; k < 5; ++k) {
            auto decoded = tok.decode({sims[k].second});
            std::cout << "\"" << decoded << "\" (" << sims[k].first << ")";
            if (k < 4) std::cout << ", ";
        }
        std::cout << "\n";
    }

    std::cout << "\n[7] Text generation (temperature sampling):\n";
    const char* prompts[] = { "neural networks learn", "machine learning is", "the pen is" };
    for (auto prompt : prompts) {
        auto ids = tok.encode(prompt);
        std::vector<int> history(ids);
        for (int step = 0; step < 15; ++step) {
            std::vector<int> input_seq;
            if ((int)history.size() >= seq_len) {
                for (int i = (int)history.size() - seq_len; i < (int)history.size(); ++i)
                    input_seq.push_back(history[i]);
            } else {
                input_seq = history;
            }
            while ((int)input_seq.size() < seq_len)
                input_seq.insert(input_seq.begin(), 0);

            std::vector<double> row;
            for (int id : input_seq) row.push_back((double)id);
            Tensor input_tensor = seq_to_tensor(row);

            Tensor logits = model.forward(input_tensor);

            int offset = (seq_len - 1) * vocab;
            Tensor last_logits(1, vocab);
            for (int i = 0; i < vocab; ++i)
                last_logits[0][i] = logits[0][offset + i];

            int next_token = sample_token(last_logits, 0.8);
            history.push_back(next_token);
            if (next_token == tok.eos_id()) break;
        }
        std::cout << "  \"" << prompt << "\" -> \"" << tok.decode(history) << "\"\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}