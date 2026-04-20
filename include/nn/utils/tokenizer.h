#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

// Simple word-level tokenizer with frequency-based vocab + byte fallback
class Tokenizer {
public:
    Tokenizer() : next_id(0), eos_id_(256) {
        for (int i = 0; i < 256; ++i) {
            byte_to_id[std::string(1, (char)i)] = i;
            id_to_byte[i] = std::string(1, (char)i);
        }
        next_id = 256;
    }

    void train(const std::string& corpus, int vocab_size = 500, int min_freq = 1) {
        std::unordered_map<std::string, int> word_freq;
        std::string word;
        for (char c : corpus) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!word.empty()) { word_freq[word]++; word.clear(); }
            } else {
                word.push_back(c);
            }
        }
        if (!word.empty()) word_freq[word]++;

        std::vector<std::pair<int, std::string>> sorted;
        for (auto& w : word_freq)
            if (w.second >= min_freq) sorted.emplace_back(w.second, w.first);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());

        for (auto& p : sorted) {
            if ((int)vocab.size() >= vocab_size - 1) break;
            vocab[p.second] = next_id;
            id_to_token[next_id] = p.second;
            next_id++;
        }
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::string word;
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!word.empty()) {
                    auto it = vocab.find(word);
                    if (it != vocab.end()) ids.push_back(it->second);
                    else for (char wc : word) ids.push_back((unsigned char)wc);
                    word.clear();
                    ids.push_back(eos_id_);
                }
            } else {
                word.push_back(c);
            }
        }
        if (!word.empty()) {
            auto it = vocab.find(word);
            if (it != vocab.end()) ids.push_back(it->second);
            else for (char wc : word) ids.push_back((unsigned char)wc);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id == eos_id_) { result += ' '; continue; }
            if (id < 256) result += (char)id;
            else { auto it = id_to_token.find(id); if (it != id_to_token.end()) result += it->second; }
        }
        return result;
    }

    int vocab_size() const { return next_id; }
    int eos_id() const { return eos_id_; }
    int unk_id() const { return 0; }

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    std::unordered_map<std::string, int> byte_to_id;
    std::unordered_map<int, std::string> id_to_byte;
    int next_id;
    int eos_id_;
};

#endif