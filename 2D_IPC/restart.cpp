#include "restart.h"
#include <fstream>
#include <iostream>
#include <utility>

namespace {
constexpr int kMagic = 0x32495043; // "2IPC"
constexpr int kVersion = 2;

template <typename T>
bool write_value(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(out);
}

template <typename T>
bool read_value(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

bool write_vec(std::ofstream& out, const Vec& values) {
    const int n = static_cast<int>(values.size());
    if (!write_value(out, n)) return false;
    out.write(reinterpret_cast<const char*>(values.data()), n * static_cast<int>(sizeof(double)));
    return static_cast<bool>(out);
}

bool read_vec(std::ifstream& in, Vec& values) {
    int n = 0;
    if (!read_value(in, n) || n < 0) return false;
    values.resize(n);
    in.read(reinterpret_cast<char*>(values.data()), n * static_cast<int>(sizeof(double)));
    return static_cast<bool>(in);
}
}

bool write_checkpoint(const std::string& filename, const State2D& state) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: cannot write checkpoint " << filename << "\n";
        return false;
    }

    const int num_positions = state.size();
    if (!write_value(out, kMagic) || !write_value(out, kVersion) ||
        !write_value(out, num_positions)) {
        return false;
    }

    return write_vec(out, state.x) && write_vec(out, state.v);
}

bool read_checkpoint(const std::string& filename, State2D& state) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: cannot read checkpoint " << filename << "\n";
        return false;
    }

    int magic = 0;
    int version = 0;
    int num_positions = 0;
    if (!read_value(in, magic) || !read_value(in, version) ||
        !read_value(in, num_positions) || magic != kMagic ||
        version != kVersion || num_positions != state.size()) {
        std::cerr << "Error: checkpoint " << filename << " does not match this 2D IPC scene\n";
        return false;
    }

    Vec x;
    Vec v;
    if (!read_vec(in, x) || !read_vec(in, v) ||
        static_cast<int>(x.size()) != 2 * state.size() ||
        static_cast<int>(v.size()) != 2 * state.size()) {
        std::cerr << "Error: checkpoint " << filename << " has incompatible state\n";
        return false;
    }

    state.x = std::move(x);
    state.v = std::move(v);
    return true;
}
