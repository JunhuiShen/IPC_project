#include "state_io.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

std::string state_filename(const std::string& dir, int frame) {
    std::ostringstream ss;
    ss << dir << "/state_" << std::setw(4) << std::setfill('0') << frame << ".bin";
    return ss.str();
}

}  // namespace

void serialize_state(const std::string& dir, int frame, const DeformedState& state) {
    std::ofstream out(state_filename(dir, frame), std::ios::binary);
    if (!out) { std::cerr << "Error: cannot write state file for frame " << frame << "\n"; return; }

    auto write_vec = [&](const std::vector<Vec3>& v) {
        uint64_t n = v.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& p : v) {
            double x = p.x(), y = p.y(), z = p.z();
            out.write(reinterpret_cast<const char*>(&x), sizeof(double));
            out.write(reinterpret_cast<const char*>(&y), sizeof(double));
            out.write(reinterpret_cast<const char*>(&z), sizeof(double));
        }
    };

    write_vec(state.deformed_positions);
    write_vec(state.velocities);
}

bool deserialize_state(const std::string& dir, int frame, DeformedState& state) {
    std::ifstream in(state_filename(dir, frame), std::ios::binary);
    if (!in) { std::cerr << "Error: cannot read state file for frame " << frame << "\n"; return false; }

    auto read_vec = [&](std::vector<Vec3>& v) {
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        v.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            double x, y, z;
            in.read(reinterpret_cast<char*>(&x), sizeof(double));
            in.read(reinterpret_cast<char*>(&y), sizeof(double));
            in.read(reinterpret_cast<char*>(&z), sizeof(double));
            v[i] = Vec3(x, y, z);
        }
    };

    read_vec(state.deformed_positions);
    read_vec(state.velocities);
    return in.good();
}
