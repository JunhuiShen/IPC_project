#include "state_io.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {

constexpr std::array<char, 8> kGeneralizedStateMagic = {
    'R', 'B', 'S', 'T', 'A', 'T', 'E', '1'};

std::string state_filename(const std::string& dir, int frame) {
    std::ostringstream ss;
    ss << dir << "/state_" << std::setw(4) << std::setfill('0') << frame << ".bin";
    return ss.str();
}

}  // namespace

void serialize_state(const std::string& dir, int frame, const DeformedState& state) {
    std::ofstream out(state_filename(dir, frame), std::ios::binary);
    if (!out) { std::cerr << "Error: cannot write state file for frame " << frame << "\n"; return; }

    auto write_vec3 = [&](const std::vector<Vec3>& v) {
        uint64_t n = v.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& p : v) {
            double x = p.x(), y = p.y(), z = p.z();
            out.write(reinterpret_cast<const char*>(&x), sizeof(double));
            out.write(reinterpret_cast<const char*>(&y), sizeof(double));
            out.write(reinterpret_cast<const char*>(&z), sizeof(double));
        }
    };

    auto write_vec4 = [&](const std::vector<Vec4>& v) {
        const uint64_t n = v.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const Vec4& q : v) {
            for (int component = 0; component < 4; ++component) {
                const double value = q[component];
                out.write(
                    reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
    };

    // Keep the original two arrays at the front for compatibility with old
    // checkpoints, then append a tagged generalized rigid-body state block.
    write_vec3(state.deformed_positions);
    write_vec3(state.velocities);
    out.write(kGeneralizedStateMagic.data(), kGeneralizedStateMagic.size());
    write_vec3(state.x_coms);
    write_vec3(state.v_coms);
    write_vec4(state.orientations);
    write_vec3(state.omega);
}

bool deserialize_state(const std::string& dir, int frame, DeformedState& state) {
    std::ifstream in(state_filename(dir, frame), std::ios::binary);
    if (!in) { std::cerr << "Error: cannot read state file for frame " << frame << "\n"; return false; }

    auto read_vec3 = [&](std::vector<Vec3>& v) {
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!in)
            return false;
        v.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            double x, y, z;
            in.read(reinterpret_cast<char*>(&x), sizeof(double));
            in.read(reinterpret_cast<char*>(&y), sizeof(double));
            in.read(reinterpret_cast<char*>(&z), sizeof(double));
            if (!in)
                return false;
            v[i] = Vec3(x, y, z);
        }
        return true;
    };

    auto read_vec4 = [&](std::vector<Vec4>& v) {
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!in)
            return false;
        v.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            for (int component = 0; component < 4; ++component) {
                in.read(
                    reinterpret_cast<char*>(&v[i][component]),
                    sizeof(double));
                if (!in)
                    return false;
            }
        }
        return true;
    };

    if (!read_vec3(state.deformed_positions)
        || !read_vec3(state.velocities)) {
        return false;
    }

    std::array<char, kGeneralizedStateMagic.size()> magic{};
    in.read(magic.data(), magic.size());
    if (in.gcount() == 0 && in.eof()) {
        // Legacy checkpoints end after particle velocities. Preserve the
        // generalized arrays supplied by the scene builder.
        in.clear();
        return true;
    }
    if (!in || !std::equal(magic.begin(), magic.end(), kGeneralizedStateMagic.begin())) {
        return false;
    }

    std::vector<Vec3> x_coms;
    std::vector<Vec3> v_coms;
    std::vector<Vec4> orientations;
    std::vector<Vec3> omega;
    if (!read_vec3(x_coms) || !read_vec3(v_coms) || !read_vec4(orientations) || !read_vec3(omega)) {
        return false;
    }
    if (x_coms.size() != v_coms.size() || x_coms.size() != orientations.size() || x_coms.size() != omega.size()) {
        return false;
    }
    state.x_coms = std::move(x_coms);
    state.v_coms = std::move(v_coms);
    state.orientations = std::move(orientations);
    state.omega = std::move(omega);
    return true;
}
