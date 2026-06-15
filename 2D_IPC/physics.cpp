#include "physics.h"
#include "spring_energy.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {
constexpr int kMagic = 0x32495043; // "2IPC"
constexpr int kVersion = 2;

std::string state_path(const std::string& dir, int frame) {
    std::ostringstream path;
    path << dir << "/state_" << std::setw(4) << std::setfill('0') << frame << ".bin";
    return path.str();
}

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
    for (const Vec2& value : values) {
        if (!write_value(out, value.x) || !write_value(out, value.y)) return false;
    }
    return true;
}

bool read_vec(std::ifstream& in, Vec& values) {
    int n = 0;
    if (!read_value(in, n) || n < 0) return false;
    values.resize(n);
    for (Vec2& value : values) {
        if (!read_value(in, value.x) || !read_value(in, value.y)) return false;
    }
    return true;
}
}

Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat,
                       const RefMesh& ref_mesh,
                       const std::vector<Pin>& pins,
                       const PinMap* pin_map,
                       double dt, double k_spring, const Vec2 &g_accel) {

    Vec2 xi = get_xi(x, i), xhi = get_xi(xhat, i);
    Vec2 gi{0.0, 0.0};
    const double mass = ref_mesh.mass[i];

    gi.x += mass * (xi.x - xhi.x);
    gi.y += mass * (xi.y - xhi.y);

    Vec2 gs = local_spring_grad(i, x, k_spring, ref_mesh);
    gi.x += dt * dt * gs.x;
    gi.y += dt * dt * gs.y;

    gi.x -= dt * dt * mass * g_accel.x;
    gi.y -= dt * dt * mass * g_accel.y;

    constexpr double k_pin = 5e6;

    const int pin_index = pin_map ? (*pin_map)[i] : -1;
    if (pin_index >= 0) {
        const Vec2 xpi = pins[pin_index].target_position;
        gi.x += dt * dt * k_pin * (xi.x - xpi.x);
        gi.y += dt * dt * k_pin * (xi.y - xpi.y);
    }

    return gi;
}

Mat2 local_hess_no_barrier(int i, const Vec &x,
                       const RefMesh& ref_mesh,
                       const std::vector<Pin>& pins,
                       const PinMap* pin_map,
                       double dt, double k_spring) {
    (void)pins;

    const double mass = ref_mesh.mass[i];
    Mat2 H{mass, 0, 0, mass};

    Mat2 Hs = local_spring_hess(i, x, k_spring, ref_mesh);
    H.a11 += dt * dt * Hs.a11;
    H.a12 += dt * dt * Hs.a12;
    H.a21 += dt * dt * Hs.a21;
    H.a22 += dt * dt * Hs.a22;

    constexpr double k_pin = 5e6;

    const int pin_index = pin_map ? (*pin_map)[i] : -1;
    if (pin_index >= 0) {
        H.a11 += dt * dt * k_pin;
        H.a22 += dt * dt * k_pin;
    }

    return H;
}

void serialize_state(const std::string& dir, int frame, const DeformedState& state) {
    const std::string filename = state_path(dir, frame);
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: cannot write checkpoint " << filename << "\n";
        return;
    }

    const int num_positions = static_cast<int>(state.deformed_positions.size());
    if (!write_value(out, kMagic) || !write_value(out, kVersion) ||
        !write_value(out, num_positions)) {
        std::cerr << "Error: failed to write checkpoint header " << filename << "\n";
        return;
    }

    if (!write_vec(out, state.deformed_positions) || !write_vec(out, state.velocities)) {
        std::cerr << "Error: failed to write checkpoint data " << filename << "\n";
    }
}

bool deserialize_state(const std::string& dir, int frame, DeformedState& state) {
    const std::string filename = state_path(dir, frame);
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
        version != kVersion || num_positions != static_cast<int>(state.deformed_positions.size())) {
        std::cerr << "Error: checkpoint " << filename << " does not match this 2D IPC scene\n";
        return false;
    }

    Vec positions;
    Vec velocities;
    if (!read_vec(in, positions) || !read_vec(in, velocities) ||
        static_cast<int>(positions.size()) != static_cast<int>(state.deformed_positions.size()) ||
        static_cast<int>(velocities.size()) != static_cast<int>(state.deformed_positions.size())) {
        std::cerr << "Error: checkpoint " << filename << " has incompatible state\n";
        return false;
    }

    state.deformed_positions = std::move(positions);
    state.velocities = std::move(velocities);
    return true;
}
