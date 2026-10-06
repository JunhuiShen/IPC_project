#include "example.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "rigid_body_ipc.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;

std::filesystem::path wrecking_ball_asset_directory(
    const std::string& data_directory) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    if (!data_directory.empty()) {
        candidates.push_back(fs::path(data_directory) / "wrecking_ball");
        candidates.push_back(fs::path(data_directory));
    }
    candidates.push_back(fs::path("example_obj") / "wrecking_ball");
    candidates.push_back(
        fs::path(__FILE__).parent_path() / "example_obj" / "wrecking_ball");

    std::error_code error;
    for (const fs::path& candidate : candidates) {
        const bool has_link = fs::is_regular_file(
            candidate / "link.obj", error);
        error.clear();
        const bool has_ball = fs::is_regular_file(
            candidate / "ball.obj", error);
        error.clear();
        if (has_link && has_ball)
            return candidate;
    }
    throw std::runtime_error(
        "Example 14 could not find link.obj and ball.obj. Run from the "
        "repository root or pass --datadir example_obj (or the "
        "wrecking_ball asset directory).");
}

void append_box_mesh(
    const Vec3& lo, const Vec3& hi,
    std::vector<Vec3>& x, std::vector<int>& tris) {
    const int base = static_cast<int>(x.size());
    x.push_back(Vec3(lo.x(), lo.y(), lo.z()));
    x.push_back(Vec3(hi.x(), lo.y(), lo.z()));
    x.push_back(Vec3(hi.x(), hi.y(), lo.z()));
    x.push_back(Vec3(lo.x(), hi.y(), lo.z()));
    x.push_back(Vec3(lo.x(), lo.y(), hi.z()));
    x.push_back(Vec3(hi.x(), lo.y(), hi.z()));
    x.push_back(Vec3(hi.x(), hi.y(), hi.z()));
    x.push_back(Vec3(lo.x(), hi.y(), hi.z()));

    static constexpr int box_tris[36] = {
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6,
        3, 0, 4, 3, 4, 7
    };
    for (int index : box_tris)
        tris.push_back(base + index);
}

int append_rigid_cube(
    const Vec3& center, const double edge_length, const double density,
    const Vec3& velocity, RefMesh& ref_mesh, DeformedState& state) {
    if (!center.allFinite() || !velocity.allFinite())
        throw std::invalid_argument("append_rigid_cube: inputs must be finite");
    if (!std::isfinite(edge_length) || edge_length <= 0.0)
        throw std::invalid_argument("append_rigid_cube: edge length must be positive and finite");
    if (!std::isfinite(density) || density <= 0.0)
        throw std::invalid_argument("append_rigid_cube: density must be positive and finite");
    if (state.deformed_positions.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max() - 8)) {
        throw std::overflow_error("append_rigid_cube: global node index exceeds int range");
    }

    const double half_extent = 0.5 * edge_length;
    std::vector<Vec3> positions;
    std::vector<int> local_tris;
    append_box_mesh(
        center - Vec3::Constant(half_extent),
        center + Vec3::Constant(half_extent), positions, local_tris);

    const int node_base = static_cast<int>(state.deformed_positions.size());
    ref_mesh.tris.reserve(ref_mesh.tris.size() + local_tris.size());
    for (const int local_node : local_tris)
        ref_mesh.tris.push_back(node_base + local_node);

    return create_rigid_body(
        positions, velocity, Vec4(1.0, 0.0, 0.0, 0.0), Vec3::Zero(),
        density * edge_length * edge_length * edge_length,
        ref_mesh, state);
}

// Replaces create_rigid_body's repository-wide equal-vertex mass-property
// approximation for one already appended, connected closed body. Rigid IPC
// integrates mass properties over the enclosed volume, so Example 14 applies
// this local correction without changing the behavior of any existing example.
void use_closed_volume_rigid_mass_properties(
    const int rigid_body, const std::size_t triangle_begin,
    const std::size_t triangle_end,
    RefMesh& ref_mesh, DeformedState& state) {
    if (rigid_body < 0
        || static_cast<std::size_t>(rigid_body)
            >= ref_mesh.total_mass.size()
        || triangle_begin >= triangle_end
        || triangle_end > ref_mesh.tris.size() / 3) {
        throw std::invalid_argument(
            "use_closed_volume_rigid_mass_properties: invalid body or triangle range");
    }

    const Vec3 reference = state.x_coms[
        static_cast<std::size_t>(rigid_body)];
    double signed_volume = 0.0;
    Vec3 signed_first_moment = Vec3::Zero();
    Mat33 signed_second_moment = Mat33::Zero();

    for (std::size_t triangle = triangle_begin;
         triangle < triangle_end; ++triangle) {
        const int* nodes = ref_mesh.tris.data() + 3 * triangle;
        for (int local = 0; local < 3; ++local) {
            if (nodes[local] < 0
                || static_cast<std::size_t>(nodes[local])
                    >= state.deformed_positions.size()
                || ref_mesh.node_to_rb[
                       static_cast<std::size_t>(nodes[local])]
                    != rigid_body) {
                throw std::invalid_argument(
                    "use_closed_volume_rigid_mass_properties: triangle ownership mismatch");
            }
        }

        const Vec3 a = state.deformed_positions[
            static_cast<std::size_t>(nodes[0])] - reference;
        const Vec3 b = state.deformed_positions[
            static_cast<std::size_t>(nodes[1])] - reference;
        const Vec3 c = state.deformed_positions[
            static_cast<std::size_t>(nodes[2])] - reference;
        const double tetra_volume = a.dot(b.cross(c)) / 6.0;
        const Vec3 sum = a + b + c;

        signed_volume += tetra_volume;
        signed_first_moment += 0.25 * tetra_volume * sum;
        signed_second_moment += (tetra_volume / 20.0) * (
            sum * sum.transpose()
            + a * a.transpose()
            + b * b.transpose()
            + c * c.transpose());
    }

    if (!std::isfinite(signed_volume)
        || std::abs(signed_volume) <= 1.0e-15
        || !signed_first_moment.allFinite()
        || !signed_second_moment.allFinite()) {
        throw std::invalid_argument(
            "use_closed_volume_rigid_mass_properties: invalid enclosed volume moments");
    }
    const double winding_sign = signed_volume >= 0.0 ? 1.0 : -1.0;
    const double volume = winding_sign * signed_volume;
    const Vec3 first_moment = winding_sign * signed_first_moment;
    const Mat33 second_moment = winding_sign * signed_second_moment;
    const Vec3 local_center = first_moment / volume;
    const Vec3 world_center = reference + local_center;
    Mat33 world_second_moment = second_moment
        - volume * local_center * local_center.transpose();
    world_second_moment = 0.5 * (
        world_second_moment + world_second_moment.transpose());

    const double total_mass = ref_mesh.total_mass[
        static_cast<std::size_t>(rigid_body)];
    const double density = total_mass / volume;
    world_second_moment *= density;

    const Vec4 orientation = quaternion_normalize(
        state.orientations[static_cast<std::size_t>(rigid_body)]);
    Mat33 rotation;
    for (int axis = 0; axis < 3; ++axis) {
        rotation.col(axis) = quaternion_rotate(
            orientation, Vec3::Unit(axis));
    }
    Mat33 body_second_moment = rotation.transpose()
        * world_second_moment * rotation;
    body_second_moment = 0.5 * (
        body_second_moment + body_second_moment.transpose());
    if (!body_second_moment.allFinite()) {
        throw std::invalid_argument(
            "use_closed_volume_rigid_mass_properties: non-finite body moment");
    }
    const Eigen::SelfAdjointEigenSolver<Mat33> eigensolver(
        body_second_moment, Eigen::EigenvaluesOnly);
    const double eigenvalue_tolerance = 64.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(body_second_moment.trace()));
    if (eigensolver.info() != Eigen::Success
        || eigensolver.eigenvalues().minCoeff() <= eigenvalue_tolerance) {
        throw std::invalid_argument(
            "use_closed_volume_rigid_mass_properties: invalid body moment");
    }

    state.x_coms[static_cast<std::size_t>(rigid_body)] = world_center;
    ref_mesh.I_hat[static_cast<std::size_t>(rigid_body)] =
        body_second_moment;

    const std::vector<int>& body_nodes = ref_mesh.rb_nodes[
        static_cast<std::size_t>(rigid_body)];
    std::vector<Vec3>& reference_positions = ref_mesh.ref_positions[
        static_cast<std::size_t>(rigid_body)];
    if (body_nodes.empty()
        || reference_positions.size() != body_nodes.size()) {
        throw std::invalid_argument(
            "use_closed_volume_rigid_mass_properties: invalid body node storage");
    }
    const double proxy_mass = total_mass
        / static_cast<double>(body_nodes.size());
    const Vec3 v_com = state.v_coms[
        static_cast<std::size_t>(rigid_body)];
    const Vec3 omega = state.omega[
        static_cast<std::size_t>(rigid_body)];
    for (std::size_t local = 0; local < body_nodes.size(); ++local) {
        const int node = body_nodes[local];
        const Vec3 world_offset = state.deformed_positions[
            static_cast<std::size_t>(node)] - world_center;
        reference_positions[local] = quaternion_inverse_rotate(
            orientation, world_offset);
        ref_mesh.mass[static_cast<std::size_t>(node)] = proxy_mass;
        state.velocities[static_cast<std::size_t>(node)] =
            v_com + omega.cross(world_offset);
    }
}

// Rotate `p` about the +x line through `axis_point` by `theta`.
Vec3 rotate_about_x_axis(const Vec3& p, const Vec3& axis_point, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double dy = p.y() - axis_point.y();
    const double dz = p.z() - axis_point.z();
    return Vec3(p.x(),
                axis_point.y() + c * dy - s * dz,
                axis_point.z() + s * dy + c * dz);
}

// Rotate `p` about the +y line through `axis_point` by `theta`.
Vec3 rotate_about_y_axis(const Vec3& p, const Vec3& axis_point, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double dx = p.x() - axis_point.x();
    const double dz = p.z() - axis_point.z();
    return Vec3(axis_point.x() + c * dx + s * dz,
                p.y(),
                axis_point.z() - s * dx + c * dz);
}

Mat33 rotation_about_y_matrix(double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    Mat33 rotation;
    rotation << c, 0.0, s,
                0.0, 1.0, 0.0,
                -s, 0.0, c;
    return rotation;
}

SDFMaterialPose material_pose_about_y_axis(
    const Vec3& axis_point, double theta) {
    SDFMaterialPose pose;
    pose.rotation = rotation_about_y_matrix(theta);
    pose.translation = axis_point - pose.rotation * axis_point;
    return pose;
}

// Rotate `p` about the +z line through `axis_point` by `theta`.
// Positive theta lifts +x toward +y; in example 3 we pass a negative theta so
// the +x edge of the catcher drops toward the sphere.
Vec3 rotate_about_z_axis(const Vec3& p, const Vec3& axis_point, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double dx = p.x() - axis_point.x();
    const double dy = p.y() - axis_point.y();
    return Vec3(axis_point.x() + c * dx - s * dy,
                axis_point.y() + s * dx + c * dy,
                p.z());
}

// Trapezoidal phase: ramp-up over t_ramp, hold at omega for t_steady, ramp-down.
double trapezoid_theta(double s, double omega, double t_ramp, double t_steady) {
    if (s <= 0.0)        return 0.0;
    if (s <= t_ramp)     return 0.5 * omega * s * s / t_ramp;
    const double s1 = s - t_ramp;
    if (s1 <= t_steady)  return 0.5 * omega * t_ramp + omega * s1;
    const double s2 = s1 - t_steady;
    if (s2 < t_ramp)
        return 0.5 * omega * t_ramp + omega * t_steady
             + omega * s2 - 0.5 * omega * s2 * s2 / t_ramp;
    return omega * (t_ramp + t_steady);
}

// Pin-target angle vs wall time. max_abs_theta=0 means open-ended ramp+steady;
// untwist=true mirrors the forward trapezoid back to 0 after a t_hold dwell.
double effective_theta(double omega, double t, double t_settle, double t_ramp,
                       double max_abs_theta = 0.0,
                       bool untwist = false,
                       double t_hold = 0.0) {
    if (t <= t_settle) return 0.0;

    const double abs_omega = std::abs(omega);
    const double sgn       = (omega >= 0.0) ? 1.0 : -1.0;
    const double s         = t - t_settle;

    if (max_abs_theta <= 0.0 || abs_omega <= 0.0) {
        if (t_ramp <= 0.0)       return omega * s;
        if (s >= t_ramp)         return omega * (s - 0.5 * t_ramp);
        return 0.5 * omega * s * s / t_ramp;
    }

    // Steady duration sized so accel + steady + decel hits max_abs_theta exactly
    // (pure triangle if the two ramps alone would already overshoot).
    const double t_steady = std::max(0.0, max_abs_theta / abs_omega - t_ramp);
    const double t_fwd    = 2.0 * t_ramp + t_steady;

    if (s <= t_fwd)              return sgn * trapezoid_theta(s, abs_omega, t_ramp, t_steady);
    if (!untwist)                return sgn * max_abs_theta;

    const double s2 = s - t_fwd;
    if (s2 <= t_hold)            return sgn * max_abs_theta;

    const double s3 = s2 - t_hold;
    if (s3 <= t_fwd)             return sgn * (max_abs_theta - trapezoid_theta(s3, abs_omega, t_ramp, t_steady));

    return 0.0;
}
} // namespace


// ---------------------------------------------------------------------------
// Example 1: twisting cloth
// ---------------------------------------------------------------------------
// Square cloth with both short edges clamped, edges counter-rotate about the
// +x axis at twist_rate Hz.
void build_twisting_cloth_example(const IPCArgs3D& args,
                                  RefMesh& ref_mesh,
                                  DeformedState& state,
                                  std::vector<Vec2>& X,
                                  std::vector<Pin>& pins,
                                  TwistSpec& spec) {
    clear_model(ref_mesh, state, X, pins);

    const int    nx     = args.twist_nx;
    const int    ny     = args.twist_ny;
    const double width  = args.twist_size;
    const double height = args.twist_size;
    const double y0     = args.sheet_y;

    const Vec3 origin(-0.5 * width, y0, -0.5 * height);
    const int base = build_square_mesh(ref_mesh, state, X, nx, ny, width, height, origin);

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    // Each side spins at +/- omega; relative rate is 2*omega.
    const double omega = (kTwoPi * args.twist_rate) / 2.0;

    spec = TwistSpec{};
    spec.axis_point  = Vec3(0.0, y0, 0.0);
    spec.omega_left  = -omega;
    spec.omega_right =  omega;

    const int npin = ny + 1;
    spec.left_pin_indices.reserve(npin);
    spec.right_pin_indices.reserve(npin);
    spec.left_initial_targets.reserve(npin);
    spec.right_initial_targets.reserve(npin);

    // build_square_mesh stores grid (i, j) at base + j * (nx + 1) + i.
    for (int j = 0; j <= ny; ++j) {
        const int v_left  = base + j * (nx + 1) + 0;
        const int v_right = base + j * (nx + 1) + nx;

        spec.left_pin_indices.push_back(static_cast<int>(pins.size()));
        append_pin(pins, v_left, state.deformed_positions);
        spec.left_initial_targets.push_back(pins.back().target_position);

        spec.right_pin_indices.push_back(static_cast<int>(pins.size()));
        append_pin(pins, v_right, state.deformed_positions);
        spec.right_initial_targets.push_back(pins.back().target_position);
    }
}

void update_twist_pins(std::vector<Pin>& pins, const TwistSpec& spec, double t) {
    const double theta_left  = spec.omega_left  * t;
    const double theta_right = spec.omega_right * t;

    const int n_left = static_cast<int>(spec.left_pin_indices.size());
    for (int k = 0; k < n_left; ++k) {
        pins[spec.left_pin_indices[k]].target_position =
            rotate_about_x_axis(spec.left_initial_targets[k], spec.axis_point, theta_left);
    }
    const int n_right = static_cast<int>(spec.right_pin_indices.size());
    for (int k = 0; k < n_right; ++k) {
        pins[spec.right_pin_indices[k]].target_position =
            rotate_about_x_axis(spec.right_initial_targets[k], spec.axis_point, theta_right);
    }
}


// ---------------------------------------------------------------------------
// Example 2: two-cylinder twist
// ---------------------------------------------------------------------------
// N closed-loop cloth strips wrap two horizontal cylinders. Both cylinders
// counter-rotate about +y, dragging the wrap rows via pin springs and twisting
// the strips together in the gap. Pin targets, visual mesh, and SDF axes all
// yaw about the same +y line so the wrap pin (orbiting at radius pin_r > r)
// never crosses the rotating SDF surface.
void build_two_cylinder_twist_example(const IPCArgs3D& args,
                                      RefMesh& ref_mesh,
                                      DeformedState& state,
                                      std::vector<Vec2>& X,
                                      std::vector<Pin>& pins,
                                      SimParams& params,
                                      std::vector<Vec3>& static_x,
                                      std::vector<int>&  static_tris,
                                      CylinderTwistSpec& spec) {
    clear_model(ref_mesh, state, X, pins);
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();

    const int    n_strips = std::max(1, args.tcyl_n_strips);
    const int    nx       = args.tcyl_nx;
    const int    ny       = args.tcyl_ny;
    const double strip_w  = args.tcyl_strip_w;
    const double H        = 0.5 * args.tcyl_cloth_h;       // half y-distance between cyl axes
    const double r        = args.tcyl_radius;
    const double omega    = kTwoPi * args.tcyl_twist_rate;

    const Vec3 top_center(0.0,  H, 0.0);
    const Vec3 bot_center(0.0, -H, 0.0);

    // Two infinite-cylinder SDFs, axes initially along +x. update_cylinder_sdfs
    // yaws them per substep alongside the pin update; substep co-rotation is
    // what stops the pin from sitting inside a lagging SDF mid-frame.
    params.sdf_cylinders.push_back(CylinderSDF{ top_center, Vec3::UnitX(), r });
    params.sdf_cylinders.push_back(CylinderSDF{ bot_center, Vec3::UnitX(), r });

    // Each strip is a flat belt wrapping both cylinders, parameterised by arc
    // length s ∈ [0, loop_L):
    //   [0,        s_top_end)   top  wrap, back→front, length π·pin_r
    //   [s_top_end, s_front_end) front drop, y from +H to -H at z=+pin_r
    //   [s_front_end, s_bot_end) bot  wrap, front→back, length π·pin_r
    //   [s_bot_end, loop_L)     back drop, y from -H to +H at z=-pin_r
    // pin_r is set just outside r so the initial polyline doesn't touch the
    // cylinder mesh. With default eps_sdf = 0.002 this also coincides with the
    // SDF's force-free rest distance, so the wrap pin and the SDF agree.
    const double pin_r        = r + 0.002;
    const double wrap_len     = kPi * pin_r;
    const double drop_len     = 2.0 * H;
    const double loop_L       = 2.0 * wrap_len + 2.0 * drop_len;
    const double s_top_end    = wrap_len;
    const double s_front_end  = wrap_len + drop_len;
    const double s_bot_end    = 2.0 * wrap_len + drop_len;

    auto loop_position = [&](double s) -> Vec3 {
        if (s <= s_top_end) {
            const double phi = kPi - s / pin_r;
            return Vec3(0.0, H + pin_r * std::sin(phi), pin_r * std::cos(phi));
        }
        if (s <= s_front_end) {
            return Vec3(0.0, H - (s - s_top_end), pin_r);
        }
        if (s <= s_bot_end) {
            const double phi = -(s - s_front_end) / pin_r;
            return Vec3(0.0, -H + pin_r * std::sin(phi), pin_r * std::cos(phi));
        }
        return Vec3(0.0, -H + (s - s_bot_end), -pin_r);
    };

    // Visual cylinder mesh (export only). build_cylinder_mesh emits +z aligned;
    // the (x,y,z) → (z,y,x) swap below rotates onto +x to match the SDF.
    const double r_visual = std::max(0.001, r - args.tcyl_visual_shrink);
    auto append_x_axis_cylinder = [&](const Vec3& center) {
        RefMesh        s_ref;
        DeformedState  s_state;
        std::vector<Vec2> s_X;
        build_cylinder_mesh(s_ref, s_state, s_X, args.tcyl_nu, r_visual, args.tcyl_length, Vec3::Zero());
        const int base_v = static_cast<int>(static_x.size());
        for (const Vec3& p : s_state.deformed_positions) {
            static_x.push_back(Vec3(p.z() + center.x(),
                                    p.y() + center.y(),
                                    p.x() + center.z()));
        }
        for (int t : s_ref.tris) static_tris.push_back(base_v + t);
    };
    const int top_v_begin = 0;
    append_x_axis_cylinder(top_center);
    const int top_v_end = static_cast<int>(static_x.size());
    append_x_axis_cylinder(bot_center);
    const int bot_v_end = static_cast<int>(static_x.size());

    spec = CylinderTwistSpec{};
    spec.top_axis_point = top_center;
    spec.bot_axis_point = bot_center;
    spec.omega_top      =  omega;
    spec.omega_bot      = -omega;
    spec.t_settle       = std::max(0.0, args.tcyl_settle_time);
    spec.t_ramp         = std::max(0.0, args.tcyl_ramp_time);
    spec.max_abs_theta  = std::max(0.0, kTwoPi * args.tcyl_max_turn);
    spec.untwist        = args.tcyl_untwist;
    spec.t_hold         = std::max(0.0, args.tcyl_hold_time);
    spec.static_x_rest  = static_x;
    spec.top_v_begin    = top_v_begin;
    spec.top_v_end      = top_v_end;
    spec.bot_v_begin    = top_v_end;
    spec.bot_v_end      = bot_v_end;

    // j=ny and j=0 sample the same loop position (s=loop_L wraps to s=0). The
    // mesh has them as separate vertices, so we nudge j=ny by -z by seam_offset
    // to keep them apart — coincident barrier pairs blow up the gradient.
    const double seam_offset = std::max(1.5 * params.d_hat, 0.005);
    const double span = args.tcyl_strip_span_z;

    for (int strip = 0; strip < n_strips; ++strip) {
        const double x_center = (n_strips == 1)
            ? 0.0
            : (-0.5 * span + (strip + 0.5) * (span / n_strips));

        const Vec3 build_origin(-0.5 * strip_w, 0.0, 0.0);
        const int  base = build_square_mesh(ref_mesh, state, X,
                                            nx, ny, strip_w, loop_L, build_origin);

        // Remap each panel row to its 3D position along the loop.
        for (int j = 0; j <= ny; ++j) {
            const Vec3 p = loop_position((static_cast<double>(j) / ny) * loop_L);
            for (int i = 0; i <= nx; ++i) {
                const double dx = (static_cast<double>(i) / nx - 0.5) * strip_w;
                state.deformed_positions[base + j * (nx + 1) + i] =
                    Vec3(x_center + dx, p.y(), p.z());
            }
        }

        // Pin every wrap-row vertex; j=ny is folded back to s=0 with the seam
        // offset already applied. Pin targets are the rotated initial positions
        // (see update_cylinder_twist_pins), which yaw about +y at radius pin_r.
        for (int j = 0; j <= ny; ++j) {
            const double s = (j == ny) ? 0.0 : (static_cast<double>(j) / ny) * loop_L;
            const bool on_top_wrap = (s <= s_top_end);
            const bool on_bot_wrap = (s >= s_front_end && s <= s_bot_end);
            if (!on_top_wrap && !on_bot_wrap) continue;

            for (int i = 0; i <= nx; ++i) {
                const int v = base + j * (nx + 1) + i;
                if (j == ny) state.deformed_positions[v].z() -= seam_offset;

                auto& pin_indices = on_top_wrap ? spec.top_pin_indices     : spec.bot_pin_indices;
                auto& targets     = on_top_wrap ? spec.top_initial_targets : spec.bot_initial_targets;
                pin_indices.push_back(static_cast<int>(pins.size()));
                append_pin(pins, v, state.deformed_positions);
                targets.push_back(pins.back().target_position);
            }
        }
    }

    // Re-initialise hinges with the wrapped 3D positions so bar_theta captures
    // the curved rest pose; otherwise bending would push the cloth flat.
    ref_mesh.initialize(X, state.deformed_positions);

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
}

void update_cylinder_sdfs(SimParams& params,
                          const CylinderTwistSpec& spec, double t) {
    if (params.sdf_cylinders.size() < 2) return;
    const double previous_t = t - params.dt();
    const double theta_top = effective_theta(spec.omega_top, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    const double theta_bot = effective_theta(spec.omega_bot, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    const double previous_theta_top = effective_theta(
        spec.omega_top, previous_t, spec.t_settle, spec.t_ramp,
        spec.max_abs_theta, spec.untwist, spec.t_hold);
    const double previous_theta_bot = effective_theta(
        spec.omega_bot, previous_t, spec.t_settle, spec.t_ramp,
        spec.max_abs_theta, spec.untwist, spec.t_hold);
    // Yaw the SDF axis about +y (axis_point at origin → pure direction rotate)
    // by the same theta that drives the pins, so pin and SDF surface co-rotate.
    params.sdf_cylinders[0].point = spec.top_axis_point;
    params.sdf_cylinders[0].axis  = rotate_about_y_axis(Vec3::UnitX(), Vec3::Zero(), theta_top);
    params.sdf_cylinders[0].material_motion.previous =
        material_pose_about_y_axis(spec.top_axis_point, previous_theta_top);
    params.sdf_cylinders[0].material_motion.current =
        material_pose_about_y_axis(spec.top_axis_point, theta_top);
    params.sdf_cylinders[1].point = spec.bot_axis_point;
    params.sdf_cylinders[1].axis  = rotate_about_y_axis(Vec3::UnitX(), Vec3::Zero(), theta_bot);
    params.sdf_cylinders[1].material_motion.previous =
        material_pose_about_y_axis(spec.bot_axis_point, previous_theta_bot);
    params.sdf_cylinders[1].material_motion.current =
        material_pose_about_y_axis(spec.bot_axis_point, theta_bot);
}

void update_cylinder_twist_pins(std::vector<Pin>& pins,
                                const CylinderTwistSpec& spec, double t) {
    const double theta_top = effective_theta(spec.omega_top, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    const double theta_bot = effective_theta(spec.omega_bot, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    const int n_top = static_cast<int>(spec.top_pin_indices.size());
    for (int k = 0; k < n_top; ++k) {
        pins[spec.top_pin_indices[k]].target_position =
            rotate_about_y_axis(spec.top_initial_targets[k], spec.top_axis_point, theta_top);
    }
    const int n_bot = static_cast<int>(spec.bot_pin_indices.size());
    for (int k = 0; k < n_bot; ++k) {
        pins[spec.bot_pin_indices[k]].target_position =
            rotate_about_y_axis(spec.bot_initial_targets[k], spec.bot_axis_point, theta_bot);
    }
}

void update_cylinder_visuals(std::vector<Vec3>& static_x,
                             const CylinderTwistSpec& spec,
                             double t) {
    const double theta_top = effective_theta(spec.omega_top, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    const double theta_bot = effective_theta(spec.omega_bot, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    for (int i = spec.top_v_begin; i < spec.top_v_end; ++i) {
        static_x[i] = rotate_about_y_axis(spec.static_x_rest[i], spec.top_axis_point, theta_top);
    }
    for (int i = spec.bot_v_begin; i < spec.bot_v_end; ++i) {
        static_x[i] = rotate_about_y_axis(spec.static_x_rest[i], spec.bot_axis_point, theta_bot);
    }
}


// ---------------------------------------------------------------------------
// Example 3: twist-untwist
// ---------------------------------------------------------------------------
// Rectangular cloth (tu_width x tu_size) draped under one cylinder
// (axis +x, at (0, sheet_y, 0)). Pre-pose: back drop -> bottom-wrap semicircle
// -> front drop, so j=0 and j=ny rows end at y=corner_y on either side of the
// cylinder. Both top edges (j=0 and j=ny rows in full) are statically pinned
// as stretchy clamping bars; the bottom-wrap rows are also pinned, and their
// targets co-rotate with the SDF axis about +y, twisting the cloth between
// rotating wrap and fixed bars.
void build_twist_untwist_example(const IPCArgs3D& args,
                                 RefMesh& ref_mesh,
                                 DeformedState& state,
                                 std::vector<Vec2>& X,
                                 std::vector<Pin>& pins,
                                 SimParams& params,
                                 std::vector<Vec3>& static_x,
                                 std::vector<int>&  static_tris,
                                 TwistUntwistSpec& spec) {
    clear_model(ref_mesh, state, X, pins);
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();

    const int    nx       = args.tu_nx;
    const int    ny       = args.tu_ny;
    const double strip_w  = args.tu_width;         // cloth x-width (along cyl axis)
    const double cloth_L  = args.tu_size;          // cloth arc length (drops + bottom wrap)
    const double cyl_y    = args.sheet_y;          // cylinder axis sits at sheet_y
    const double r        = args.tu_cyl_radius;
    const double pin_r    = r + 0.002;             // 2mm SDF rest offset, same as example 2
    const Vec3   cyl_pt(0.0, cyl_y, 0.0);

    // Arc partition: bottom wrap (pi*pin_r) plus two equal drops use the rest
    // of the cloth length. Floor at 0.05 m guards against a cloth too short
    // for even the wrap.
    const double wrap_len = kPi * pin_r;
    const double drop_len = std::max((cloth_L - wrap_len) * 0.5, 0.05);
    const double total_arc = 2.0 * drop_len + wrap_len;
    const double corner_y  = cyl_y + drop_len;

    auto arc_position = [&](double s) -> Vec3 {
        if (s <= drop_len) {
            return Vec3(0.0, corner_y - s, -pin_r);
        }
        if (s <= drop_len + wrap_len) {
            // phi sweeps -pi/2 (back tangent) -> 0 (bottom) -> +pi/2 (front).
            const double phi = -kPi / 2.0 + (s - drop_len) / pin_r;
            return Vec3(0.0,
                        cyl_y - pin_r * std::cos(phi),
                        pin_r * std::sin(phi));
        }
        const double s_in = s - drop_len - wrap_len;
        return Vec3(0.0, cyl_y + s_in, +pin_r);
    };

    // build_square_mesh lays out (i,j) at base + j*(nx+1) + i; we overwrite
    // the deformed positions below to bend the flat grid onto the arc.
    const Vec3 build_origin(-0.5 * strip_w, 0.0, 0.0);
    const int  base = build_square_mesh(ref_mesh, state, X,
                                        nx, ny, strip_w, total_arc, build_origin);

    for (int j = 0; j <= ny; ++j) {
        const Vec3 p = arc_position((static_cast<double>(j) / ny) * total_arc);
        for (int i = 0; i <= nx; ++i) {
            const double dx = (static_cast<double>(i) / nx - 0.5) * strip_w;
            state.deformed_positions[base + j * (nx + 1) + i] =
                Vec3(dx, p.y(), p.z());
        }
    }
    // Re-init hinges from the curved pose so bending doesn't try to flatten
    // the wrap back out (same fix as example 2).
    ref_mesh.initialize(X, state.deformed_positions);
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    spec = TwistUntwistSpec{};
    spec.cyl_axis_point = cyl_pt;
    spec.omega          = kTwoPi * args.tu_twist_rate;
    spec.t_settle       = std::max(0.0, args.tu_settle_time);
    spec.t_ramp         = std::max(0.0, args.tu_ramp_time);
    spec.max_abs_theta  = std::max(0.0, kTwoPi * args.tu_max_turn);
    spec.untwist        = args.tu_untwist;
    spec.t_hold         = std::max(0.0, args.tu_hold_time);

    // Pinning matches the SIGGRAPH knot demo's "stretchy clamping bars":
    // both top edges (j=0 and j=ny) are pinned along their full length, and
    // the bottom-wrap rows co-rotate with the cylinder.
    for (int j = 0; j <= ny; ++j) {
        const double s = (static_cast<double>(j) / ny) * total_arc;
        const bool on_wrap = (s > drop_len) && (s < drop_len + wrap_len);

        if (on_wrap) {
            for (int i = 0; i <= nx; ++i) {
                const int v = base + j * (nx + 1) + i;
                spec.wrap_pin_indices.push_back(static_cast<int>(pins.size()));
                append_pin(pins, v, state.deformed_positions);
                spec.wrap_initial_targets.push_back(pins.back().target_position);
            }
        } else if (j == 0 || j == ny) {
            for (int i = 0; i <= nx; ++i) {
                const int v = base + j * (nx + 1) + i;
                spec.end_pin_indices.push_back(static_cast<int>(pins.size()));
                append_pin(pins, v, state.deformed_positions);
                spec.end_initial_targets.push_back(pins.back().target_position);
            }
        }
    }

    // SDF axis starts +x; update_twist_untwist_sdf yaws it about +y per substep.
    spec.cyl_sdf_index = static_cast<int>(params.sdf_cylinders.size());
    params.sdf_cylinders.push_back(CylinderSDF{cyl_pt, Vec3::UnitX(), r});

    // Visual cylinder: build_cylinder_mesh emits +z-aligned at origin; swap
    // (x,y,z) -> (z,y,x) to align with +x, then translate to cyl_pt. Radius
    // tracks the cloth's rest radius (pin_r), not the SDF radius (r), so the
    // wrap sits flush against the visible surface with no gap.
    const double r_visual = std::max(0.001, pin_r - args.tu_visual_shrink);
    {
        RefMesh           s_ref;
        DeformedState     s_state;
        std::vector<Vec2> s_X;
        build_cylinder_mesh(s_ref, s_state, s_X, args.tu_cyl_nu,
                            r_visual, args.tu_cyl_length, Vec3::Zero());
        spec.visual_v_begin = static_cast<int>(static_x.size());
        const int base_v = spec.visual_v_begin;
        for (const Vec3& p : s_state.deformed_positions) {
            static_x.push_back(Vec3(p.z() + cyl_pt.x(),
                                    p.y() + cyl_pt.y(),
                                    p.x() + cyl_pt.z()));
        }
        for (int t : s_ref.tris) static_tris.push_back(base_v + t);
        spec.visual_v_end = static_cast<int>(static_x.size());
        spec.visual_v_rest.assign(static_x.begin() + spec.visual_v_begin,
                                  static_x.begin() + spec.visual_v_end);
    }
}

void update_twist_untwist_pins(std::vector<Pin>& pins,
                               const TwistUntwistSpec& spec, double t) {
    // Re-snap the static top-bar pins each step so a restart from any frame
    // recovers their targets.
    const int n_end = static_cast<int>(spec.end_pin_indices.size());
    for (int k = 0; k < n_end; ++k) {
        pins[spec.end_pin_indices[k]].target_position = spec.end_initial_targets[k];
    }
    const double theta = effective_theta(spec.omega, t, spec.t_settle, spec.t_ramp,
                                         spec.max_abs_theta, spec.untwist, spec.t_hold);
    const int n_wrap = static_cast<int>(spec.wrap_pin_indices.size());
    for (int k = 0; k < n_wrap; ++k) {
        pins[spec.wrap_pin_indices[k]].target_position =
            rotate_about_y_axis(spec.wrap_initial_targets[k], spec.cyl_axis_point, theta);
    }
}

void update_twist_untwist_sdf(SimParams& params,
                              const TwistUntwistSpec& spec, double t) {
    if (spec.cyl_sdf_index < 0 ||
        spec.cyl_sdf_index >= static_cast<int>(params.sdf_cylinders.size())) return;
    // Same theta as the wrap pins: per-substep so the pin never sits inside
    // a lagging SDF mid-step.
    const double theta = effective_theta(spec.omega, t, spec.t_settle, spec.t_ramp,
                                         spec.max_abs_theta, spec.untwist, spec.t_hold);
    const double previous_theta = effective_theta(
        spec.omega, t - params.dt(), spec.t_settle, spec.t_ramp,
        spec.max_abs_theta, spec.untwist, spec.t_hold);
    params.sdf_cylinders[spec.cyl_sdf_index].point = spec.cyl_axis_point;
    params.sdf_cylinders[spec.cyl_sdf_index].axis  =
        rotate_about_y_axis(Vec3::UnitX(), Vec3::Zero(), theta);
    params.sdf_cylinders[spec.cyl_sdf_index].material_motion.previous =
        material_pose_about_y_axis(spec.cyl_axis_point, previous_theta);
    params.sdf_cylinders[spec.cyl_sdf_index].material_motion.current =
        material_pose_about_y_axis(spec.cyl_axis_point, theta);
}

void build_avatar_clothing_example(const IPCArgs3D& args,
                                   RefMesh& ref_mesh,
                                   DeformedState& state,
                                   std::vector<Pin>& /*pins*/,
                                   SimParams& params,
                                   std::vector<Vec3>& static_x,
                                   std::vector<int>&  static_tris) {
    //load_obj_mesh(args.datadir + "/body_0000.obj", static_x, static_tris);
    load_obj_mesh(args.datadir + "/dress_0000.obj", ref_mesh, state,
                  /*scale=*/1.0, /*origin=*/Vec3::Zero());

    for(int i=0;i<state.deformed_positions.size();i++)
        state.deformed_positions[i](1)+=.75;

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    params.sdf_planes.push_back({Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)});
}

void update_twist_untwist_visual(std::vector<Vec3>& static_x,
                                 const TwistUntwistSpec& spec, double t) {
    const double theta = effective_theta(spec.omega, t, spec.t_settle, spec.t_ramp,
                                         spec.max_abs_theta, spec.untwist, spec.t_hold);
    for (int i = spec.visual_v_begin; i < spec.visual_v_end; ++i) {
        static_x[i] = rotate_about_y_axis(spec.visual_v_rest[i - spec.visual_v_begin],
                                          spec.cyl_axis_point, theta);
    }
}


// ---------------------------------------------------------------------------
// Example 5: freely rotating rigid tennis racket
// ---------------------------------------------------------------------------
// Command line: ./build/3D_sim --example 5 --num_frames 500 --substeps 30 --tol_abs 1e-12 --tol_rel 1e-10 --outdir racket_output 
void build_rotating_tennis_racket_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params) {
    clear_model(ref_mesh, state, X, pins);

    params.gravity = Vec3::Zero();
    params.d_hat = 0.0;
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd = false;
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;

    std::vector<Vec3> x;
    std::vector<int> tris;

    // Elliptical annular head, lying initially in the x-y plane.
    constexpr int head_segments = 64;
    constexpr double head_center_y = 0.25;
    constexpr double outer_rx = 0.36;
    constexpr double outer_ry = 0.48;
    constexpr double inner_rx = 0.29;
    constexpr double inner_ry = 0.39;
    constexpr double half_thickness = 0.018;

    for (int i = 0; i < head_segments; ++i) {
        const double theta = kTwoPi * static_cast<double>(i)
            / static_cast<double>(head_segments);
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        x.push_back(Vec3(outer_rx * c, head_center_y + outer_ry * s,
                         half_thickness));
        x.push_back(Vec3(outer_rx * c, head_center_y + outer_ry * s,
                         -half_thickness));
        x.push_back(Vec3(inner_rx * c, head_center_y + inner_ry * s,
                         half_thickness));
        x.push_back(Vec3(inner_rx * c, head_center_y + inner_ry * s,
                         -half_thickness));
    }

    for (int i = 0; i < head_segments; ++i) {
        const int j = (i + 1) % head_segments;
        const int oi_f = 4 * i + 0;
        const int oi_b = 4 * i + 1;
        const int ii_f = 4 * i + 2;
        const int ii_b = 4 * i + 3;
        const int oj_f = 4 * j + 0;
        const int oj_b = 4 * j + 1;
        const int ij_f = 4 * j + 2;
        const int ij_b = 4 * j + 3;

        const int patch[24] = {
            oi_f, oj_f, ii_f, oj_f, ij_f, ii_f,
            oi_b, ii_b, oj_b, oj_b, ii_b, ij_b,
            oi_f, oi_b, oj_f, oj_f, oi_b, oj_b,
            ii_f, ij_f, ii_b, ij_f, ij_b, ii_b
        };
        tris.insert(tris.end(), std::begin(patch), std::end(patch));
    }

    // Handle and throat. The handle overlaps the bottom of the frame so the
    // exported surface reads as one racket even though rigid kinematics do not
    // require a topologically connected mesh.
    append_box_mesh(
        Vec3(-0.055, -0.98, -0.025),
        Vec3( 0.055, -0.16,  0.025), x, tris);
    append_box_mesh(
        Vec3(-0.13, -0.25, -0.022),
        Vec3( 0.13, -0.12,  0.022), x, tris);

    // Thin box-shaped strings clipped to the inner ellipse.
    constexpr double string_half_width = 0.003;
    constexpr double string_half_thickness = 0.003;
    for (int k = -3; k <= 3; ++k) {
        const double string_x = 0.065 * static_cast<double>(k);
        const double ratio = string_x / inner_rx;
        const double half_y = inner_ry * std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
        append_box_mesh(
            Vec3(string_x - string_half_width,
                 head_center_y - half_y,
                 -string_half_thickness),
            Vec3(string_x + string_half_width,
                 head_center_y + half_y,
                  string_half_thickness), x, tris);
    }
    for (int k = -4; k <= 4; ++k) {
        const double string_y = 0.075 * static_cast<double>(k);
        const double ratio = string_y / inner_ry;
        const double half_x = inner_rx * std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
        append_box_mesh(
            Vec3(-half_x,
                 head_center_y + string_y - string_half_width,
                 -string_half_thickness),
            Vec3( half_x,
                 head_center_y + string_y + string_half_width,
                  string_half_thickness), x, tris);
    }

    ref_mesh.tris = tris;
    X.reserve(x.size());
    for (const Vec3& position : x)
        X.push_back(position.head<2>());

    create_rigid_body(
        x, Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0),
        Vec3(double(5), double(0.02), double(0.01)),
        0.30, ref_mesh, state);
}


// ---------------------------------------------------------------------------
// Example 6: freely rotating space tool
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 6 --num_frames 2000 --substeps 30 --tol_abs 1e-12 --tol_rel 1e-10 --outdir space_tool_output 
void build_rotating_space_tool_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params) {
    clear_model(ref_mesh, state, X, pins);

    params.gravity = Vec3::Zero();
    params.d_hat = 0.0;
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd = false;
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;

    // The rotational residual is small in physical units. These tolerances
    // ensure the torque-free angular-velocity update is not skipped.
    params.tol_abs = 1.0e-12;
    params.tol_rel = 1.0e-8;

    std::vector<Vec3> x;
    std::vector<int> tris;

    // A vertical tool body lying initially in the x-y plane. A short handle
    // protrudes from the right side near its middle, matching a "|-" profile.
    append_box_mesh(
        Vec3(-0.080, -0.50, -0.060),
        Vec3( 0.080,  0.50,  0.060), x, tris); // thick vertical body
    append_box_mesh(
        Vec3(0.070, -0.040, -0.040),
        Vec3(0.38,  0.040,  0.040), x, tris);  // thinner side handle

    ref_mesh.tris = tris;
    X.reserve(x.size());
    for (const Vec3& position : x)
        X.push_back(position.head<2>());

    // The asymmetric "|-" geometry rotates the in-plane principal axes away
    // from the coordinate axes. Compute them from the same equal nodal masses
    // used by create_rigid_body, then spin mostly around the intermediate one.
    constexpr double total_mass = 0.60;
    const double nodal_mass = total_mass / static_cast<double>(x.size());
    Vec3 x_com = Vec3::Zero();
    for (const Vec3& position : x)
        x_com += nodal_mass * position;
    x_com /= total_mass;

    std::vector<Vec3> centered_positions;
    centered_positions.reserve(x.size());
    for (const Vec3& position : x)
        centered_positions.push_back(position - x_com);
    const std::vector<double> masses(x.size(), nodal_mass);
    const Mat33 second_moment =
        body_second_moment(masses, centered_positions);
    const Mat33 physical_inertia =
        second_moment.trace() * Mat33::Identity() - second_moment;
    const Eigen::SelfAdjointEigenSolver<Mat33> eigensolver(physical_inertia);
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error(
            "build_rotating_space_tool_example: inertia eigensolve failed");
    }

    const Mat33 principal_axes = eigensolver.eigenvectors();
    const Vec3 initial_omega =
        5.0 * principal_axes.col(1)
        + 0.04 * principal_axes.col(0)
        + 0.02 * principal_axes.col(2);

    create_rigid_body(
        x, Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0),
        initial_omega, total_mass, ref_mesh, state);
}


// ---------------------------------------------------------------------------
// Example 7: twenty rigid polygonal prisms initialized in a static vertical stack
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 7 --num_frames 100 --substeps 10 --d_hat 0.001 --eps_sdf 0.0002 --rigid_density 25 --gy 0 --outdir twenty_polygon_static_stack_output --format obj
void build_twenty_rigid_polygon_static_stack_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris) {
    clear_model(ref_mesh, state, X, pins);
    static_x.clear();
    static_tris.clear();

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = args.k_sdf;
    params.eps_sdf = args.eps_sdf;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.sdf_planes.push_back(
        {Vec3::Zero(), Vec3::UnitY()});
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;

    constexpr int polygon_count = 10;
    constexpr double radius = 0.28;
    const double density = params.rigid_density;
    constexpr double thickness = 0.12;
    // Each prism is laid flat: material z (the extrusion direction) maps to
    // world y, so its large polygonal caps form the horizontal contact faces.
    // Place the surfaces just outside the active SDF/barrier ranges. With zero
    // gravity and zero initial velocities, the assembled tower stays at rest.
    constexpr double clearance_margin = 1.0e-4;
    const double ground_clearance =
        std::max(params.eps_sdf, 0.0) + clearance_margin;
    const double interbody_clearance =
        std::max(params.d_hat, 0.0) + clearance_margin;
    const double lowest_center_y =
        0.5 * thickness + ground_clearance;
    const double center_spacing =
        thickness + interbody_clearance;
    const double flat_half_angle = 0.25 * kPi;
    const Vec4 flat_orientation(
        std::cos(flat_half_angle), std::sin(flat_half_angle),
        0.0, 0.0);

    for (int polygon = 0; polygon < polygon_count; ++polygon) {
        const double center_y =
            lowest_center_y + center_spacing * polygon;
        append_rigid_polygon(
            6, state, ref_mesh,
            Vec3(0.0, center_y, 0.0),
            radius, density, thickness,
            Vec3::Zero(), flat_orientation, Vec3::Zero());
    }

    // Flat ground
    static_x = {
        Vec3(-1.5, 0.0, -1.5),
        Vec3( 1.5, 0.0, -1.5),
        Vec3( 1.5, 0.0,  1.5),
        Vec3(-1.5, 0.0,  1.5),
    };
    static_tris = {
        0, 2, 1,
        0, 3, 2,
    };
}


// ---------------------------------------------------------------------------
// Example 8: fifty small rigid polygonal prisms falling onto a
// four-corner-pinned rectangular cloth
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 8 --num_frames 200 --substeps 10 --max_substep_iters 20 --fixed_iters --outdir fifty_polygons_on_pinned_cloth_output --format obj
void build_fifty_rigid_polygons_drop_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params) {
    clear_model(ref_mesh, state, X, pins);

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;
    params.use_ogc = false;
    params.use_ogc_solver = false;

    // build_square_mesh places its grid in the world x-z plane. Using unequal
    // width and depth makes this a rectangular cloth centered at the origin.
    constexpr int cloth_nx = 100;
    constexpr int cloth_nz = 100;
    constexpr double cloth_width = 4.0;
    constexpr double cloth_depth = 4.0;
    constexpr double cloth_height = 1.2;
    const Vec3 cloth_origin(
        -0.5 * cloth_width, cloth_height, -0.5 * cloth_depth);
    const int cloth_base = build_square_mesh(
        ref_mesh, state, X, cloth_nx, cloth_nz,
        cloth_width, cloth_depth, cloth_origin);
    state.velocities.assign(
        state.deformed_positions.size(), Vec3::Zero());

    const auto cloth_node = [cloth_base](int i, int j) {
        return cloth_base + j * (cloth_nx + 1) + i;
    };
    append_pin(pins, cloth_node(0, 0), state.deformed_positions);
    append_pin(pins, cloth_node(cloth_nx, 0), state.deformed_positions);
    append_pin(pins, cloth_node(0, cloth_nz), state.deformed_positions);
    append_pin(pins, cloth_node(cloth_nx, cloth_nz),state.deformed_positions);

    // The polygon helper extrudes along material z. A -90-degree rotation
    // about x turns that extrusion into the world y direction, so every prism
    // lands flat on the horizontal cloth. A world-y yaw gives each footprint
    // a different in-plane orientation without tilting it.
    const double half_angle = 0.25 * kPi;
    const Vec4 flat_orientation(
        std::cos(half_angle), -std::sin(half_angle), 0.0, 0.0);

    constexpr int polygon_count = 50;
    constexpr int columns = 10;
    constexpr double radius = 0.10;
    const double density = params.rigid_density;
    constexpr double thickness = 0.06;

    for (int polygon = 0; polygon < polygon_count; ++polygon) {
        const int row = polygon / columns;
        const int column = polygon % columns;
        const double yaw = polygon * kPi / 17.0;
        const Vec4 yaw_orientation(
            std::cos(0.5 * yaw), 0.0,
            std::sin(0.5 * yaw), 0.0);
        const Vec4 orientation = quaternion_normalize(
            quaternion_multiply(yaw_orientation, flat_orientation));
        const Vec3 center(
            (column - 4.5) * 0.34,
            2.00 + 0.06 * ((column + 2 * row) % 5),
            (row - 2) * 0.34);

        // Use every regular prism from a triangle through a dodecagon five
        // times. The 0.34 spacing leaves a gap between radius-0.10 bodies.
        append_rigid_polygon(
            3 + polygon % 10, state, ref_mesh, center,
            radius, density, thickness,
            Vec3::Zero(), orientation, Vec3::Zero());
    }

    ref_mesh.build_deformable_nodes();
}

// ---------------------------------------------------------------------------
// Example 9: Bunny-solid / Spot-solid / rigid-cube / rigid-gear cycles,
// repeated twice in one vertical stack above a pinned cloth
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 9 --datadir example_obj --num_frames 200 --fps 30 --substeps 20 --max_substep_iters 600 --fixed_iters --E 1.25e9 --nu 0.25 --thickness 0.001 --solid_E 1.25e5 --solid_nu 0.25 --d_hat 0.019 --k_barrier 1000 --outdir multi_physics_output --format obj
void build_two_bunny_spot_cube_gear_cycles_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params) {
    clear_model(ref_mesh, state, X, pins);

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;
    params.use_ogc = false;
    params.use_ogc_solver = false;

    // Cloth must be initialized before solid and rigid collision triangles so
    // shell rest data remains the leading triangle prefix.
    constexpr int cloth_nx = 30;
    constexpr int cloth_nz = 30;
    constexpr double cloth_width = 4.0;
    constexpr double cloth_depth = 4.0;
    constexpr double cloth_height = 1.2;
    const int cloth_base = build_square_mesh(
        ref_mesh, state, X, cloth_nx, cloth_nz,
        cloth_width, cloth_depth,
        Vec3(-0.5 * cloth_width, cloth_height,
             -0.5 * cloth_depth));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    const auto cloth_node = [cloth_base](const int i, const int j) {
        return cloth_base + j * (cloth_nx + 1) + i;
    };
    for (int j = 0; j <= cloth_nz; ++j) {
        append_pin(pins, cloth_node(0, j), state.deformed_positions);
        append_pin(
            pins, cloth_node(cloth_nx, j), state.deformed_positions);
    }

    // Every body shares one vertical axis. The physical bottom-to-top order is
    // Bunny, Spot, cube, gear, repeated twice. The spacing keeps every adjacent
    // pair initially disjoint so the objects reach the cloth one by one.
    constexpr int copies = 2;
    constexpr double solid_max_extent = 0.26;
    constexpr double rigid_max_extent = 0.14;
    constexpr double initial_cloth_clearance = 0.02;
    constexpr double first_center_y =
        cloth_height + 0.5 * solid_max_extent + initial_cloth_clearance;
    constexpr double vertical_spacing = 0.34;
    const auto stack_center = [=](const int stack_index) {
        return Vec3(
            0.0,
            first_center_y + vertical_spacing * stack_index,
            0.0);
    };
    constexpr const char* bunny_node_filename =
        "example_obj/bunny_coarse/bunny_2000f.1.node";
    constexpr const char* bunny_element_filename =
        "example_obj/bunny_coarse/bunny_2000f.1.ele";
    for (int copy = 0; copy < copies; ++copy) {
        append_normalized_tetgen_solid(
            bunny_node_filename, bunny_element_filename,
            state, ref_mesh, stack_center(4 * copy), solid_max_extent,
            params.solid_density,
            /*zero_based_index=*/true);
    }

    constexpr const char* spot_node_filename =
        "example_obj/spot/spot_2000f.1.node";
    constexpr const char* spot_element_filename =
        "example_obj/spot/spot_2000f.1.ele";
    for (int copy = 0; copy < copies; ++copy) {
        append_normalized_tetgen_solid(
            spot_node_filename, spot_element_filename,
            state, ref_mesh, stack_center(4 * copy + 1), solid_max_extent,
            params.solid_density,
            /*zero_based_index=*/true);
    }

    for (int copy = 0; copy < copies; ++copy) {
        append_rigid_cube(
            stack_center(4 * copy + 2), rigid_max_extent,
            params.rigid_density, Vec3::Zero(), ref_mesh, state);
    }

    // The gear OBJ is extruded along material z. Rotate that axis onto world
    // y so both gears begin flat, then vary only their in-plane yaw.
    constexpr const char* gear_filename =
        "example_obj/gear_z18_coarse.obj";
    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);
    for (int gear = 0; gear < copies; ++gear) {
        const double yaw = static_cast<double>(gear) * kPi / 8.0;
        const Vec4 yaw_orientation(
            std::cos(0.5 * yaw), 0.0, std::sin(0.5 * yaw), 0.0);
        const Vec4 orientation = quaternion_normalize(
            quaternion_multiply(yaw_orientation, flat_orientation));
        append_normalized_obj_rigid_body(
            gear_filename, state, ref_mesh,
            stack_center(4 * gear + 3),
            rigid_max_extent, params.rigid_density,
            Vec3::Zero(), orientation, Vec3::Zero());
    }

    ref_mesh.build_deformable_nodes();

    // Respect the global IPC discretization requirement automatically rather
    // than making the default --d_hat reject these detailed surface meshes.
    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (std::size_t triangle = 0; triangle < ref_mesh.tris.size() / 3;
         ++triangle) {
        const int* tri = ref_mesh.tris.data() + 3 * triangle;
        for (int local = 0; local < 3; ++local) {
            const Vec3& a = state.deformed_positions[
                static_cast<std::size_t>(tri[local])];
            const Vec3& b = state.deformed_positions[
                static_cast<std::size_t>(tri[(local + 1) % 3])];
            minimum_surface_edge = std::min(
                minimum_surface_edge, (b - a).norm());
        }
    }
    if (params.d_hat > 0.0 && std::isfinite(minimum_surface_edge)) {
        params.d_hat = std::min(
            params.d_hat, 0.45 * minimum_surface_edge);
    }
}

// ---------------------------------------------------------------------------
// Example 10: a dynamic threaded bolt falling into a fixed threaded nut
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 10 --num_frames 200 --substeps 20 --max_substep_iters 10 --fixed_iters --outdir bolt_into_fixed_nut_output --format obj
void build_dynamic_bolt_into_fixed_nut_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params) {
    clear_model(ref_mesh, state, X, pins);

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;
    params.use_ogc = false;
    params.use_ogc_solver = false;

    constexpr const char* bolt_filename =
        "example_obj/bolt_and_nut/bolt_coarse_bolt.obj";
    constexpr const char* nut_filename =
        "example_obj/bolt_and_nut/bolt_coarse_nut.obj";

    // Both meshes were authored in one coordinate system with a 3-unit
    // thread pitch and a +z thread axis. Give them one common scale; normalizing
    // the two assets independently would change their radial clearance and the
    // threads would no longer mate.
    constexpr double source_to_world_scale = 0.30 / 58.0;
    constexpr double bolt_source_max_extent = 58.0;
    constexpr double nut_source_max_extent = 34.64102;
    constexpr double bolt_target_max_extent =
        source_to_world_scale * bolt_source_max_extent;
    constexpr double nut_target_max_extent =
        source_to_world_scale * nut_source_max_extent;

    // Rotate material +z onto world +y. The bolt's source-z=0 tip then points
    // downward, toward the nut, without changing their shared helical phase.
    const Vec4 upright_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);
    constexpr double nut_center_y = 0.35;
    constexpr double initial_insertion_source = 14.0;
    // The bolt bottom and nut top are 37 source units apart when their AABB
    // centers coincide. Insert 14 of the nut's 16 source-unit height, leaving
    // the tip 2 source units above the nut bottom. This matches the deeply
    // engaged starting configuration of the rigid-IPC bolt benchmark.
    constexpr double bolt_center_y =
        nut_center_y
        + source_to_world_scale * (37.0 - initial_insertion_source);
    // The resulting relative source-axis translation is 12 units: exactly
    // four complete 3-unit pitches. The two meshes therefore keep their
    // authored thread phase with no compensating yaw. Like the reference
    // benchmark, the bolt starts at rest; gravity and thread-normal contact
    // generate its rotation.

    // Append the nut first so rigid body 0 is the fixed collision obstacle.
    // RigidBodyUpdateMode::None suppresses both generalized-coordinate updates
    // while retaining all of its proxy triangles in broad phase and IPC.
    append_normalized_obj_rigid_body(
        nut_filename, state, ref_mesh,
        Vec3(0.0, nut_center_y, 0.0), nut_target_max_extent,
        params.rigid_density, Vec3::Zero(), upright_orientation,
        Vec3::Zero(), RigidBodyUpdateMode::None);
    append_normalized_obj_rigid_body(
        bolt_filename, state, ref_mesh,
        Vec3(0.0, bolt_center_y, 0.0), bolt_target_max_extent,
        params.rigid_density, Vec3::Zero(), upright_orientation,
        Vec3::Zero(), RigidBodyUpdateMode::TranslationAndOrientation);

    ref_mesh.build_deformable_nodes();

    // The authored surface edge length is much smaller than the default
    // activation distance. Clamp to less than half the shortest surface edge,
    // while preserving a smaller CLI value.
    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (std::size_t triangle = 0; triangle < ref_mesh.tris.size() / 3;
         ++triangle) {
        const int* tri = ref_mesh.tris.data() + 3 * triangle;
        for (int local = 0; local < 3; ++local) {
            const Vec3& a = state.deformed_positions[
                static_cast<std::size_t>(tri[local])];
            const Vec3& b = state.deformed_positions[
                static_cast<std::size_t>(tri[(local + 1) % 3])];
            minimum_surface_edge = std::min(
                minimum_surface_edge, (b - a).norm());
        }
    }
    if (params.d_hat > 0.0 && std::isfinite(minimum_surface_edge)) {
        params.d_hat = std::min(
            params.d_hat, 0.45 * minimum_surface_edge);
    }
}

// ---------------------------------------------------------------------------
// Example 11: four Bunny / Spot / cube / gear rows, each with one elevated
// object above another, falling onto a pinned cloth
// ---------------------------------------------------------------------------
// Smaller substep/iteration budgets can suppress the cubes' bounce.
/* command line:
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
./build/3D_sim \
  --example 11 --num_frames 200 --fps 30 \
  --substeps 15 --max_substep_iters 25 --fixed_iters \
  --E 1.25e9 --nu 0.25 --thickness 0.001 \
  --solid_E 1.25e5 --solid_nu 0.25 \
  --d_hat 0.019 --k_barrier 1000 \
  --friction_coefficient 0 --use_ccd true \
  --node_box_update_count 10 \
  --use_parallel true \
  --use_basic_experimental true --use_simd true \
  --write_substeps false --format obj \
  --outdir outputs/example11_general_v2 \
  --use_colored_ccd_guess true --colored_ccd_guess_iters 10
*/
void build_four_bunny_spot_cube_gear_rows_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params) {
    clear_model(ref_mesh, state, X, pins);

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;
    params.use_ogc = false;
    params.use_ogc_solver = false;

    // Build and initialize the cloth before appending solid and rigid boundary
    // triangles, preserving the shell rest-data prefix used by the solver.
    constexpr int cloth_nx = 30;
    constexpr int cloth_nz = 30;
    constexpr double cloth_width = 4.0;
    constexpr double cloth_depth = 4.0;
    constexpr double cloth_height = 1.2;
    const int cloth_base = build_square_mesh(
        ref_mesh, state, X, cloth_nx, cloth_nz,
        cloth_width, cloth_depth,
        Vec3(-0.5 * cloth_width, cloth_height,
             -0.5 * cloth_depth));
    state.velocities.assign(
        state.deformed_positions.size(), Vec3::Zero());

    const auto cloth_node = [cloth_base](const int i, const int j) {
        return cloth_base + j * (cloth_nx + 1) + i;
    };
    for (int j = 0; j <= cloth_nz; ++j) {
        append_pin(
            pins, cloth_node(0, j), state.deformed_positions);
        append_pin(
            pins, cloth_node(cloth_nx, j),
            state.deformed_positions);
    }

    // Pack four collision-free rows containing one Bunny, Spot, cube, and
    // gear each into three occupied columns per row. One column is a pair
    // with an upper body that can fall onto the lower body as the cloth slows
    // it down. Keep authored orientations, scales and velocities unchanged.
    constexpr int rows = 4;
    constexpr double object_center_y = 1.75;
    constexpr double solid_max_extent = 0.44;
    constexpr double rigid_max_extent = 0.22;
    constexpr double object_gap = 0.01;
    constexpr double row_spacing = 0.45;
    enum BodyType {
        Bunny = 0, Spot = 1, Cube = 2, Gear = 3, BodyTypeCount = 4};
    static constexpr int row_order[rows][BodyTypeCount] = {
        {Bunny, Cube, Gear, Spot},
        {Gear, Spot, Bunny, Cube},
        {Spot, Gear, Cube, Bunny},
        {Cube, Bunny, Spot, Gear},
    };
    static constexpr int upper_body[rows] = {Cube, Gear, Bunny, Spot};
    static constexpr int lower_body[rows] = {Bunny, Spot, Cube, Gear};
    // A solid's half extent is at most 0.22 and a rigid body's at most 0.11,
    // leaving at least 0.02 of initial vertical AABB clearance for each pair.
    constexpr double upper_body_lift = 0.35;

    // Type-aware packing follows the normalized production x AABBs. Spot's
    // source x/z extent ratio is 0.9425986 / 1.716426; Bunny, cube, and gear
    // all use their requested maximum extent along x. Give a stacked column
    // the larger body's width, remove the lifted body's old slot, and leave
    // 10 mm between occupied column AABBs. Spot has the largest z extent
    // (0.44 m), so the unchanged row spacing also leaves 10 mm between rows.
    constexpr double spot_x_extent =
        solid_max_extent * 0.9425986 / 1.716426;
    static constexpr double type_x_extent[BodyTypeCount] = {
        solid_max_extent, spot_x_extent,
        rigid_max_extent, rigid_max_extent};
    double object_center_x[rows][BodyTypeCount] = {};
    for (int row = 0; row < rows; ++row) {
        const auto column_width = [&](const int type) {
            return type == lower_body[row]
                ? std::max(type_x_extent[type], type_x_extent[upper_body[row]])
                : type_x_extent[type];
        };
        double packed_row_width = (BodyTypeCount - 2) * object_gap;
        for (int type = 0; type < BodyTypeCount; ++type) {
            if (type != upper_body[row]) packed_row_width += column_width(type);
        }
        double cursor = -0.5 * packed_row_width;
        for (int slot = 0; slot < BodyTypeCount; ++slot) {
            const int type = row_order[row][slot];
            if (type == upper_body[row]) continue;
            const double width = column_width(type);
            object_center_x[row][type] =
                cursor + 0.5 * width;
            cursor += width + object_gap;
        }
        object_center_x[row][upper_body[row]] = object_center_x[row][lower_body[row]];
    }

    const Vec3 drop_velocity(0.0, -0.75, 0.0);
    const Vec4 identity_orientation(1.0, 0.0, 0.0, 0.0);
    const auto object_center = [&](const int type, const int row) {
        const bool elevated = type == upper_body[row];
        return Vec3(
            object_center_x[row][type],
            object_center_y + (elevated ? upper_body_lift : 0.0),
            (static_cast<double>(row) - 1.5) * row_spacing);
    };

    constexpr const char* bunny_node_filename =
        "example_obj/bunny_coarse/bunny_2000f.1.node";
    constexpr const char* bunny_element_filename =
        "example_obj/bunny_coarse/bunny_2000f.1.ele";
    for (int row = 0; row < rows; ++row) {
        const int body_base = append_normalized_tetgen_solid(
            bunny_node_filename, bunny_element_filename,
            state, ref_mesh, object_center(Bunny, row),
            solid_max_extent, params.solid_density,
            /*zero_based_index=*/true);
        for (std::size_t node = static_cast<std::size_t>(body_base);
             node < state.velocities.size(); ++node) {
            state.velocities[node] = drop_velocity;
        }
    }

    constexpr const char* spot_node_filename =
        "example_obj/spot/spot_2000f.1.node";
    constexpr const char* spot_element_filename =
        "example_obj/spot/spot_2000f.1.ele";
    for (int row = 0; row < rows; ++row) {
        const int body_base = append_normalized_tetgen_solid(
            spot_node_filename, spot_element_filename,
            state, ref_mesh, object_center(Spot, row),
            solid_max_extent, params.solid_density,
            /*zero_based_index=*/true);
        for (std::size_t node = static_cast<std::size_t>(body_base);
             node < state.velocities.size(); ++node) {
            state.velocities[node] = drop_velocity;
        }
    }

    for (int row = 0; row < rows; ++row) {
        append_rigid_cube(
            object_center(Cube, row), rigid_max_extent,
            params.rigid_density, drop_velocity, ref_mesh, state);
    }

    constexpr const char* gear_filename =
        "example_obj/gear_z18_coarse.obj";
    for (int row = 0; row < rows; ++row) {
        append_normalized_obj_rigid_body(
            gear_filename, state, ref_mesh,
            object_center(Gear, row), rigid_max_extent,
            params.rigid_density, drop_velocity, identity_orientation,
            Vec3::Zero());
    }

    ref_mesh.build_deformable_nodes();

    // The imported solids and gear are much more finely tessellated than the
    // cloth. Keep the IPC activation distance strictly below the global
    // half-edge bound while respecting any smaller value supplied by the
    // caller.
    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (std::size_t triangle = 0; triangle < ref_mesh.tris.size() / 3;
         ++triangle) {
        const int* tri = ref_mesh.tris.data() + 3 * triangle;
        for (int local = 0; local < 3; ++local) {
            const Vec3& a = state.deformed_positions[
                static_cast<std::size_t>(tri[local])];
            const Vec3& b = state.deformed_positions[
                static_cast<std::size_t>(tri[(local + 1) % 3])];
            minimum_surface_edge = std::min(
                minimum_surface_edge, (b - a).norm());
        }
    }
    if (params.d_hat > 0.0 && std::isfinite(minimum_surface_edge)) {
        params.d_hat = std::min(
            params.d_hat, 0.45 * minimum_surface_edge);
    }
}

// ---------------------------------------------------------------------------
// Example 12: a pinned cloth roll unrolling down an SDF ramp
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 12 --num_frames 200 --substeps 20 --max_substep_iters 80 --fixed_iters --kB 0.0025 --friction_coefficient 0.1 --friction_velocity_epsilon 0.01 --outdir rolled_cloth_on_steep_ramp_output_new --format obj
// the drift direction switches sharply between 0.00225 and 0.0025
void build_cloth_unrolling_down_fixed_ramp_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris) {
    clear_model(ref_mesh, state, X, pins);
    static_x.clear();
    static_tris.clear();

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = args.k_sdf;
    params.eps_sdf = args.eps_sdf;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;
    params.use_ogc = false;
    params.use_ogc_solver = false;

    constexpr int cloth_nx = 24;
    // Keep enough samples around the enlarged roll that its piecewise-linear
    // layers remain more than the 5 mm d_hat cap apart. With only 140 rows,
    // chordal shortcuts reduce the nominal 6.5 mm pitch to a 4.89 mm gap.
    constexpr int cloth_ny = 160;
    constexpr double cloth_width = 0.90;
    constexpr double ramp_width = 1.40;
    // Match the 2.4 m horizontal run for a 45-degree incline. The previous
    // 1.5 m height produced a much shallower roughly 32-degree ramp.
    constexpr double ramp_height = 2.40;
    constexpr double ramp_back_z = -1.20;
    constexpr double ramp_front_z = 1.20;
    constexpr int leader_rows = 8;
    // The 10 cm outer radius stores 3.81608 m of material without changing
    // the tight 6.5 mm layer pitch. That is 0.42196 m longer than the finite
    // 3.39411 m ramp, so the released sheet can visibly pass its toe.
    constexpr double outer_radius = 0.100;
    constexpr double inner_radius = 0.05;
    constexpr double layer_pitch = 0.0065;
    constexpr double maximum_roll_d_hat = 0.005;

    const double spiral_rate = layer_pitch / kTwoPi;
    const double theta_max =
        (outer_radius - inner_radius) / spiral_rate;
    const auto spiral_primitive = [spiral_rate](const double radius) {
        const double root = std::sqrt(
            radius * radius + spiral_rate * spiral_rate);
        return 0.5 * (
            radius * root
            + spiral_rate * spiral_rate
                * std::asinh(radius / spiral_rate));
    };
    const double outer_primitive = spiral_primitive(outer_radius);
    const double spiral_length =
        (outer_primitive - spiral_primitive(inner_radius)) / spiral_rate;
    // Give the straight leader exactly eight material rows. Below, its
    // downslope projection is shortened just enough to accommodate the raised
    // pin without stretching any of those edges.
    const double leader_length =
        static_cast<double>(leader_rows) * spiral_length
        / static_cast<double>(cloth_ny - leader_rows);
    const double cloth_length = leader_length + spiral_length;

    // Keep every caller-provided activation distance strictly inside both the
    // nominal grid-edge bound and the tightly wound 6.5 mm layer pitch. A
    // 5 mm activation distance supports neighboring turns after only a small
    // relative motion instead of allowing the loose roll to collapse first.
    if (params.d_hat > 0.0) {
        const double nominal_min_edge = std::min(
            cloth_width / static_cast<double>(cloth_nx),
            cloth_length / static_cast<double>(cloth_ny));
        params.d_hat = std::min(
            std::min(params.d_hat, maximum_roll_d_hat),
            0.45 * nominal_min_edge);
    }
    // Keep the tightly wound roll immediately outside both the cloth IPC
    // barrier and the SDF penalty range, but raise the pinned edge to a
    // visibly separate 2 cm clearance. This also keeps the initial pose
    // force-free when a caller chooses eps_sdf larger than d_hat.
    constexpr double roll_clearance_margin = 1.0e-4;
    constexpr double pin_clearance_margin = 1.0e-3;
    constexpr double minimum_clearance = 0.002;
    constexpr double minimum_pin_clearance = 0.020;
    const double active_contact_range = std::max(
        std::max(params.d_hat, 0.0),
        std::max(params.eps_sdf, 0.0));
    const double roll_clearance = std::max(
        active_contact_range + roll_clearance_margin,
        minimum_clearance);
    const double pin_clearance = std::max(
        active_contact_range + pin_clearance_margin,
        minimum_pin_clearance);
    const double leader_clearance_drop =
        pin_clearance - roll_clearance;
    const double leader_downslope_span = std::sqrt(
        leader_length * leader_length
        - leader_clearance_drop * leader_clearance_drop);
    const double roll_downslope_offset = leader_downslope_span;

    // The alternating diagonals remove the one-sided hinge pattern of a
    // uniformly triangulated grid. The helper still records a flat membrane
    // metric and flat hinge rest angles, so repositioning afterward winds the
    // same flat-rest sheet without changing its material coordinates or
    // preferred curvature.
    const int cloth_base = build_square_mesh_alternating_diagonals(
        ref_mesh, state, X, cloth_nx, cloth_ny,
        cloth_width, cloth_length,
        Vec3(-0.5 * cloth_width, 0.0, 0.0));

    const double ramp_run = ramp_front_z - ramp_back_z;
    const double slope_length = std::hypot(ramp_height, ramp_run);
    const Vec3 downslope(
        0.0, -ramp_height / slope_length, ramp_run / slope_length);
    const Vec3 ramp_normal(
        0.0, ramp_run / slope_length, ramp_height / slope_length);
    const Vec3 ramp_top(0.0, ramp_height, ramp_back_z);

    // The SDF obstacle is the union of the half-spaces below these two
    // planes. Since obstacle evaluation selects the minimum signed distance,
    // the incline is active before the toe and the horizontal ground is
    // active after it. Their zero sets meet at (y=0, z=ramp_front_z), so the
    // cloth transitions from the finite visible ramp onto the ground without
    // putting any rigid/static collision geometry into RefMesh.
    params.sdf_planes.push_back(
        PlaneSDF{Vec3::Zero(), Vec3::UnitY()});
    params.sdf_planes.push_back(
        PlaneSDF{ramp_top, ramp_normal});

    const Vec3 roll_center =
        ramp_top + roll_downslope_offset * downslope
        + (outer_radius + roll_clearance) * ramp_normal;

    // The straight leader drops gently from the raised pin to the roll's outer
    // bottom point (phase zero). Its downslope projection is chosen with the
    // Pythagorean relation above, so all eight leader edges remain strain-free
    // and the roll moves only 0.583 mm upslope. The join is C0; its small
    // tangent mismatch combines the leader slope and spiral radial rate.

    const auto spiral_arc_length = [&](const double theta) {
        const double radius = outer_radius - spiral_rate * theta;
        return (outer_primitive - spiral_primitive(radius)) / spiral_rate;
    };
    const auto theta_at_arc_length = [&](const double target_length) {
        double lower = 0.0;
        double upper = theta_max;
        for (int iteration = 0; iteration < 60; ++iteration) {
            const double middle = 0.5 * (lower + upper);
            if (spiral_arc_length(middle) < target_length)
                lower = middle;
            else
                upper = middle;
        }
        return 0.5 * (lower + upper);
    };

    const auto cloth_node = [cloth_base](const int i, const int j) {
        return cloth_base + j * (cloth_nx + 1) + i;
    };
    for (int j = 0; j <= cloth_ny; ++j) {
        const double s = cloth_length
            * static_cast<double>(j) / static_cast<double>(cloth_ny);
        Vec3 centerline;
        if (s <= leader_length) {
            const double leader_fraction = s / leader_length;
            centerline = ramp_top
                + leader_fraction * leader_downslope_span * downslope
                + (pin_clearance
                    - leader_fraction * leader_clearance_drop) * ramp_normal;
        } else {
            const double theta = theta_at_arc_length(s - leader_length);
            const double radius = outer_radius - spiral_rate * theta;
            centerline = roll_center + radius * (
                std::sin(theta) * downslope
                - std::cos(theta) * ramp_normal);
        }

        for (int i = 0; i <= cloth_nx; ++i) {
            // Reverse the across-ramp traversal in world x so the existing
            // triangle winding faces along the ramp's outward normal.
            const double across = cloth_width
                * (0.5 - static_cast<double>(i)
                    / static_cast<double>(cloth_nx));
            state.deformed_positions[cloth_node(i, j)] =
                centerline + across * Vec3::UnitX();
        }
    }
    state.velocities.assign(
        state.deformed_positions.size(), Vec3::Zero());

    // The uphill short edge is the only pinned part of the sheet. With the
    // recommended 5 mm d_hat it starts 2 cm above the ramp, while the straight
    // leader descends without stretching to the roll's 5.1 mm clearance.
    for (int i = 0; i <= cloth_nx; ++i)
        append_pin(pins, cloth_node(i, 0), state.deformed_positions);

    // Closed triangular prism used only to visualize the two-plane SDF ramp.
    // It never enters RefMesh, the broad phase, CCD, or the nonlinear solve.
    const double half_ramp_width = 0.5 * ramp_width;
    const std::vector<Vec3> ramp_positions = {
        Vec3(-half_ramp_width, 0.0, ramp_back_z),
        Vec3( half_ramp_width, 0.0, ramp_back_z),
        Vec3(-half_ramp_width, ramp_height, ramp_back_z),
        Vec3( half_ramp_width, ramp_height, ramp_back_z),
        Vec3(-half_ramp_width, 0.0, ramp_front_z),
        Vec3( half_ramp_width, 0.0, ramp_front_z),
    };
    static constexpr int ramp_triangles[24] = {
        0, 1, 5, 0, 5, 4,  // bottom, outward -y
        0, 2, 3, 0, 3, 1,  // back, outward -z
        2, 4, 5, 2, 5, 3,  // slope, outward ramp_normal
        0, 4, 2,            // left cap, outward -x
        1, 3, 5,            // right cap, outward +x
    };
    static_x = ramp_positions;
    static_tris.assign(ramp_triangles, ramp_triangles + 24);

    // Large, upward-facing visualization for the y=0 ground SDF. Render it
    // one millimeter below the analytic surface to avoid coplanar z-fighting
    // with the bottom of the wedge visualization.
    constexpr double ground_y = -0.001;
    constexpr double ground_x_min = -3.0;
    constexpr double ground_x_max = 3.0;
    constexpr double ground_z_min = -2.0;
    constexpr double ground_z_max = 4.0;
    const int ground_base = static_cast<int>(static_x.size());
    static_x.insert(
        static_x.end(),
        {Vec3(ground_x_min, ground_y, ground_z_min),
         Vec3(ground_x_min, ground_y, ground_z_max),
         Vec3(ground_x_max, ground_y, ground_z_max),
         Vec3(ground_x_max, ground_y, ground_z_min)});
    static_tris.insert(
        static_tris.end(),
        {ground_base, ground_base + 1, ground_base + 2,
         ground_base, ground_base + 2, ground_base + 3});

    ref_mesh.build_deformable_nodes();
}

// ---------------------------------------------------------------------------
// Example 13: three stacked cloth layers between fixed and oscillating edges
// ---------------------------------------------------------------------------
// Command line:  OMP_NUM_THREADS=10 OMP_DYNAMIC=FALSE ./build/3D_sim --example 13 --num_frames 200 --substeps 30 --max_substep_iters 100 --fixed_iters --friction_coefficient 0.0 --friction_velocity_epsilon 0.01 --outdir oscillating_cloth_layers_output --format geo --kB 0.01 --osc_amplitude 0.01 --osc_frequency 6.0 --osc_length 1.0 --gy 0
void build_oscillating_cloth_layers_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    OscillatingClothLayersSpec& spec) {
    clear_model(ref_mesh, state, X, pins);

    if (args.osc_nx <= 0 || args.osc_nz <= 0) {
        throw std::invalid_argument(
            "example 13 requires osc_nx and osc_nz to be positive");
    }
    if (!std::isfinite(args.osc_length) || args.osc_length <= 0.0
        || !std::isfinite(args.osc_width) || args.osc_width <= 0.0) {
        throw std::invalid_argument(
            "example 13 requires positive finite cloth dimensions");
    }
    if (!std::isfinite(args.osc_layer_gap) || args.osc_layer_gap <= 0.0) {
        throw std::invalid_argument(
            "example 13 requires a positive finite osc_layer_gap");
    }
    if (!std::isfinite(args.osc_amplitude) || args.osc_amplitude < 0.0) {
        throw std::invalid_argument(
            "example 13 requires a nonnegative finite osc_amplitude");
    }
    if (!std::isfinite(args.osc_frequency) || args.osc_frequency < 0.0) {
        throw std::invalid_argument(
            "example 13 requires a nonnegative finite osc_frequency");
    }

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();

    // Table 1 gives a 2 mm contact radius for this experiment. Interpret it
    // as this IPC scene's maximum barrier activation distance, while still
    // respecting the repository-wide half-shortest-edge requirement.
    constexpr double published_contact_radius = 0.002;
    const double nominal_min_edge = std::min(
        args.osc_length / static_cast<double>(args.osc_nx),
        args.osc_width / static_cast<double>(args.osc_nz));
    if (params.d_hat > 0.0) {
        params.d_hat = std::min(
            params.d_hat,
            std::min(published_contact_radius,
                     0.45 * nominal_min_edge));
    }
    if (params.d_hat > 0.0 && args.osc_layer_gap <= params.d_hat) {
        throw std::invalid_argument(
            "example 13 requires osc_layer_gap > the effective d_hat");
    }

    spec = OscillatingClothLayersSpec{};
    spec.motion_direction = Vec3::UnitY();
    spec.amplitude = args.osc_amplitude;
    spec.frequency_hz = args.osc_frequency;

    constexpr int layer_count = 3;
    constexpr double center_y = 0.20;
    const std::size_t driven_count = static_cast<std::size_t>(layer_count)
        * static_cast<std::size_t>(args.osc_nz + 1);
    spec.driven_pin_indices.reserve(driven_count);
    spec.driven_initial_targets.reserve(driven_count);

    for (int layer = 0; layer < layer_count; ++layer) {
        const double layer_y = center_y
            + static_cast<double>(layer - 1) * args.osc_layer_gap;
        const Vec3 origin(
            -0.5 * args.osc_length,
            layer_y,
            -0.5 * args.osc_width);
        const int base = build_square_mesh_alternating_diagonals(
            ref_mesh, state, X,
            args.osc_nx, args.osc_nz,
            args.osc_length, args.osc_width, origin);

        // The left edge is prescribed and the opposite right edge is fixed.
        // The two widthwise edges remain free except at their shared clamped
        // corner vertices.
        for (int j = 0; j <= args.osc_nz; ++j) {
            const int driven_vertex =
                base + j * (args.osc_nx + 1);
            const int fixed_vertex = driven_vertex + args.osc_nx;

            spec.driven_pin_indices.push_back(
                static_cast<int>(pins.size()));
            append_pin(pins, driven_vertex, state.deformed_positions);
            spec.driven_initial_targets.push_back(
                pins.back().target_position);

            append_pin(pins, fixed_vertex, state.deformed_positions);
        }
    }

    state.velocities.assign(
        state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_deformable_nodes();
}

void update_oscillating_cloth_layer_pins(
    std::vector<Pin>& pins,
    const OscillatingClothLayersSpec& spec,
    const double t) {
    const double displacement = spec.amplitude * std::sin(
        kTwoPi * spec.frequency_hz * t);
    const Vec3 offset = displacement * spec.motion_direction;
    const std::size_t driven_count = spec.driven_pin_indices.size();
    for (std::size_t driven = 0; driven < driven_count; ++driven) {
        pins[static_cast<std::size_t>(
            spec.driven_pin_indices[driven])].target_position =
                spec.driven_initial_targets[driven] + offset;
    }
}

// ---------------------------------------------------------------------------
// Example 14: rigid IPC Figure 8 wrecking ball and 560-cube wall
// ---------------------------------------------------------------------------
// Command line: OMP_NUM_THREADS=10 OMP_DYNAMIC=FALSE ./build/3D_sim --example 14 --num_frames 200 --substeps 30 --max_substep_iters 200 --fixed_iters --d_hat 0.001 --k_barrier 1e9 --k_sdf 1e8 --eps_sdf 0.002 --friction_coefficient 0.1 --outdir wrecking_ball_tuned_output --format geo
void build_wrecking_ball_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris) {
    clear_model(ref_mesh, state, X, pins);
    static_x.clear();
    static_tris.clear();

    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = args.k_sdf;
    params.eps_sdf = args.eps_sdf;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;
    params.use_ogc = false;
    params.use_ogc_solver = false;

    // Use the material densities from rigid-ipc's final paper fixture.
    constexpr int chain_body_count = 14;
    constexpr int ordinary_link_count = chain_body_count - 1;
    constexpr double link_density = 7680.0;
    constexpr double cube_density = 1000.0;
    constexpr double link_height = 1.5;
    constexpr double padded_link_thickness = 0.3;
    constexpr double link_center_spacing =
        link_height - 2.0 * padded_link_thickness;
    constexpr double inclination = kPi / 6.0;
    // The source fixture uses ground y=-1. Translate the complete scene by
    // +1 m so its horizontal ground lies at y=0 and all bodies start above it.
    constexpr double scene_y_translation = 1.0;
    constexpr double fixture_y_offset = 12.0 + scene_y_translation;
    constexpr double link_target_max_extent = 1.5;
    constexpr double ball_target_max_extent = 4.75;
    const std::filesystem::path asset_directory =
        wrecking_ball_asset_directory(args.datadir);
    const std::string link_filename =
        (asset_directory / "link.obj").string();
    const std::string ball_filename =
        (asset_directory / "ball.obj").string();

    // The fixture's XYZ Euler transform is Rz(-60 degrees) Ry(alternating
    // 0/90 degrees) Rx(0). Quaternion multiplication follows the same active
    // rotation order. Alternating the ring planes makes adjacent bodies truly
    // interlinked; there are no artificial joints or springs in this scene.
    const Vec4 rotate_z_minus_sixty(
        std::cos(kPi / 6.0), 0.0, 0.0, -std::sin(kPi / 6.0));
    const Vec4 rotate_y_ninety(
        std::cos(kPi / 4.0), 0.0, std::sin(kPi / 4.0), 0.0);
    const Vec4 crossed_link_orientation = quaternion_normalize(
        quaternion_multiply(rotate_z_minus_sixty, rotate_y_ninety));

    Vec3 fixed_link_fixture_position = Vec3::Zero();
    for (int link = 0; link < chain_body_count; ++link) {
        const double chain_coordinate =
            static_cast<double>(chain_body_count - link)
            * link_center_spacing;
        const Vec3 fixture_position(
            chain_coordinate * std::cos(inclination),
            fixture_y_offset
                + chain_coordinate * std::sin(inclination),
            0.0);
        if (link == 0)
            fixed_link_fixture_position = fixture_position;

        const Vec4 orientation = (link % 2 == 0)
            ? rotate_z_minus_sixty
            : crossed_link_orientation;
        const bool is_ball = link == ordinary_link_count;
        // The normalized importer places the source AABB center at `center`.
        // link.obj is centered at its model origin. ball.obj is intentionally
        // not: its sphere hangs below an integrated terminal ring, and its
        // source AABB center is (0,-1.625,0). This correction reconstructs the
        // uniformly translated reference transform
        // x_world = fixture_position + R*x_source. The link's 0.5 micrometre
        // z offset is retained as authored rather than silently removed by
        // AABB recentering.
        const Vec3 source_aabb_center = is_ball
            ? Vec3(0.0, -1.625, 0.0)
            : Vec3(0.0, 0.0, 0.0000005);
        const Vec3 importer_center = fixture_position
            + quaternion_rotate(orientation, source_aabb_center);

        const std::size_t triangle_begin = ref_mesh.tris.size() / 3;
        const int rigid_body = append_normalized_obj_rigid_body(
            is_ball ? ball_filename : link_filename,
            state, ref_mesh, importer_center,
            is_ball ? ball_target_max_extent : link_target_max_extent,
            link_density, Vec3::Zero(), orientation, Vec3::Zero(),
            link == 0
                ? RigidBodyUpdateMode::None
                : RigidBodyUpdateMode::TranslationAndOrientation);
        use_closed_volume_rigid_mass_properties(
            rigid_body, triangle_begin, ref_mesh.tris.size() / 3,
            ref_mesh, state);
    }

    // The original fixed plane is centered below the top link and spans
    // 20 by 20 metres. After the uniform scene translation, use y=0 as an
    // infinite upward-facing SDF plane; the finite quad preserves its footprint.
    constexpr double ground_y = -1.0 + scene_y_translation;
    params.sdf_planes.push_back(PlaneSDF{
        Vec3(fixed_link_fixture_position.x(), ground_y, 0.0),
        Vec3::UnitY()});

    // Tightly packed version of the paper wall: 8 x 7 x 10 unit cubes. A
    // nonzero 2 mm separation avoids singular touching contact while removing
    // the reference fixture's visibly falling 5 cm gaps. Use the same 2 mm
    // clearance under the first row. itertools.product in the reference makes
    // z the fastest-moving coordinate, which this loop order preserves.
    constexpr int wall_width = 8;
    constexpr int wall_height = 7;
    constexpr int wall_depth = 10;
    constexpr double cube_edge = 1.0;
    constexpr double cube_gap = 0.002;
    constexpr double cube_spacing = cube_edge + cube_gap;
    constexpr double cube_ground_clearance = 0.002;
    const Vec3 first_cube_center = Vec3(
        fixed_link_fixture_position.x()
            + 0.5 - 0.5 * static_cast<double>(wall_width),
        ground_y + 0.5 * cube_edge + cube_ground_clearance,
        0.5 - 0.5 * static_cast<double>(wall_depth));
    for (int width = 0; width < wall_width; ++width) {
        for (int height = 0; height < wall_height; ++height) {
            for (int depth = 0; depth < wall_depth; ++depth) {
                const Vec3 center = first_cube_center + cube_spacing * Vec3(
                    static_cast<double>(width),
                    static_cast<double>(height),
                    static_cast<double>(depth));
                const std::size_t triangle_begin =
                    ref_mesh.tris.size() / 3;
                const int rigid_body = append_rigid_cube(
                    center, cube_edge, cube_density, Vec3::Zero(),
                    ref_mesh, state);
                use_closed_volume_rigid_mass_properties(
                    rigid_body, triangle_begin,
                    ref_mesh.tris.size() / 3, ref_mesh, state);
            }
        }
    }

    constexpr double ground_half_extent = 10.0;
    const double ground_center_x = fixed_link_fixture_position.x();
    static_x = {
        Vec3(ground_center_x - ground_half_extent, ground_y,
             -ground_half_extent),
        Vec3(ground_center_x - ground_half_extent, ground_y,
              ground_half_extent),
        Vec3(ground_center_x + ground_half_extent, ground_y,
              ground_half_extent),
        Vec3(ground_center_x + ground_half_extent, ground_y,
             -ground_half_extent),
    };
    static_tris = {0, 1, 2, 0, 2, 3};

    // Imported ball triangles contain the finest edges. Retain a caller's
    // smaller activation distance while enforcing both the simulator-wide
    // discretization bound and enough separation from the dense cube grid to
    // avoid activating nearly every lateral neighbor from solver-scale motion.
    double minimum_surface_edge = std::numeric_limits<double>::infinity();
    for (std::size_t triangle = 0;
         triangle < ref_mesh.tris.size() / 3; ++triangle) {
        const int* nodes = ref_mesh.tris.data() + 3 * triangle;
        for (int local = 0; local < 3; ++local) {
            const Vec3& first = state.deformed_positions[
                static_cast<std::size_t>(nodes[local])];
            const Vec3& second = state.deformed_positions[
                static_cast<std::size_t>(nodes[(local + 1) % 3])];
            minimum_surface_edge = std::min(
                minimum_surface_edge, (second - first).norm());
        }
    }
    if (params.d_hat > 0.0 && std::isfinite(minimum_surface_edge)) {
        params.d_hat = std::min({
            params.d_hat, 0.45 * minimum_surface_edge,
            0.5 * cube_gap});
    }

    ref_mesh.build_deformable_nodes();
}

// ---------------------------------------------------------------------------
// Example 15: separated cloth sheets falling onto a long horizontal cylinder
// ---------------------------------------------------------------------------
void build_cloth_cylinder_drop_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris) {
    if (args.drop_stack_count < 1 || args.drop_cloth_nx < 1
        || args.drop_cloth_ny < 1 || args.cyl_nu < 3 || args.cyl_cap_rings < 1) {
        throw std::invalid_argument(
            "example 15 requires positive cloth count and grid subdivisions, "
            "cyl_cap_rings >= 1, and cyl_nu >= 3");
    }
    const std::size_t sheets = static_cast<std::size_t>(args.drop_stack_count);
    const std::size_t nx = static_cast<std::size_t>(args.drop_cloth_nx);
    const std::size_t ny = static_cast<std::size_t>(args.drop_cloth_ny);
    const std::size_t index_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (nx + 1 > index_limit / (ny + 1) / sheets
        || nx > index_limit / 6 / ny / sheets) {
        throw std::invalid_argument("example 15 cloth stack exceeds mesh index limits");
    }
    const std::size_t cloth_vertices = sheets * (nx + 1) * (ny + 1);
    const std::size_t cloth_indices = sheets * 6 * nx * ny;
    for (const double length : {args.drop_cloth_w, args.drop_cloth_h,
                               args.drop_spacing, args.cyl_ground_size, args.cyl_ground_cell_size,
                               args.cyl_radius, args.cyl_length}) {
        if (!std::isfinite(length) || length <= 0.0) {
            throw std::invalid_argument(
                "example 15 requires positive finite cloth dimensions, "
                "drop_spacing, ground size and cell size, cylinder radius and length");
        }
    }
    if (!std::isfinite(args.k_sdf) || args.k_sdf < 0.0
        || !std::isfinite(args.cyl_sdf_padding) || args.cyl_sdf_padding < 0.0) {
        throw std::invalid_argument(
            "example 15 requires nonnegative finite k_sdf and cyl_sdf_padding");
    }
    if (!std::isfinite(params.d_hat) || params.d_hat >= args.drop_spacing) {
        throw std::invalid_argument(
            "example 15 requires a finite d_hat < drop_spacing");
    }
    const double collision_radius = args.cyl_radius + args.cyl_sdf_padding;
    const Vec3 cylinder_center(args.cyl_cx, args.cyl_cy, args.cyl_cz);
    if (!cylinder_center.allFinite() || !std::isfinite(args.drop_first_y)
        || !std::isfinite(args.drop_cx) || !std::isfinite(args.drop_cz)) {
        throw std::invalid_argument(
            "example 15 requires finite cylinder and cloth-stack coordinates");
    }
    if (args.drop_first_y <= std::max(0.0, args.cyl_cy + collision_radius)
                                + std::max(0.0, params.eps_sdf)) {
        throw std::invalid_argument(
            "example 15 requires drop_first_y above the ground and cylinder "
            "padded top, with eps_sdf clearance");
    }

    // Export-only tessellation: contact still uses the analytic SDFs below.
    // The cylinder SDF is infinite along z, as in the other cylinder examples.
    const double ground_cells = std::ceil(args.cyl_ground_size / args.cyl_ground_cell_size);
    if (!std::isfinite(ground_cells)
        || ground_cells >= std::sqrt(static_cast<double>(std::numeric_limits<int>::max())) - 1.0) {
        throw std::invalid_argument("example 15 ground tessellation exceeds mesh index limits");
    }
    const int ground_n = std::max(1, static_cast<int>(ground_cells));
    clear_model(ref_mesh, state, X, pins);
    params.k_sdf = args.k_sdf;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    RefMesh static_ref;
    DeformedState static_state;
    std::vector<Vec2> static_X;
    build_square_mesh(
        static_ref, static_state, static_X, ground_n, ground_n,
        args.cyl_ground_size, args.cyl_ground_size,
        Vec3(args.cyl_cx - 0.5 * args.cyl_ground_size, 0.0,
             args.cyl_cz - 0.5 * args.cyl_ground_size));
    // The cloth grid's winding faces -y; the visible floor should face +y.
    for (std::size_t triangle = 0; triangle < static_ref.tris.size(); triangle += 3)
        std::swap(static_ref.tris[triangle + 1], static_ref.tris[triangle + 2]);
    build_cylinder_mesh(
        static_ref, static_state, static_X,
        args.cyl_nu, args.cyl_radius, args.cyl_length, cylinder_center, args.cyl_cap_rings);
    static_x = std::move(static_state.deformed_positions);
    static_tris = std::move(static_ref.tris);

    // Match the visible floor at y=0. Lowering this plane also lowers the
    // penalty's force-free height and lets cloth sink through the visual mesh.
    params.sdf_planes.push_back(PlaneSDF{Vec3::Zero(), Vec3::UnitY()});
    // SDF forces act at cloth vertices. Coarse edges/faces can chord through
    // the curved cylinder even with every vertex outside it, so enlarge only
    // the collision radius. This margin is tunable without changing resolution.
    params.sdf_cylinders.push_back(CylinderSDF{
        cylinder_center, Vec3::UnitZ(), collision_radius});

    // Place the cloth stack independently so an off-center cylinder gives
    // unequal overhangs and gravity can pull the sheets toward the ground.
    X.reserve(cloth_vertices);
    state.deformed_positions.reserve(cloth_vertices);
    ref_mesh.tris.reserve(cloth_indices);
    for (int sheet = 0; sheet < args.drop_stack_count; ++sheet) {
        const Vec3 origin(
            args.drop_cx - 0.5 * args.drop_cloth_w,
            args.drop_first_y + sheet * args.drop_spacing,
            args.drop_cz - 0.5 * args.drop_cloth_h);
        append_square_mesh(
            ref_mesh, state, X, args.drop_cloth_nx, args.drop_cloth_ny,
            args.drop_cloth_w, args.drop_cloth_h, origin);
    }
    ref_mesh.initialize(X, state.deformed_positions);
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_deformable_nodes();
}
