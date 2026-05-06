#include "example.h"
#include "make_shape.h"

#include <cmath>

namespace {

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;

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
    const double theta_top = effective_theta(spec.omega_top, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    const double theta_bot = effective_theta(spec.omega_bot, t, spec.t_settle, spec.t_ramp,
                                              spec.max_abs_theta, spec.untwist, spec.t_hold);
    // Yaw the SDF axis about +y (axis_point at origin → pure direction rotate)
    // by the same theta that drives the pins, so pin and SDF surface co-rotate.
    params.sdf_cylinders[0].point = spec.top_axis_point;
    params.sdf_cylinders[0].axis  = rotate_about_y_axis(Vec3::UnitX(), Vec3::Zero(), theta_top);
    params.sdf_cylinders[1].point = spec.bot_axis_point;
    params.sdf_cylinders[1].axis  = rotate_about_y_axis(Vec3::UnitX(), Vec3::Zero(), theta_bot);
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
// Example 3: cloth pile
// ---------------------------------------------------------------------------
// Square ground sheet pinned at its four corners catches a stack of smaller
// cloths dropped from above. Contact is pure IPC log-barrier (no SDF), and
// nothing is scripted -- the only inputs after construction are gravity and
// the initial drop velocity.
void build_cloth_pile_example(const IPCArgs3D& args,
                              RefMesh& ref_mesh,
                              DeformedState& state,
                              std::vector<Vec2>& X,
                              std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);

    // Ground sheet in the xz plane at y=0; corners pinned to form a hammock.
    const int    g_nx = args.pile_ground_nx;
    const int    g_ny = args.pile_ground_ny;
    const double g_w  = args.pile_ground_size;
    const double g_h  = args.pile_ground_size;

    const int g_base = build_square_mesh(
            ref_mesh, state, X,
            g_nx, g_ny, g_w, g_h,
            Vec3(-0.5 * g_w, 0.0, -0.5 * g_h));

    // build_square_mesh stores grid (i, j) at base + j * (g_nx + 1) + i, so
    // the four corners are (0,0), (g_nx,0), (0,g_ny), (g_nx,g_ny).
    const int row_stride = g_nx + 1;
    append_pin(pins, g_base + 0,                            state.deformed_positions);
    append_pin(pins, g_base + g_nx,                         state.deformed_positions);
    append_pin(pins, g_base + g_ny * row_stride,            state.deformed_positions);
    append_pin(pins, g_base + g_ny * row_stride + g_nx,     state.deformed_positions);

    // Falling cloths, evenly spaced in y above the hammock. A small per-cloth
    // xz offset breaks perfect symmetry so the pile doesn't lock into a
    // straight column. All falling vertices share an initial -y velocity.
    const int    n      = args.pile_count;
    const int    nx     = args.pile_nx;
    const int    ny     = args.pile_ny;
    const double w      = args.pile_cloth_size;
    const double h      = args.pile_cloth_size;
    const double first  = args.pile_first_y;
    const double dy     = args.pile_spacing;
    const double v_drop = args.pile_drop_speed;

    const int falling_begin = static_cast<int>(state.deformed_positions.size());

    for (int s = 0; s < n; ++s) {
        const double x_off = 0.005 * (s - 1);
        const double z_off = 0.004 * (s - 1);
        const Vec3 origin(-0.5 * w + x_off,
                          first + s * dy,
                          -0.5 * h + z_off);
        build_square_mesh(ref_mesh, state, X, nx, ny, w, h, origin);
    }

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    for (int i = falling_begin; i < static_cast<int>(state.velocities.size()); ++i) {
        state.velocities[i] = Vec3(0.0, -v_drop, 0.0);
    }
}
