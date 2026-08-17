#include "example.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "rigid_body_ipc.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;

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
    params.sdf_cylinders[spec.cyl_sdf_index].point = spec.cyl_axis_point;
    params.sdf_cylinders[spec.cyl_sdf_index].axis  =
        rotate_about_y_axis(Vec3::UnitX(), Vec3::Zero(), theta);
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
// Commend line: ./build/3D_sim --example 5 --num_frames 500 --substeps 30 --tol_abs 1e-12 --tol_rel 1e-10 --outdir racket_output 
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
// Example 7: rigid box and hexagonal prism falling onto a ground plane
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 7 --num_frames 200 --substeps 10 --tol_abs 1e-12 --tol_rel 1e-10 --outdir drop_box_output --format obj
void build_rigid_box_drop_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris) {
    clear_model(ref_mesh, state, X, pins);
    static_x.clear();
    static_tris.clear();

    // Keep these values controllable from the command line. Their defaults
    // provide Earth gravity and a stiff, slightly softened ground contact.
    params.gravity = Vec3(args.gx, args.gy, args.gz);
    params.k_sdf = args.k_sdf;
    params.eps_sdf = args.eps_sdf;
    params.d_hat = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.sdf_planes.push_back(
        {Vec3::Zero(), Vec3::UnitY()});
    params.use_ccd = false;
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;

    std::vector<Vec3> x;
    std::vector<int> tris;
    const Vec3 box_center(-0.55, 5.0, 0.0);
    const Vec3 box_half_extent(0.18, 0.14, 0.16);
    append_box_mesh(
        box_center - box_half_extent,
        box_center + box_half_extent, x, tris);

    ref_mesh.tris = tris;
    X.reserve(x.size());
    for (const Vec3& position : x)
        X.push_back(position.head<2>());

    create_rigid_body(
        x, Vec3::Zero(), Vec4(1, 0.0, 0.0, 0.0),
        Vec3{1.0, 0.0, 0.0},
        8.0 * box_half_extent.x() * box_half_extent.y()
            * box_half_extent.z() * params.rigid_density,
        ref_mesh, state);

    append_rigid_polygon(
        6, state, ref_mesh,
        Vec3(0.55, 5.0, 0.0),
        /*radius=*/0.22,
        params.rigid_density,
        /*thickness=*/0.28,
        Vec3::Zero(), Vec4(1.0, 0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0));

    // Flat visual ground at the same y=0 surface used by the plane SDF.
    static_x = {
        Vec3(-2.0, 0.0, -2.0),
        Vec3( 2.0, 0.0, -2.0),
        Vec3( 2.0, 0.0,  2.0),
        Vec3(-2.0, 0.0,  2.0),
    };
    static_tris = {
        0, 2, 1,
        0, 3, 2,
    };
}


// ---------------------------------------------------------------------------
// Example 8: two rigid polygonal prisms moving toward one another
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 8 --num_frames 60 --max_substep_iters 500 --substeps 10 --tol_rel 1e-10 --outdir polygon_collision_output --format obj
void build_two_rigid_polygon_collision_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params) {
    clear_model(ref_mesh, state, X, pins);

    params.gravity = Vec3::Zero();
    params.d_hat = args.d_hat;
    params.k_barrier = args.k_barrier;
    params.k_sdf = 0.0;
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;

    constexpr double radius = 0.30;
    const double density = params.rigid_density;
    constexpr double thickness = 0.20;
    // Rotating a regular hexagon by 30 degrees places flat side faces at its
    // +/-x extrema. Giving both bodies this orientation produces side-to-side
    // contact as they approach along x.
    const double collision_half_angle = kPi / 12.0;
    const Vec4 collision_orientation(
        std::cos(collision_half_angle), 0.0, 0.0,
        std::sin(collision_half_angle));

    append_rigid_polygon(
        6, state, ref_mesh,
        Vec3(-0.65, 0.0, 0.0),
        radius, density, thickness,
        Vec3(1.0, 0.0005, 0.0),
        collision_orientation, Vec3(1.2, -1.2, 2.0));

    append_rigid_polygon(
        6, state, ref_mesh,
        Vec3(0.65, 0.0, 0.0),
        radius, density, thickness,
        Vec3(-1.0, -0.0005, 0.0),
        collision_orientation, Vec3(-1.2, 1.2, -2.0));
}

// ---------------------------------------------------------------------------
// Example 9: twenty rigid polygonal prisms initialized in a static vertical stack
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 9 --num_frames 100 --substeps 10 --d_hat 0.001 --eps_sdf 0.0002 --outdir twenty_polygon_static_stack_output --format obj
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

    constexpr int polygon_count = 20;
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
// Example 10: five equally oriented rigid polygons dropping onto one another
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 10 --num_frames 100 --substeps 10 --outdir five_polygon_aligned_stack_output --format obj
void build_five_rigid_polygon_drop_scatter_example(
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

    constexpr int polygon_count = 5;
    constexpr double radius = 0.22;
    const double density = params.rigid_density;
    constexpr double thickness = 0.14;
    constexpr double lowest_center_y = 0.45;
    constexpr double center_spacing = 0.56;

    // Previous per-polygon orientations, kept for easy restoration:
    // const Vec3 xyz_angles[polygon_count] = {
    //     Vec3( 0.35,  0.15, -0.20),
    //     Vec3(-0.45,  0.30,  0.25),
    //     Vec3( 0.25, -0.40,  0.50),
    //     Vec3(-0.30, -0.20, -0.45),
    //     Vec3( 0.55,  0.35,  0.10),
    // };
    // const auto axis_angle_quaternion =
    //     [](const Vec3& axis, double angle) {
    //         const double half_angle = 0.5 * angle;
    //         const double sin_half_angle = std::sin(half_angle);
    //         return Vec4(
    //             std::cos(half_angle),
    //             sin_half_angle * axis.x(),
    //             sin_half_angle * axis.y(),
    //             sin_half_angle * axis.z());
    //     };

    const Vec3 initial_omega[polygon_count] = {
        Vec3( 0.8,  0.3, -0.5),
        Vec3(-0.6,  0.9,  0.4),
        Vec3( 0.5, -0.7,  0.8),
        Vec3(-0.9, -0.4,  0.3),
        Vec3( 0.4,  0.6, -0.8),
    };

    // Material z is the prism extrusion direction. Keeping it horizontal
    // makes the polygonal cap planes exactly vertical in world space.
    const Vec4 common_orientation(1.0, 0.0, 0.0, 0.0);

    for (int polygon = 0; polygon < polygon_count; ++polygon) {
        // Previous per-polygon orientation construction:
        // const Vec4 qx = axis_angle_quaternion(
        //     Vec3::UnitX(), xyz_angles[polygon].x());
        // const Vec4 qy = axis_angle_quaternion(
        //     Vec3::UnitY(), xyz_angles[polygon].y());
        // const Vec4 qz = axis_angle_quaternion(
        //     Vec3::UnitZ(), xyz_angles[polygon].z());
        // const Vec4 orientation = quaternion_normalize(
        //     quaternion_multiply(
        //         qz, quaternion_multiply(qy, qx)));

        // All centers share the same x-z position, so the bodies fall onto
        // one another instead of being given an artificial lateral scatter.
        const Vec3 center(
            0.0,
            lowest_center_y + center_spacing * polygon,
            0.0);
        append_rigid_polygon(
            6, state, ref_mesh, center,
            radius, density, thickness,
            Vec3::Zero(),
            common_orientation, initial_omega[polygon]);
    }

    // Flat visual ground at the y=0 plane SDF.
    static_x = {
        Vec3(-2.0, 0.0, -2.0),
        Vec3( 2.0, 0.0, -2.0),
        Vec3( 2.0, 0.0,  2.0),
        Vec3(-2.0, 0.0,  2.0),
    };
    static_tris = {
        0, 2, 1,
        0, 3, 2,
    };
}

// ---------------------------------------------------------------------------
// Example 11: one hundred rigid polygonal prisms of varied shapes falling
// into an open-top box
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 11 --num_frames 200 --substeps 10 --max_substep_iters 20 --fixed_iters --outdir hundred_polygon_box_fixed_iter_output --format obj
// ./build/3D_sim --example 11 --num_frames 200 --substeps 80 --max_substep_iters 5000 --outdir hundred_polygon_box_output --format obj
void build_hundred_rigid_polygon_box_drop_example(
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

    // The positive side of every plane is the box interior. With no ceiling
    // plane, these five half-spaces form an open-top container.
    constexpr double box_half_width = 1.60;
    constexpr double box_half_depth = 1.00;
    constexpr double wall_height = 1.25;
    params.sdf_planes.push_back(
        {Vec3(0.0, 0.0, 0.0), Vec3::UnitY()});
    params.sdf_planes.push_back(
        {Vec3(-box_half_width, 0.0, 0.0), Vec3::UnitX()});
    params.sdf_planes.push_back(
        {Vec3( box_half_width, 0.0, 0.0), -Vec3::UnitX()});
    params.sdf_planes.push_back(
        {Vec3(0.0, 0.0, -box_half_depth), Vec3::UnitZ()});
    params.sdf_planes.push_back(
        {Vec3(0.0, 0.0,  box_half_depth), -Vec3::UnitZ()});

    // Arrange the bodies in five collision-free 5 x 4 layers. The enclosing
    // sphere of every prism has radius sqrt(radius^2 + (t/2)^2), which is
    // smaller than half of every center-to-center spacing below.
    constexpr int column_count = 5;
    constexpr int row_count = 4;
    constexpr int layer_count = 5;
    constexpr int polygon_count = column_count * row_count * layer_count;
    constexpr double radius = 0.20;
    const double density = params.rigid_density;
    constexpr double thickness = 0.10;
    constexpr double column_spacing = 0.55;
    constexpr double row_spacing = 0.48;
    constexpr double vertical_spacing = 0.52;
    constexpr double lowest_center_y = 0.65;

    const auto axis_angle_quaternion = [](const Vec3& axis, double angle) {
        const double half_angle = 0.5 * angle;
        const double sin_half_angle = std::sin(half_angle);
        return Vec4(
            std::cos(half_angle),
            sin_half_angle * axis.x(),
            sin_half_angle * axis.y(),
            sin_half_angle * axis.z());
    };

    static constexpr int polygon_side_counts[] = {3, 4, 5, 7, 8};
    static constexpr int shape_count =
        sizeof(polygon_side_counts) / sizeof(polygon_side_counts[0]);
    for (int index = 0; index < polygon_count; ++index) {
        const int layer = index / (column_count * row_count);
        const int index_in_layer = index % (column_count * row_count);
        const int row = index_in_layer / column_count;
        const int column = index_in_layer % column_count;

        // Cycle through triangles, squares, pentagons, heptagons, and
        // octagons. Mixing all three grid coordinates prevents a polygon type
        // from lining up in a single column or layer. Every prism has the same
        // circumscribed radius, so the collision-free spacing remains valid.
        const int shape_index =
            (column + 2 * row + 3 * layer) % shape_count;
        const int polygon_sides =
            polygon_side_counts[shape_index];
        const double x_angle = ((index % 3) - 1) * 0.18;
        const double y_angle = (((index / 3) % 3) - 1) * 0.22;
        const double z_angle = (index % 7) * (kPi / 7.0);
        const Vec4 qx = axis_angle_quaternion(Vec3::UnitX(), x_angle);
        const Vec4 qy = axis_angle_quaternion(Vec3::UnitY(), y_angle);
        const Vec4 qz = axis_angle_quaternion(Vec3::UnitZ(), z_angle);
        const Vec4 orientation = quaternion_normalize(
            quaternion_multiply(
                qz, quaternion_multiply(qy, qx)));

        const Vec3 center(
            (column - 0.5 * (column_count - 1)) * column_spacing,
            lowest_center_y + layer * vertical_spacing,
            (row - 0.5 * (row_count - 1)) * row_spacing);

        append_rigid_polygon(
            polygon_sides, state, ref_mesh, center,
            radius, density, thickness,
            Vec3::Zero(), orientation, Vec3::Zero());
    }

    // Flat visual quads coincide with the ground and four side SDF planes.
    const auto append_visual_plane = [&static_x, &static_tris](
        const Vec3& x0, const Vec3& x1,
        const Vec3& x2, const Vec3& x3) {
        const int base = static_cast<int>(static_x.size());
        static_x.push_back(x0);
        static_x.push_back(x1);
        static_x.push_back(x2);
        static_x.push_back(x3);
        static_tris.insert(
            static_tris.end(),
            {base, base + 1, base + 2,
             base, base + 2, base + 3});
    };

    // Ground, upward normal +y.
    append_visual_plane(
        Vec3(-box_half_width, 0.0, -box_half_depth),
        Vec3(-box_half_width, 0.0,  box_half_depth),
        Vec3( box_half_width, 0.0,  box_half_depth),
        Vec3( box_half_width, 0.0, -box_half_depth));
    // Left and right walls, inward normals +x and -x.
    append_visual_plane(
        Vec3(-box_half_width, 0.0,         -box_half_depth),
        Vec3(-box_half_width, wall_height, -box_half_depth),
        Vec3(-box_half_width, wall_height,  box_half_depth),
        Vec3(-box_half_width, 0.0,          box_half_depth));
    append_visual_plane(
        Vec3(box_half_width, 0.0,          box_half_depth),
        Vec3(box_half_width, wall_height,  box_half_depth),
        Vec3(box_half_width, wall_height, -box_half_depth),
        Vec3(box_half_width, 0.0,         -box_half_depth));
    // Back and front walls, inward normals +z and -z.
    append_visual_plane(
        Vec3( box_half_width, 0.0,         -box_half_depth),
        Vec3( box_half_width, wall_height, -box_half_depth),
        Vec3(-box_half_width, wall_height, -box_half_depth),
        Vec3(-box_half_width, 0.0,         -box_half_depth));
    append_visual_plane(
        Vec3(-box_half_width, 0.0,          box_half_depth),
        Vec3(-box_half_width, wall_height,  box_half_depth),
        Vec3( box_half_width, wall_height,  box_half_depth),
        Vec3( box_half_width, 0.0,          box_half_depth));
}

// ---------------------------------------------------------------------------
// Example 12: fifty small rigid polygonal prisms falling onto a
// four-corner-pinned rectangular cloth
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 12 --num_frames 200 --substeps 10 --max_substep_iters 20 --fixed_iters --outdir fifty_polygons_on_pinned_cloth_output --format obj
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
// Example 13: one deformable volumetric solid falling onto an SDF ground
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 13 --num_frames 200 --substeps 20 --max_substep_iters 50 --tol_abs 1e-8 --tol_rel 1e-5 --outdir single_solid_ground_drop_output --format obj
void build_single_deformable_solid_ground_drop_example(
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
        PlaneSDF{Vec3::Zero(), Vec3::UnitY()});
    params.use_ccd_guess = false;
    params.use_verlet_guess = false;
    params.use_translation_guess = false;
    params.use_ogc = false;
    params.use_ogc_solver = false;

    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);

    constexpr int side_count = 8;
    constexpr double radius = 0.22;
    constexpr double thickness = 0.16;
    append_deformable_polygon_prism(
        side_count, state, ref_mesh, Vec3(0.0, 1.0, 0.0),
        radius, params.solid_density, thickness, flat_orientation);
    ref_mesh.build_deformable_nodes();

    // The visual plane coincides exactly with the infinite SDF ground.
    constexpr double ground_half_extent = 2.0;
    static_x = {
        Vec3(-ground_half_extent, 0.0, -ground_half_extent),
        Vec3(-ground_half_extent, 0.0,  ground_half_extent),
        Vec3( ground_half_extent, 0.0,  ground_half_extent),
        Vec3( ground_half_extent, 0.0, -ground_half_extent)};
    static_tris = {0, 1, 2, 0, 2, 3};
}

// ---------------------------------------------------------------------------
// Example 14: one deformable volumetric solid falling onto a pinned cloth (need fix)
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 14 --num_frames 200 --substeps 20 --max_substep_iters 30 --fixed_iters  --E 1e8 --outdir stiff_cloth_solid_drop_output --format obj --d_hat 0.019 --k_barrier 500
// weird if there is no --d_hat and --k_barrier. solid doesn't bounce up and looks like it sticks to the cloth. ccd issue?
void build_single_deformable_solid_drop_on_pinned_cloth_example(
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

    // Build and initialize the cloth before appending the solid boundary
    // triangles. This keeps the shell-elastic triangles as the leading
    // Dm_inverse/area prefix and the solid surface collision-only.
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

    // The material extrusion axis is +z. Rotate it exactly onto world +y so
    // both polygonal caps are horizontal, with no yaw or tilt.
    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);

    const int solid_base = append_deformable_polygon_prism(
        /*number_of_nodes=*/8, state, ref_mesh,
        /*center=*/Vec3(0.0, 1.5, 0.0),
        /*radius=*/0.30, params.solid_density,
        /*thickness=*/0.20, flat_orientation);
    // The unladen cloth begins falling under gravity too; this modest relative
    // downward speed makes the solid catch and load the sagging sheet early.
    for (std::size_t node = static_cast<std::size_t>(solid_base);
         node < state.velocities.size(); ++node) {
        state.velocities[node] = Vec3(0.0, -0.75, 0.0);
    }

    // Includes both cloth and tetrahedral nodes. The general solver derives
    // its disjoint cloth/solid block ranges from ref_mesh.tet_nodes.
    ref_mesh.build_deformable_nodes();
}

// ---------------------------------------------------------------------------
// Example 15: ten small rigid and ten larger deformable polygonal prisms
// falling onto a cloth pinned along two opposite sides
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 15 --num_frames 200 --substeps 20 --max_substep_iters 20 --fixed_iters --outdir twenty_rigid_deformable_polygons_on_pinned_cloth_output --format obj --E 1e8 --d_hat 0.019 --k_barrier 500 
void build_twenty_rigid_deformable_polygons_drop_on_pinned_cloth_example(
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

    constexpr int cloth_nx = 100;
    constexpr int cloth_nz = 100;
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

    // A small integer hash gives stable, random-looking samples on every
    // platform. Reproducibility keeps the scene and its construction test
    // deterministic while still varying size and position.
    const auto random_unit = [](std::uint32_t value) {
        value += 0x9e3779b9U;
        value = (value ^ (value >> 16U)) * 0x85ebca6bU;
        value = (value ^ (value >> 13U)) * 0xc2b2ae35U;
        value ^= value >> 16U;
        return static_cast<double>(value & 0x00ffffffU)
            / static_cast<double>(0x01000000U);
    };
    const auto sample = [&random_unit](const int polygon,
                                      const std::uint32_t channel) {
        return random_unit(
            static_cast<std::uint32_t>(polygon)
            + channel * 0x6d2b79f5U);
    };
    constexpr int polygon_count = 20;
    constexpr int columns = 5;
    constexpr double spacing = 0.72;
    const double rigid_body_density = params.rigid_density;
    const double deformable_body_density = params.solid_density;
    // The material extrusion axis is +z. Use the same exact flat orientation
    // for every rigid body and deformable solid: horizontal caps, no yaw or
    // tilt.
    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);
    for (int polygon = 0; polygon < polygon_count; ++polygon) {
        const int row = polygon / columns;
        const int column = polygon % columns;
        const int side_count = 3 + polygon / 2;
        const bool is_rigid = polygon % 2 == 0;
        // Solids are roughly twice as wide and twice as thick as the rigid
        // bodies, making the two independently solved object types visually
        // distinguishable in the exported surface mesh.
        const double radius = is_rigid
            ? 0.080 + 0.025 * sample(polygon, 0U)
            : 0.180 + 0.040 * sample(polygon, 0U);
        const double thickness = is_rigid
            ? 0.045 + 0.020 * sample(polygon, 1U)
            : 0.120 + 0.040 * sample(polygon, 1U);

        const Vec3 center(
            (column - 2.0) * spacing
                + 0.024 * (sample(polygon, 5U) - 0.5),
            2.05 + 0.12 * sample(polygon, 6U),
            (row - 1.5) * spacing
                + 0.024 * (sample(polygon, 7U) - 0.5));

        // Alternating rigid/solid pairs give ten bodies of each type and one
        // of each type for every side count from 3 through 12.
        if (is_rigid) {
            append_rigid_polygon(
                side_count, state, ref_mesh, center,
                radius, rigid_body_density, thickness,
                Vec3::Zero(), flat_orientation, Vec3::Zero());
        } else {
            append_deformable_polygon_prism(
                side_count, state, ref_mesh, center,
                radius, deformable_body_density, thickness,
                flat_orientation);
        }
    }

    // The general solver separates these independent nodes into cloth and
    // tetrahedral-solid blocks using ref_mesh.tet_nodes.
    ref_mesh.build_deformable_nodes();
}

// ---------------------------------------------------------------------------
// Example 16: ten alternating rigid and deformable polygonal prisms dropping
// onto one another above a cloth pinned along two opposite sides
// ---------------------------------------------------------------------------
// command line: ./build/3D_sim --example 16 --num_frames 200 --substeps 20 --max_substep_iters 20 --fixed_iters --outdir ten_rigid_solid_flat_stack_on_cloth_output --format obj --E 1e8 --d_hat 0.019 --k_barrier 500
// Has the same problem as Example 14.
void build_ten_alternating_rigid_solid_flat_stack_on_pinned_cloth_example(
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

    // Initialize the cloth before appending solid and rigid collision
    // triangles, so the shell-elastic triangles remain the leading prefix.
    constexpr int cloth_nx = 40;
    constexpr int cloth_nz = 40;
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

    constexpr int object_count = 10;
    constexpr double radius = 0.24;
    constexpr double thickness = 0.16;
    constexpr double lowest_center_y = 1.65;
    constexpr double center_spacing = 0.42;
    const Vec3 drop_velocity(0.0, -0.75, 0.0);

    // The material extrusion axis is +z. This exact -90 degree rotation about
    // x maps it to world +y, so every polygonal cap is horizontal. There is no
    // yaw or tilt on any rigid or deformable body.
    const Vec4 flat_orientation(
        std::cos(0.25 * kPi), -std::sin(0.25 * kPi), 0.0, 0.0);

    for (int object = 0; object < object_count; ++object) {
        const int side_count = 3 + object;
        const Vec3 center(
            0.0, lowest_center_y + center_spacing * object, 0.0);

        // Rigid and deformable objects alternate from the bottom upward. Their
        // initial gaps make the falling bodies contact the growing stack one
        // after another instead of starting in simultaneous contact.
        if (object % 2 == 0) {
            append_rigid_polygon(
                side_count, state, ref_mesh, center,
                radius, params.rigid_density, thickness,
                drop_velocity, flat_orientation, Vec3::Zero());
        } else {
            const int solid_base = append_deformable_polygon_prism(
                side_count, state, ref_mesh, center,
                radius, params.solid_density, thickness,
                flat_orientation);
            for (std::size_t node = static_cast<std::size_t>(solid_base);
                 node < state.velocities.size(); ++node) {
                state.velocities[node] = drop_velocity;
            }
        }
    }

    ref_mesh.build_deformable_nodes();
}
