#include "example.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "rigid_body_ipc.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
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
// Commend line: ./build/3D_sim --example 5 --num_frames 500 --tol_abs 1e-12 --tol_rel 1e-10 --outdir racket_output 
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
        Vec3{1.0, 0.0, 0.0}, 1, ref_mesh, state);

    append_rigid_polygon(
        6, state, ref_mesh,
        Vec3(0.55, 5.0, 0.0),
        /*radius=*/0.22,
        /*density=*/30.0,
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
