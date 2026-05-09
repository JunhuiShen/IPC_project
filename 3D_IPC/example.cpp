#include "example.h"
#include "make_shape.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
// Example 3: dragon press -- one or two translating plates squeeze the dragon
// ---------------------------------------------------------------------------
// Defaults to xyzrgb_dragon_12k.obj (12k-vert decimation by
// tools/decimate_obj.py). Gravity is forced to zero so motion comes only
// from the moving plates. Per-triangle Dm_inverse and per-hinge c_e are
// rebuilt from the 3D rest pose because the dragon is a closed surface
// (the OBJ loader's xz-projection X collapses near-vertical triangles).
// Each plate's visual mesh sits at the SDF surface, eps_sdf from the
// dragon's force-free rest level. Three variants live behind the #if
// chain below; pick by flipping which block is enabled.
void build_dragon_squeeze_example(const IPCArgs3D& args,
                                  RefMesh& ref_mesh,
                                  DeformedState& state,
                                  std::vector<Vec2>& X,
                                  std::vector<Pin>& pins,
                                  SimParams& params,
                                  std::vector<Vec3>& static_x,
                                  std::vector<int>&  static_tris,
                                  DragonSqueezeSpec& spec) {
    clear_model(ref_mesh, state, X, pins);
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();

    params.gravity = Vec3::Zero();

    const int dragon_v_begin = static_cast<int>(state.deformed_positions.size());
    const int dragon_t_begin = num_tris(ref_mesh);

    load_obj_mesh(args.dragon_path, ref_mesh, state, X,
                  args.dragon_scale, Vec3::Zero());

    const int dragon_v_end = static_cast<int>(state.deformed_positions.size());
    const int dragon_t_end = num_tris(ref_mesh);

    // Translate the loaded dragon so its AABB is centered at x=z=0 and its
    // lowest vertex sits at dragon_drop_y.
    Vec3 lo( std::numeric_limits<double>::max(),
             std::numeric_limits<double>::max(),
             std::numeric_limits<double>::max());
    Vec3 hi(-std::numeric_limits<double>::max(),
            -std::numeric_limits<double>::max(),
            -std::numeric_limits<double>::max());
    for (int v = dragon_v_begin; v < dragon_v_end; ++v) {
        const Vec3& p = state.deformed_positions[v];
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    const Vec3 center = 0.5 * (lo + hi);
    const Vec3 shift(-center.x(),
                      args.dragon_drop_y - lo.y(),
                     -center.z());
    for (int v = dragon_v_begin; v < dragon_v_end; ++v) {
        state.deformed_positions[v] += shift;
    }
    const double dragon_y_lo = lo.y() + shift.y();
    const double dragon_y_hi = hi.y() + shift.y();

    rebuild_triangle_rest_isometric(ref_mesh, state.deformed_positions,
                                    dragon_t_begin, dragon_t_end);
    rebuild_hinge_c_e_3d(ref_mesh, state.deformed_positions,
                         dragon_v_begin, dragon_v_end);

    std::vector<std::pair<double, int>> by_y;
    by_y.reserve(dragon_v_end - dragon_v_begin);
    for (int v = dragon_v_begin; v < dragon_v_end; ++v) {
        by_y.emplace_back(state.deformed_positions[v].y(), v);
    }
    const int n_pin = std::clamp(args.dragon_anchor_pin_count, 1,
                                 static_cast<int>(by_y.size()));

    const double gw    = args.dragon_ground_size;
    const int    gn    = std::max(1, args.dragon_ground_subdiv);
    const double speed = std::abs(args.dragon_squeeze_speed);

    spec = DragonSqueezeSpec{};
    spec.rise_max = std::max(0.0, args.dragon_squeeze_max);
    spec.t_settle = std::max(0.0, args.dragon_squeeze_settle);
    spec.t_hold   = args.dragon_squeeze_hold;

    // Append one moving plate (SDF + visual mesh) and record the indexing
    // the spec needs to translate it later.
    auto add_plate = [&](double sdf_y, const Vec3& normal, const Vec3& vel) {
        DragonSqueezePlate p;
        p.plane_index = static_cast<int>(params.sdf_planes.size());
        params.sdf_planes.push_back(PlaneSDF{Vec3(0.0, sdf_y, 0.0), normal});
        p.plane_point_rest = params.sdf_planes.back().point;
        p.rise_velocity    = vel;

        p.visual_v_begin = static_cast<int>(static_x.size());
        {
            RefMesh           s_ref;
            DeformedState     s_state;
            std::vector<Vec2> s_X;
            build_square_mesh(s_ref, s_state, s_X, gn, gn, gw, gw,
                              Vec3(-0.5 * gw, sdf_y, -0.5 * gw));
            const int base_v = static_cast<int>(static_x.size());
            for (const Vec3& q : s_state.deformed_positions) static_x.push_back(q);
            for (int t : s_ref.tris) static_tris.push_back(base_v + t);
        }
        p.visual_v_end = static_cast<int>(static_x.size());
        p.visual_v_rest.assign(static_x.begin() + p.visual_v_begin,
                               static_x.begin() + p.visual_v_end);
        spec.plates.push_back(std::move(p));
    };

#if 0
    // === BOTTOM-RISE: floor rises into a top-pinned dragon ====================
    // Plane normal +y; force-free rest at phi=eps_sdf is eps_sdf ABOVE the
    // SDF surface, so the visual floor sits the same eps_sdf below the
    // dragon's resting bottom.
    std::sort(by_y.begin(), by_y.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int i = 0; i < n_pin; ++i) {
        append_pin(pins, by_y[i].second, state.deformed_positions);
    }
    add_plate(dragon_y_lo - args.dragon_squeeze_gap - params.eps_sdf,
              Vec3(0.0,  1.0, 0.0),
              Vec3(0.0,  speed, 0.0));

#elif 0
    // === TOP-DESCENT: ceiling descends onto a bottom-pinned dragon ============
    // Plane normal -y so phi(x) = point.y - x.y; force-free rest at
    // phi=eps_sdf is eps_sdf BELOW the SDF surface (mirror of the floor
    // case), which puts the visual ceiling at the right wall position.
    std::sort(by_y.begin(), by_y.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (int i = 0; i < n_pin; ++i) {
        append_pin(pins, by_y[i].second, state.deformed_positions);
    }
    add_plate(dragon_y_hi + args.dragon_squeeze_gap + params.eps_sdf,
              Vec3(0.0, -1.0, 0.0),
              Vec3(0.0, -speed, 0.0));

#else
    // === BOTH-ENDS (active): floor + ceiling press toward the middle ==========
    // No anchor pins -- the two opposing plates' equal-and-opposite forces
    // cancel at the COM by symmetry, so the dragon stays centered.
    // dragon_anchor_pin_count is unused here.
    (void)n_pin;
    add_plate(dragon_y_lo - args.dragon_squeeze_gap - params.eps_sdf,
              Vec3(0.0,  1.0, 0.0),
              Vec3(0.0,  speed, 0.0));
    add_plate(dragon_y_hi + args.dragon_squeeze_gap + params.eps_sdf,
              Vec3(0.0, -1.0, 0.0),
              Vec3(0.0, -speed, 0.0));
#endif

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
}

namespace {
// Plate displacement at virtual time s = max(0, t - t_settle).
// t_hold < 0 : press at rise_velocity, capped at rise_max (legacy press-and-stay).
// t_hold >= 0: trapezoid -- press to rise_max, hold t_hold s, retract to 0.
Vec3 dragon_squeeze_displacement(const DragonSqueezePlate& p,
                                 double s, double rise_max, double t_hold) {
    if (t_hold < 0.0) {
        Vec3 d = p.rise_velocity * s;
        if (rise_max > 0.0) {
            const double mag = d.norm();
            if (mag > rise_max) d *= (rise_max / mag);
        }
        return d;
    }
    const double v = p.rise_velocity.norm();
    if (v == 0.0 || rise_max <= 0.0) return Vec3::Zero();
    const double t_press = rise_max / v;
    double phase_s;
    if (s <= t_press)                       phase_s = s;
    else if (s <= t_press + t_hold)         phase_s = t_press;
    else if (s <= 2.0 * t_press + t_hold)   phase_s = t_press - (s - t_press - t_hold);
    else                                    phase_s = 0.0;
    return p.rise_velocity * phase_s;
}
}

void update_dragon_squeeze_sdf(SimParams& params,
                               const DragonSqueezeSpec& spec, double t) {
    const double s = std::max(0.0, t - spec.t_settle);
    for (const DragonSqueezePlate& p : spec.plates) {
        if (p.plane_index < 0 ||
            p.plane_index >= static_cast<int>(params.sdf_planes.size())) continue;
        params.sdf_planes[p.plane_index].point =
            p.plane_point_rest + dragon_squeeze_displacement(p, s, spec.rise_max, spec.t_hold);
    }
}

void update_dragon_squeeze_visual(std::vector<Vec3>& static_x,
                                  const DragonSqueezeSpec& spec, double t) {
    const double s = std::max(0.0, t - spec.t_settle);
    for (const DragonSqueezePlate& p : spec.plates) {
        const Vec3 d = dragon_squeeze_displacement(p, s, spec.rise_max, spec.t_hold);
        for (int i = p.visual_v_begin; i < p.visual_v_end; ++i) {
            static_x[i] = p.visual_v_rest[i - p.visual_v_begin] + d;
        }
    }
}


// Example 4: rectangular cloth (tu_width x tu_size) draped under one cylinder
// (axis +x, at (0, sheet_y, 0)). Pre-pose: back drop -> bottom-wrap semicircle
// -> front drop, so j=0 and j=ny rows end at y=corner_y on either side of the
// cylinder. Static pins: the 4 outer corners (i in {0,nx}, j in {0,ny}). Wrap
// pins: every vertex on the bottom-semicircle rows; their targets and the SDF
// axis both yaw about +y in lock-step (same convention as example 2), so the
// cloth twists between the rotating wrap and the fixed corners.
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
    const double pin_r    = r + std::max(params.eps_sdf, 1e-3);
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

    // Pinning: wrap rows co-rotate with the cylinder; only the 4 outer corners
    // of the top edges are held static (interior of the top edges is free, so
    // the cloth sags between the corners like a hammock).
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
            for (int i : {0, nx}) {
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
    // Re-snap the 4 corner pins each step so a restart from any frame
    // recovers their static targets.
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

void update_twist_untwist_visual(std::vector<Vec3>& static_x,
                                 const TwistUntwistSpec& spec, double t) {
    const double theta = effective_theta(spec.omega, t, spec.t_settle, spec.t_ramp,
                                         spec.max_abs_theta, spec.untwist, spec.t_hold);
    for (int i = spec.visual_v_begin; i < spec.visual_v_end; ++i) {
        static_x[i] = rotate_about_y_axis(spec.visual_v_rest[i - spec.visual_v_begin],
                                          spec.cyl_axis_point, theta);
    }
}
