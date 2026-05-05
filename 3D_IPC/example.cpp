#include "example.h"
#include "make_shape.h"

#include <cmath>

namespace {
// ---------------------------------------------------------------------------
// Shared helpers used across examples.
// ---------------------------------------------------------------------------

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;

// Rotate `p` by angle `theta` about the line through `axis_point` parallel
// to +x. Only the y and z components change.
Vec3 rotate_about_x_axis(const Vec3& p, const Vec3& axis_point, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double dy = p.y() - axis_point.y();
    const double dz = p.z() - axis_point.z();
    return Vec3(p.x(),
                axis_point.y() + c * dy - s * dz,
                axis_point.z() + s * dy + c * dz);
}

// Rotate `p` by angle `theta` about the line through `axis_point` parallel
// to +y. Only the x and z components change.
Vec3 rotate_about_y_axis(const Vec3& p, const Vec3& axis_point, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double dx = p.x() - axis_point.x();
    const double dz = p.z() - axis_point.z();
    return Vec3(axis_point.x() + c * dx + s * dz,
                p.y(),
                axis_point.z() - s * dx + c * dz);
}

// Effective rotation angle: zero during the settle phase, then quadratic
// (linear-ramp on omega) over the ramp phase, then linear afterward so the
// pin target velocity is continuous and matches a steady omega. If
// `max_abs_theta` is positive, the magnitude is clamped to that cap so the
// cylinder coasts and then stops at the target angle.
double effective_theta(double omega, double t, double t_settle, double t_ramp,
                       double max_abs_theta = 0.0) {
    double theta = 0.0;
    if (t > t_settle) {
        const double s = t - t_settle;
        if (t_ramp <= 0.0)            theta = omega * s;
        else if (s >= t_ramp)         theta = omega * (s - 0.5 * t_ramp);
        else                          theta = 0.5 * omega * s * s / t_ramp;
    }
    if (max_abs_theta > 0.0) {
        if (theta >  max_abs_theta) theta =  max_abs_theta;
        if (theta < -max_abs_theta) theta = -max_abs_theta;
    }
    return theta;
}

// Shared ground-cloth builder used by examples 2 and 3: 1.2x1.2 square in the
// xz plane at y=0 with the four corners pinned. Appends to ref_mesh / state /
// X / pins in place.
void build_ground_cloth(RefMesh& ref_mesh,
                        DeformedState& state,
                        std::vector<Vec2>& X,
                        std::vector<Pin>& pins) {
    const int    ground_nx = 10;
    const int    ground_ny = 10;
    const double ground_w  = 1.2;
    const double ground_h  = 1.2;
    const double ground_y  = 0.0;

    const int ground_base = build_square_mesh(
            ref_mesh, state, X,
            ground_nx, ground_ny, ground_w, ground_h,
            Vec3(-0.5 * ground_w, ground_y, -0.5 * ground_h));

    const int ground_c00 = ground_base + 0;
    const int ground_c10 = ground_base + ground_nx;
    const int ground_c01 = ground_base + ground_ny * (ground_nx + 1);
    const int ground_c11 = ground_base + ground_ny * (ground_nx + 1) + ground_nx;

    append_pin(pins, ground_c00, state.deformed_positions);
    append_pin(pins, ground_c10, state.deformed_positions);
    append_pin(pins, ground_c01, state.deformed_positions);
    append_pin(pins, ground_c11, state.deformed_positions);
}
} // namespace


// ---------------------------------------------------------------------------
// Example 1: two side-by-side sheets with their inner corners pinned.
// ---------------------------------------------------------------------------
void build_two_sheets_example(const IPCArgs3D& args,
                              RefMesh& ref_mesh,
                              DeformedState& state,
                              std::vector<Vec2>& X,
                              std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);

    // Two sheets, side-by-side in x, embedded in the xz plane by build_square_mesh().
    const int base_left = build_square_mesh(
            ref_mesh, state, X,
            args.nx, args.ny, args.width, args.height,
            Vec3(args.left_x, args.sheet_y, args.left_z));

    const int base_right = build_square_mesh(
            ref_mesh, state, X,
            args.nx, args.ny, args.width, args.height,
            Vec3(args.right_x, args.sheet_y, args.right_z));

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    // Tiny asymmetry on the right sheet so the evolution is not perfectly symmetric.
    state.deformed_positions[base_right + 0] += Vec3(-0.02, 0.00, -0.01);
    state.deformed_positions[base_right + 2] += Vec3(-0.02, 0.00, -0.01);

    // Pin inner-side top and bottom corners.
    const int left_bottom_right = base_left + args.nx;
    const int left_top_right    = base_left + args.ny * (args.nx + 1) + args.nx;
    const int right_bottom_left = base_right + 0;
    const int right_top_left    = base_right + args.ny * (args.nx + 1);

    append_pin(pins, left_top_right,    state.deformed_positions);
    append_pin(pins, left_bottom_right, state.deformed_positions);
    append_pin(pins, right_top_left,    state.deformed_positions);
    append_pin(pins, right_bottom_left, state.deformed_positions);
}


// ---------------------------------------------------------------------------
// Example 2: ground cloth + 3 small cloths dropped from rest. Low-res, cheap.
// ---------------------------------------------------------------------------
void build_cloth_stack_example_low_res(RefMesh& ref_mesh,
                                       DeformedState& state,
                                       std::vector<Vec2>& X,
                                       std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);

    build_ground_cloth(ref_mesh, state, X, pins);

    // Three small cloths dropped from rest, well-separated so they impact one
    // at a time. Coarse nx=ny=4 keeps the mesh tiny for quick sanity runs.
    const int    stack_count   = 3;
    const int    small_nx      = 4;
    const int    small_ny      = 4;
    const double small_w       = 0.30;
    const double small_h       = 0.30;
    const double first_drop_y  = 0.30;
    const double drop_spacing  = 0.22;

    for (int s = 0; s < stack_count; ++s) {
        // Tiny asymmetric xz offset so perfect symmetry does not bias stacking.
        const double x_off = 0.005 * (s - 1);
        const double z_off = 0.004 * (s - 1);
        const Vec3 origin(
                -0.5 * small_w + x_off,
                first_drop_y + s * drop_spacing,
                -0.5 * small_h + z_off);
        build_square_mesh(ref_mesh, state, X,
                          small_nx, small_ny, small_w, small_h, origin);
    }

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
}


// ---------------------------------------------------------------------------
// Example 3: ground cloth + 5 dense, fast-falling cloths. High-res stress test.
// ---------------------------------------------------------------------------
void build_cloth_stack_example_high_res(RefMesh& ref_mesh,
                                        DeformedState& state,
                                        std::vector<Vec2>& X,
                                        std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);

    build_ground_cloth(ref_mesh, state, X, pins);

    // Five densely-spaced cloths with initial downward velocity so layers
    // engage contact near-simultaneously -- the actual stress window.
    const int    stack_count   = 5;
    const int    small_nx      = 16;
    const int    small_ny      = 16;
    const double small_w       = 0.35;
    const double small_h       = 0.35;
    const double first_drop_y  = 0.25;
    const double drop_spacing  = 0.09;
    const double drop_speed    = 2.0;  // initial -y velocity (m/s)

    const int falling_begin = static_cast<int>(state.deformed_positions.size());

    for (int s = 0; s < stack_count; ++s) {
        const double x_off = 0.005 * (s - 1);
        const double z_off = 0.004 * (s - 1);
        const Vec3 origin(
                -0.5 * small_w + x_off,
                first_drop_y + s * drop_spacing,
                -0.5 * small_h + z_off);
        build_square_mesh(ref_mesh, state, X,
                          small_nx, small_ny, small_w, small_h, origin);
    }

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    for (int i = falling_begin; i < static_cast<int>(state.velocities.size()); ++i) {
        state.velocities[i] = Vec3(0.0, -drop_speed, 0.0);
    }
}


// ---------------------------------------------------------------------------
// Example 4: cloth stack drops onto a static cylinder + pinned ground.
// Cylinder/ground are exported once as static_x/static_tris; collisions go
// through SDFs (sdf_planes + sdf_cylinders).
// ---------------------------------------------------------------------------
void build_cloth_cylinder_drop_example(const IPCArgs3D& args,
                                       RefMesh& ref_mesh,
                                       DeformedState& state,
                                       std::vector<Vec2>& X,
                                       std::vector<Pin>& pins,
                                       SimParams& params,
                                       std::vector<Vec3>& static_x,
                                       std::vector<int>&  static_tris) {
    clear_model(ref_mesh, state, X, pins);
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();

    const double ground_w  = args.cyl_ground_size;
    const double ground_h  = args.cyl_ground_size;
    const double ground_y  = 0.0;

    const int    cyl_nu     = args.cyl_nu;
    const double cyl_radius = args.cyl_radius;
    const double cyl_length = args.cyl_length;
    const Vec3   cyl_center(args.cyl_cx, args.cyl_cy, args.cyl_cz);

    // Build ground and cylinder into a scratch mesh for one-time export only.
    {
        RefMesh        s_ref;
        DeformedState  s_state;
        std::vector<Vec2> s_X;
        build_square_mesh(s_ref, s_state, s_X,
                          40, 40, ground_w, ground_h,
                          Vec3(-0.5 * ground_w, ground_y, -0.5 * ground_h));
        build_cylinder_mesh(s_ref, s_state, s_X,
                            cyl_nu, cyl_radius, cyl_length, cyl_center);
        static_x    = s_state.deformed_positions;
        static_tris = s_ref.tris;
    }

    params.sdf_planes.push_back(PlaneSDF{
            Vec3(0.0, ground_y - 2.0 * params.eps_sdf, 0.0),
            Vec3(0.0, 1.0, 0.0)});
    params.sdf_cylinders.push_back(CylinderSDF{
            cyl_center, Vec3(0.0, 0.0, 1.0), cyl_radius});

    // Vertical cloth stack dropped onto the cylinder -- same pattern as
    // build_cloth_stack_example_high_res, with the cylinder catching them
    // mid-fall before they reach the ground.
    const int    stack_count   = args.drop_stack_count;
    const int    small_nx      = args.drop_cloth_nx;
    const int    small_ny      = args.drop_cloth_ny;
    const double small_w       = args.drop_cloth_w;
    const double small_h       = args.drop_cloth_h;
    const double first_drop_y  = args.drop_first_y;
    const double drop_spacing  = args.drop_spacing;

    for (int s = 0; s < stack_count; ++s) {
        const Vec3 origin(
                -0.5 * small_w,
                first_drop_y + s * drop_spacing,
                -0.5 * small_h);
        build_square_mesh(ref_mesh, state, X, small_nx, small_ny, small_w, small_h, origin);
    }

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
}


// ---------------------------------------------------------------------------
// Example 5: square cloth with both short edges clamped, edges counter-rotate
// about the +x axis at twist_rate Hz.
// ---------------------------------------------------------------------------
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

    // Center the cloth so the midline (+x axis at y = y0, z = 0) passes
    // through the mesh centroid.
    const Vec3 origin(-0.5 * width, y0, -0.5 * height);
    const int base = build_square_mesh(ref_mesh, state, X, nx, ny, width, height, origin);

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    // Each side spins at +/- omega; relative rate is 2*omega, so args.twist_rate
    // means omega = pi * rate.
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

    // Pin the full short edge on each side. Every vertex has rotation radius
    // |z|, so the outer vertices drive the twist while the middle of the edge
    // sits on the axis and barely moves. build_square_mesh stores grid (i, j)
    // at base + j * (nx + 1) + i.
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
// Example 6: cloth stack drops onto a static sphere + pinned ground.
// ---------------------------------------------------------------------------
void build_cloth_sphere_drop_example(const IPCArgs3D& args,
                                     RefMesh& ref_mesh,
                                     DeformedState& state,
                                     std::vector<Vec2>& X,
                                     std::vector<Pin>& pins,
                                     SimParams& params,
                                     std::vector<Vec3>& static_x,
                                     std::vector<int>&  static_tris) {
    clear_model(ref_mesh, state, X, pins);
    params.sdf_planes.clear();
    params.sdf_cylinders.clear();
    params.sdf_spheres.clear();

    const double ground_w  = args.sphere_ground_size;
    const double ground_h  = args.sphere_ground_size;
    const double ground_y  = 0.0;
    const int    ground_nx = args.sphere_ground_nx;
    const int    ground_ny = args.sphere_ground_nx;

    const int    sphere_subdiv = args.sphere_subdiv;
    const double sphere_radius = args.sphere_radius;
    const Vec3   sphere_center(args.sphere_cx, args.sphere_cy, args.sphere_cz);

    // Build ground and sphere into a scratch mesh for one-time export only.
    // Collisions go through SDFs below; nothing here joins ref_mesh.
    {
        RefMesh        s_ref;
        DeformedState  s_state;
        std::vector<Vec2> s_X;
        build_square_mesh(s_ref, s_state, s_X,
                          ground_nx, ground_ny, ground_w, ground_h,
                          Vec3(-0.5 * ground_w, ground_y, -0.5 * ground_h));
        build_sphere_mesh(s_ref, s_state, s_X,
                          sphere_subdiv, sphere_radius, sphere_center);
        static_x    = s_state.deformed_positions;
        static_tris = s_ref.tris;
    }

    params.sdf_planes.push_back(PlaneSDF{
            Vec3(0.0, ground_y - 2.0 * params.eps_sdf, 0.0),
            Vec3(0.0, 1.0, 0.0)});
    params.sdf_spheres.push_back(SphereSDF{sphere_center, sphere_radius});

    const int    stack_count   = args.drop_stack_count;
    const int    small_nx      = args.drop_cloth_nx;
    const int    small_ny      = args.drop_cloth_ny;
    const double small_w       = args.sphere_cloth_size;
    const double small_h       = args.sphere_cloth_size;
    const double first_drop_y  = args.drop_first_y;
    const double drop_spacing  = args.drop_spacing;

    for (int s = 0; s < stack_count; ++s) {
        const Vec3 origin(
                -0.5 * small_w,
                first_drop_y + s * drop_spacing,
                -0.5 * small_h);
        build_square_mesh(ref_mesh, state, X, small_nx, small_ny, small_w, small_h, origin);
    }

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
}


// ---------------------------------------------------------------------------
// Example 7: N closed-loop cloth strips wrapping two horizontal cylinders.
// Both cylinders rotate about +y in opposite directions, twisting the strips
// together in the gap between them.
// ---------------------------------------------------------------------------
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
    params.sdf_spheres.clear();

    const int    n_strips = std::max(1, args.tcyl_n_strips);
    const int    nx       = args.tcyl_nx;
    const int    ny       = args.tcyl_ny;
    const double strip_w  = args.tcyl_strip_w;
    const double H        = 0.5 * args.tcyl_cloth_h;       // half y-distance between cyl axes
    const double r        = args.tcyl_radius;
    const double omega    = kTwoPi * args.tcyl_twist_rate;

    const Vec3 top_center(0.0,  H, 0.0);
    const Vec3 bot_center(0.0, -H, 0.0);

    // Each strip is a CLOSED LOOP wrapping both horizontal cylinders, like
    // a flat belt on two pulleys. Going around the loop:
    //   1. Top wrap   (length π·pin_r): from back of top cyl OVER the top
    //                                   to the front of top cyl.  Pinned.
    //   2. Front drop (length 2H):     straight down at z=+pin_r.  Free.
    //   3. Bot wrap   (length π·pin_r): from front of bot cyl UNDER the
    //                                   bottom to the back of bot cyl.  Pinned.
    //   4. Back drop  (length 2H):     straight up at z=-pin_r.  Free.
    // Wrap-to-drop transitions are tangent-continuous (no curvature kink),
    // so uniform parametric sampling stays uniform after the 3D remap.
    //
    // pin_r is slightly larger than r so the flat cloth chords between wrap
    // rows sit outside the cylinder mesh chords (no polygon-vs-polygon
    // interpenetration at frame 0). 2 mm is visually flush.
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

    // Visual cylinder meshes along +x (build_cylinder_mesh creates them
    // along +z natively; the (x,y,z) → (z,y,x) swap rotates onto +x).
    // Render the cyl mesh thinner than the physical radius r by
    // `tcyl_visual_shrink` so dynamic drop-row lag during rotation never
    // visually pokes through the cylinder. Physics is unaffected.
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
    spec.static_x_rest  = static_x;
    spec.top_v_begin    = top_v_begin;
    spec.top_v_end      = top_v_end;
    spec.bot_v_begin    = top_v_end;
    spec.bot_v_end      = bot_v_end;

    // The last panel row (j == ny) lands at s = loop_L, which geometrically
    // coincides with s = 0 (the back drop closes onto the top-wrap start).
    // To anchor the closed loop without two pinned vertices at the same
    // point (zero distance ⇒ barrier gradient blows up), we pin j=ny to
    // the top cylinder but nudge it outward in -z by `seam_offset` and
    // snap its initial position to match so there's no startup yank.
    // 5 mm is comfortably above any reasonable d_hat (cloth-cloth barrier).
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

        // Pin the wrap rows. Treat j=ny as part of the top wrap (s = 0)
        // with the seam offset applied.
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

    // Re-initialize hinges with the curved 3D rest pose so bending energy
    // is zero in the wrapped configuration (otherwise the strips would try
    // to flatten themselves).
    ref_mesh.initialize(X, state.deformed_positions);

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
}

void update_cylinder_twist_pins(std::vector<Pin>& pins,
                                const CylinderTwistSpec& spec, double t) {
    const double theta_top = effective_theta(spec.omega_top, t, spec.t_settle, spec.t_ramp, spec.max_abs_theta);
    const double theta_bot = effective_theta(spec.omega_bot, t, spec.t_settle, spec.t_ramp, spec.max_abs_theta);
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
    const double theta_top = effective_theta(spec.omega_top, t, spec.t_settle, spec.t_ramp, spec.max_abs_theta);
    const double theta_bot = effective_theta(spec.omega_bot, t, spec.t_settle, spec.t_ramp, spec.max_abs_theta);
    for (int i = spec.top_v_begin; i < spec.top_v_end; ++i) {
        static_x[i] = rotate_about_y_axis(spec.static_x_rest[i], spec.top_axis_point, theta_top);
    }
    for (int i = spec.bot_v_begin; i < spec.bot_v_end; ++i) {
        static_x[i] = rotate_about_y_axis(spec.static_x_rest[i], spec.bot_axis_point, theta_bot);
    }
}
