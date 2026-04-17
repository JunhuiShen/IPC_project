#include "example.h"
#include "make_shape.h"

#include <cmath>

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
            Vec3(args.left_x, args.sheet_y, args.left_z)
    );

    const int base_right = build_square_mesh(
            ref_mesh, state, X,
            args.nx, args.ny, args.width, args.height,
            Vec3(args.right_x, args.sheet_y, args.right_z)
    );

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

namespace {
// Shared ground-cloth builder: 1.2x1.2 square in the xz plane at y=0, four
// corners pinned. Returns nothing; appends to ref_mesh/state/X/pins in place.
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

void build_cloth_stack_example_high_res(RefMesh& ref_mesh,
                                        DeformedState& state,
                                        std::vector<Vec2>& X,
                                        std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);

    build_ground_cloth(ref_mesh, state, X, pins);

    // Five densely-spaced cloths with initial downward velocity so layers
    // engage contact near-simultaneously -- the actual stress window.
    const int    stack_count   = 5;    // number of falling cloths in the stack
    const int    small_nx      = 16;    // grid subdivisions along each cloth's x-axis (triangles = 2*nx*ny)
    const int    small_ny      = 16;    // grid subdivisions along each cloth's y-axis
    const double small_w       = 0.35; // width of each falling cloth (meters, along x)
    const double small_h       = 0.35; // height of each falling cloth (meters, along z in world space)
    const double first_drop_y  = 0.25; // y-coordinate of the lowest falling cloth at t=0
    const double drop_spacing  = 0.09; // vertical gap (meters) between successive stacked cloths at t=0
    const double drop_speed    = 2.0;  // initial downward velocity applied to every falling-cloth vertex (m/s)

    const int falling_begin = static_cast<int>(state.deformed_positions.size());

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
    for (int i = falling_begin; i < static_cast<int>(state.velocities.size()); ++i) {
        state.velocities[i] = Vec3(0.0, -drop_speed, 0.0);
    }
}

void build_cloth_cylinder_drop_example(RefMesh& ref_mesh,
                                       DeformedState& state,
                                       std::vector<Vec2>& X,
                                       std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);

    // Bigger ground cloth than the shared 1.2x1.2 helper so the falling pile
    // has room to settle without involving the pinned corners. Only the
    // corners are pinned -- the interior is left free so its sagging-under-
    // gravity gradient stays above tol_abs and keeps the global solver iterating.
    const int    ground_nx = 40;
    const int    ground_ny = 40;
    const double ground_w  = 4.0;
    const double ground_h  = 4.0;
    const double ground_y  = 0.0;

    const int ground_base = build_square_mesh(
            ref_mesh, state, X,
            ground_nx, ground_ny, ground_w, ground_h,
            Vec3(-0.5 * ground_w, ground_y, -0.5 * ground_h));

    // Pin every ground vertex so the ground behaves as a static collider
    // (experimentally checking the fully-pinned regime).
    const int ground_vertex_count = (ground_nx + 1) * (ground_ny + 1);
    for (int k = 0; k < ground_vertex_count; ++k) {
        append_pin(pins, ground_base + k, state.deformed_positions);
    }

    // Horizontal cylinder (long axis +z) acting as a static collider via
    // Dirichlet pins on every vertex.
    const int    cyl_nu     = 8;     // circumferential subdivisions (octagon cross-section)
    const int    cyl_nv     = 12;    // axial subdivisions
    const double cyl_radius = 0.03;
    const double cyl_length = 0.6;   // longer than the cloth width so cloth ends overhang both faces
    const Vec3   cyl_center(0.0, 0.25, 0.0);

    const int cyl_base = build_cylinder_mesh(
            ref_mesh, state, X,
            cyl_nu, cyl_nv, cyl_radius, cyl_length, cyl_center);

    // Pin every cylinder vertex so the cylinder behaves as a static collider.
    const int cyl_vertex_count = cyl_nu * (cyl_nv + 1);
    for (int k = 0; k < cyl_vertex_count; ++k) {
        append_pin(pins, cyl_base + k, state.deformed_positions);
    }

    // Vertical cloth stack dropped onto the cylinder -- same pattern as
    // build_cloth_stack_example_high_res, with the cylinder catching them
    // mid-fall before they reach the ground.
    const int    stack_count   = 50;   // number of falling cloths in the stack
    const int    small_nx      = 16;   // grid subdivisions along each cloth's x-axis (triangles = 2*nx*ny)
    const int    small_ny      = 16;   // grid subdivisions along each cloth's y-axis
    const double small_w       = 0.70; // width of each falling cloth (meters, along x)
    const double small_h       = 0.70; // height of each falling cloth (meters, along z in world space)
    const double first_drop_y  = 1.00; // y-coordinate of the lowest falling cloth at t=0
    const double drop_spacing  = 0.05; // vertical gap (meters) between successive stacked cloths at t=0

    for (int s = 0; s < stack_count; ++s) {
        const Vec3 origin(
                -0.5 * small_w,
                first_drop_y + s * drop_spacing,
                -0.5 * small_h);
        build_square_mesh(ref_mesh, state, X, small_nx, small_ny, small_w, small_h, origin);
    }

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
}


namespace {
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
} // namespace

void build_twisting_cloth_example(const IPCArgs3D& args,
                                  RefMesh& ref_mesh,
                                  DeformedState& state,
                                  std::vector<Vec2>& X,
                                  std::vector<Pin>& pins,
                                  TwistSpec& spec) {
    clear_model(ref_mesh, state, X, pins);

    // Square 100x100 grid with 8 total relative turns.
    const int    nx     = 99;
    const int    ny     = 99;
    const double width  = 2.5;
    const double height = 2.5;
    const double y0     = args.sheet_y;

    // Center the cloth so the midline (+x axis at y = y0, z = 0) passes
    // through the mesh centroid.
    const Vec3 origin(-0.5 * width, y0, -0.5 * height);
    const int base = build_square_mesh(ref_mesh, state, X, nx, ny, width, height, origin);

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    const double duration = static_cast<double>(args.num_frames) / args.fps;
    const double omega = (kTwoPi * args.twist_turns) / (2.0 * duration);

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

