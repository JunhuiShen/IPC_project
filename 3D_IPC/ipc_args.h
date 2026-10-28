#pragma once

#include "args.h"
#include "physics.h"
#include "output.h"

#include <cmath>
#include <stdexcept>
#include <string>

// CLI-facing view of SimParams (with scene-placement knobs on top).
struct IPCArgs3D : ArgParser {

    // --- time integration ---
    double fps          = 30.0;
    int    substeps     = 3;
    int    num_frames   = 60;

    // --- physics ---
    double E            = 1e6;    // Pa  (Young's modulus)
    double nu           = 0.3;    // (Poisson ratio)
    double density      = 900.0;  // kg/m^3
    double thickness    = 0.001;  // m
    double solid_E      = 5e4;    // Pa  (soft rubber-toy stiffness)
    double solid_nu     = 0.45;   // nearly incompressible jelly-like response
    double solid_density = 900.0; // kg/m^3 (tetrahedral solids)
    double rigid_density = 900.0; // kg/m^3 (rigid bodies)
    double kB           = 1e-3;   // J   (bending stiffness; 0 disables bending)
    double kpin         = 1e5;    // N/m (pin spring stiffness)
    double gx           = 0.0;    // m/s^2
    double gy           = -9.81;  // m/s^2
    double gz           = 0.0;    // m/s^2

    // --- solver ---
    int    max_substep_iters = 500;  // per-substep cap; per-frame total = max_substep_iters * substeps
    double tol_abs       = 1e-6;  // residual norm tolerance
    double tol_rel       = 1e-3;  // relative tolerance: stop when residual < tol_rel * initial
    double d_hat         = 0.005; // barrier activation distance; 0 disables contact
    double friction_coefficient = 0.0; // dynamic Coulomb coefficient for mesh and analytic-SDF contact; 0 disables
    double friction_velocity_epsilon = 0.01; // m/s smoothing threshold for contact friction
    double k_sdf         = 1e5;   // SDF penalty stiffness; 0 disables the SDF term
    double eps_sdf       = 0.002; // SDF soft-barrier range (m); cloth rest at phi=eps_sdf. 0 = hard quadratic at the surface.
    bool   use_parallel  = true;
    bool   use_basic_experimental = false;
    bool   use_simd = false;
    bool   use_jacobi_dat = false;
    double dat_gamma = 0.9;
    double dat_query_radius = 0.0;
    double dat_node_box_padding = 1.2;
    bool   use_cloth_grid = false;
    bool   cloth_grid_auto_dx = false;
    double cloth_grid_dx = 0.05;
    bool   write_substeps = false;
    bool   use_ccd       = true;
    bool   use_ccd_guess = true;
    bool   use_colored_ccd_guess = false;
    bool   use_colored_ccd_guess_level_based = false;
    int    colored_ccd_guess_iters = 10;
    double colored_ccd_guess_fraction = 1.0;
    bool   use_verlet_guess = false;
    bool   use_translation_guess = false;
    bool   use_ogc = false;
    bool   use_ogc_solver = false;
    double ogc_box_pad = 0.005;
    bool   fixed_iters = false;
    bool   verbose = false;
    double node_box_max            = 0.01;
    double node_box_min            = 0.001;
    double theta_box_min           = 0.01;
    double theta_box_max           = 0.1;
    int    node_box_update_count   = 10;
    double k_barrier                   = 100.0;
    double damping                     = 1.0;    
    bool   use_ticcd                   = false;    // true: Tight-Inclusion library | false: self-written linear CCD
    bool   exact_computation_fallback  = true;
    bool   ccd_exact_step_bound        = false;

    // --- scene selection ---
    int         example      = 1;   // Built-in scene ID
    double      sheet_y      = 0.20; // m -- midline y for example 1

    // example 1: square cloth twisted in place
    double      twist_rate   = 0.5; // relative twist rate in Hz (full turns per second)
    int         twist_nx     = 99;  // grid subdivisions along x
    int         twist_ny     = 99;  // grid subdivisions along y
    double      twist_size   = 2.5; // edge length (m)

    // example 2: N closed-loop cloth strips twisted between two horizontal cylinders
    int         tcyl_n_strips    = 4;      // number of cloth strips along x (along the cylinder axis)
    double      tcyl_strip_w     = 0.18;   // each strip's x-width (m, along beam axis)
    double      tcyl_strip_span_z= 1.20;   // total x-span over which strips are distributed (along cyl axis, m)
    double      tcyl_cloth_h     = 1.0;    // pin-to-pin y distance between cylinders (m)
    int         tcyl_nx          = 16;     // strip subdivisions along x (cols = tcyl_nx+1)
    int         tcyl_ny          = 180;    // strip subdivisions along y (rows = tcyl_ny+1)
    double      tcyl_radius      = 0.06;   // both cylinders' radius (m); axis runs along x
    double      tcyl_length      = 1.50;   // both cylinders' length along x (m); should exceed tcyl_strip_span_z
    int         tcyl_nu          = 64;     // circumferential subdivisions for visual cylinder mesh
    double      tcyl_visual_shrink = 0.005;// render cylinders this much thinner than r (m). With SDF active a small shrink (≈ d_hat) is enough.
    double      tcyl_twist_rate  = 0.15;   // turns/second per cylinder (sign-flipped between top/bot)
    double      tcyl_settle_time = 0.2;    // seconds with omega=0 so cloth settles under gravity
    double      tcyl_ramp_time   = 0.5;    // seconds to linearly ramp omega from 0 to full
    double      tcyl_max_turn    = 1.5;    // |total turns| cap per cylinder (0 = no cap). 0.5=180°, 1.5=540°
    bool        tcyl_untwist     = true;   // after reaching max_turn, smoothly reverse rotation back to 0
    double      tcyl_hold_time   = 0.0;    // seconds to dwell at peak twist before reversing (untwist only)

    // example 3: rectangular cloth wrapping one horizontal cylinder, twisted between two clamped top edges
    double      tu_size           = 4.0;   // cloth arc length (m): drop + bottom wrap + drop
    double      tu_width          = 0.60;  // cloth width along the cylinder axis (m)
    int         tu_nx             = 55;    // grid subdivisions along x (cyl axis)
    int         tu_ny             = 180;   // grid subdivisions along the arc
    double      tu_twist_rate     = 0.15;  // turns/second
    double      tu_settle_time    = 0.2;
    double      tu_ramp_time      = 0.5;
    double      tu_max_turn       = 2.0;   // |total turns| cap (0 = uncapped)
    bool        tu_untwist        = true;  // mirror the trapezoid back to 0
    double      tu_hold_time      = 0.0;   // dwell at peak twist before reversing
    double      tu_cyl_radius     = 0.156;
    double      tu_cyl_length     = 3.9;   // along x (m)
    int         tu_cyl_nu         = 64;    // circumferential subdivisions for the visual mesh
    double      tu_visual_shrink  = 0.002; // visual cyl this much thinner than pin_r (m); avoids SDF/barrier fighting at contact

    // example 12: three stacked cloth layers with one sinusoidally driven edge
    int         osc_nx          = 49;    // subdivisions along the driven direction
    int         osc_nz          = 49;    // subdivisions across each driven edge
    double      osc_length      = 1.50;  // fixed-to-driven edge distance (m)
    double      osc_width       = 0.50;  // sheet width (m)
    double      osc_layer_gap   = 0.0022;// vertical mid-surface gap (m)
    double      osc_amplitude   = 0.02;  // driven-edge sinusoid amplitude (m)
    double      osc_frequency   = 0.50;  // driven-edge frequency (Hz)

    // examples 14/15: vertically separated cloth sheets falling onto a long cylinder
    int         drop_stack_count = 50;
    int         drop_cloth_nx    = 69;   // 70 vertices along x
    int         drop_cloth_ny    = 69;   // 70 vertices along z
    double      drop_first_y     = 1.20; // lowest sheet's initial height (m)
    double      drop_spacing     = 0.005;// vertical sheet spacing (m)
    double      drop_cloth_w     = 2.07; // sheet width along x (m), 3 cm cells
    double      drop_cloth_h     = 2.07; // sheet length along z (m), 3 cm cells
    double      drop_cx          = 0.0;  // cloth-stack center x (m)
    double      drop_cz          = 0.0;  // cloth-stack center z (m)
    double      cyl_ground_size  = 12.5; // visual ground edge length (m)
    double      cyl_ground_cell_size = 0.25; // maximum visual ground cell edge (m)
    int         cyl_nu           = 64;   // circumferential visual subdivisions
    int         cyl_cap_rings    = 12;   // radial subdivisions of the visual end caps
    double      cyl_radius       = 0.35; // visible cylinder radius (m)
    double      cyl_sdf_padding  = 0.012;// extra collision radius for coarse cloth faces (m)
    double      cyl_length       = 4.0;  // visual length along z (m); covers the observed contact region
    double      cyl_cx           = -0.70;// offset left beneath the cloth stack
    double      cyl_cy           = 0.60;
    double      cyl_cz           = 0.0;

    // Example 16: interlocking rigid mesh links supporting a rigid-object drop.
    int         chain_net_rows = 17;
    int         chain_net_cols = 17;
    double      chain_net_link_scale = 0.9375; // 17x17 compact net retains the 24.75 x 24 m span
    bool        chain_net_mixed_objects = true;
    double      chain_net_object_size = 0.0; // 0: one eighth of the smaller net span, capped to fit
    double      chain_net_object_density = 1000.0;
    double      chain_net_ball_radius = 0.0; // 0: one quarter of the smaller net span
    double      chain_net_ball_gap = 3.0;    // surface clearance above the highest link
    double      chain_net_ball_density = 1000.0;
    int         chain_net_ball_subdiv = 2;

    // Example 18: prescribed soft-pin motor speed about world Z (degrees/second).
    double bike_chain_drive_speed = -25.0;
    double bike_chain_motor_radius = 0.03; // Radius of the four moving motor pins (m).
    double bike_chain_shaft_span = 0.05; // Distance from each gear center to its shaft support points (m).

    // --- output / restart ---
    std::string outdir       = "frames_sim3d";
    std::string format       = "geo";
    int         restart_frame = -1;

    // --- data ---
    std::string datadir      = "";

    IPCArgs3D() {
        add_double("fps",         fps,         30.0,       "Output frames per second");
        add_int   ("substeps",    substeps,    3,          "Solver substeps per frame (solver_dt = 1/(fps*substeps))");
        add_int   ("num_frames",  num_frames,  60,        "Number of frames to simulate");

        add_double("E",           E,           1e6,        "Shell Young's modulus (Pa); membrane Lamé parameters use E * thickness");
        add_double("nu",          nu,          0.3,        "Shell Poisson ratio");
        add_double("density",     density,     900.0,       "Shell mass density (kg/m^3)");
        add_double("thickness",   thickness,   0.001,       "Shell thickness (m)");
        add_double("solid_E",     solid_E,     5e4,         "Volumetric-solid Young's modulus (Pa)");
        add_double("solid_nu",    solid_nu,    0.45,        "Volumetric-solid Poisson ratio");
        add_double("solid_density", solid_density, 900.0,   "Volumetric-solid mass density (kg/m^3)");
        add_double("rigid_density", rigid_density, 900.0,   "Rigid-body mass density (kg/m^3)");
        add_double("kB",          kB,          1e-3,       "Bending stiffness (0 disables bending)");
        add_double("kpin",        kpin,        1e5,        "Pin spring stiffness");
        add_double("gx",          gx,          0.0,        "Gravity x-component");
        add_double("gy",          gy,          -9.81,      "Gravity y-component");
        add_double("gz",          gz,          0.0,        "Gravity z-component");

        add_int   ("max_substep_iters", max_substep_iters, 500, "Max solver iterations per substep (per-frame total = max_substep_iters * substeps)");
        add_double("tol_abs",     tol_abs,     1e-6,       "Absolute convergence tolerance (residual force)");
        add_double("tol_rel",     tol_rel,     1e-3,       "Relative tolerance: stop when residual < tol_rel * initial_residual (0 disables)");
        add_double("d_hat",       d_hat,       0.005,      "Barrier activation distance (0 = off)");
        add_double("friction_coefficient", friction_coefficient, 0.0, "Dynamic Coulomb coefficient for mesh node-triangle/edge-edge and analytic-SDF contact (0 = off)");
        add_double("friction_velocity_epsilon", friction_velocity_epsilon, 0.01, "Velocity threshold (m/s) that smooths contact friction near zero slip");
        add_double("k_sdf",       k_sdf,       1e5,        "SDF penalty stiffness (0 = off)");
        add_double("eps_sdf",     eps_sdf,     0.002,      "SDF soft-barrier range (m). Cloth's force-free rest is at phi=eps_sdf. 0 = hard quadratic at the surface.");
        add_bool  ("use_parallel",   use_parallel,   true,  "Use parallel solver updates (Gauss-Seidel requires coloring)");
        add_bool  ("use_basic_experimental", use_basic_experimental, false, "Use experimental cloth solving; with use_simd also selects experimental general solving");
        add_bool  ("use_simd", use_simd, false, "With use_basic_experimental, select SIMD v2 for cloth or the general cloth/solid/rigid solver");
        add_bool  ("use_jacobi_dat", use_jacobi_dat, false, "Deformable cloth/solid only: Jacobi updates filtered by planar DAT; overrides experimental/SIMD solvers and starts from previous positions, skipping initial guesses; incompatible with OGC/grid");
        add_double("dat_gamma", dat_gamma, 0.9, "DAT relaxation factor, strictly between 0 and 1");
        add_double("dat_query_radius", dat_query_radius, 0.0, "DAT query radius; 0 selects 1.5*d_hat when d_hat > 0, otherwise node_box_max");
        add_double("dat_node_box_padding", dat_node_box_padding, 1.2, "DAT query-box padding multiplier, at least 1");
        add_bool  ("use_cloth_grid", use_cloth_grid, false, "Basic cloth only: parallel parity-colored grid cells, serial vertices inside each cell");
        add_bool  ("cloth_grid_auto_dx", cloth_grid_auto_dx, false, "Grid only: enlarge dx from dependency spans and node boxes at every rebuild for one batch per occupied parity color");
        add_double("cloth_grid_dx", cloth_grid_dx, 0.05, "Cloth grid side length, or minimum side with cloth_grid_auto_dx. Fixed mode requires > 2 * node_box_max and can split colors into batches");
        add_bool  ("write_substeps", write_substeps, false, "Write an output file after every substep (useful for visual debugging)");

        add_bool  ("use_ccd",      use_ccd,      true,   "Run CCD step clamping in per_vertex_safe_step");
        add_bool  ("use_ccd_guess",    use_ccd_guess,    true,  "Use ccd_initial_guess as the substep start point (ignored with OGC)");
        add_bool  ("use_colored_ccd_guess", use_colored_ccd_guess, false, "Cloth/solid: original collision-colored linear CCD initial guess toward xhat; mutually exclusive with use_colored_ccd_guess_level_based; rigid bodies stay fixed; overrides CCD, Verlet, and translation guess flags, but is ignored with OGC");
        add_bool  ("use_colored_ccd_guess_level_based", use_colored_ccd_guess_level_based, false, "Cloth/solid: level-based colored CCD toward xhat, with one discarded trial sweep to build levels; mutually exclusive with use_colored_ccd_guess; rigid bodies stay fixed; overrides CCD, Verlet, and translation guess flags, but is ignored with OGC");
        add_int   ("colored_ccd_guess_iters", colored_ccd_guess_iters, 10, "Maximum actual updating sweeps for either colored CCD initial guess; excludes the level-based trial sweep (0 skips trial and updates)");
        add_double("colored_ccd_guess_fraction", colored_ccd_guess_fraction, 1.0, "Fixed target fraction of xhat minus current positions for either colored CCD initial guess, finite in [0, 1]; applied once before sweeps and the level trial; does not change the solver's xhat");
        add_bool  ("use_verlet_guess", use_verlet_guess, false, "Start GS from xhat + dt^2*gravity (Verlet predictor)");
        add_bool  ("use_translation_guess", use_translation_guess, false, "Start GS from x^n + C");
        add_bool  ("use_ogc", use_ogc, false, "Use trust-region narrow phase instead of CCD for step clamping");
        add_bool  ("use_ogc_solver", use_ogc_solver, false, "Use the serial OGC Gauss-Seidel solver (rebuilds BVH per iter; partial leaf refit per move)");
        add_double("ogc_box_pad", ogc_box_pad, 0.005, "Padding on OGC node boxes / tri-edge unions for the per-iter BVH rebuild (floored to d_hat at use)");
        add_bool  ("fixed_iters",      fixed_iters,      false, "Run exactly max_substep_iters sweeps per substep with no tolerance / convergence check");
        add_bool  ("verbose",          verbose,          false, "Print solver cache rebuilds and residuals after each GS iteration (residuals require fixed_iters=false)");
        add_double("node_box_max",         node_box_max,         0.01,  "Upper bound on node box half-extent used by the basic solver");
        add_double("node_box_min",         node_box_min,         0.001, "Lower bound on node box half-extent (floor when prev disp is near zero)");
        add_double("theta_box_min",        theta_box_min,        0.01,  "Lower bound on the rigid orientation-box angular radius (rad)");
        add_double("theta_box_max",        theta_box_max,        0.1,   "Upper bound on the rigid orientation-box angular radius (rad)");
        add_int   ("node_box_update_count", node_box_update_count, 10,  "GS iterations between node-box/contact-color rebuilds; set to max_substep_iters to rebuild once per substep");
        add_double("k_barrier",                k_barrier,                100.0, "Barrier stiffness multiplier");
        add_double("damping",                  damping,                  1.0,   "Newton-step damping used by deformable and rigid solvers; <1 can stabilize an update");
        add_bool  ("use_ticcd",                use_ticcd,                false, "CCD backend for *_only_one_node_moves: true=Tight-Inclusion library, false=self-written linear (default)");
        add_bool  ("exact_computation_fallback", exact_computation_fallback, true, "Enable additional exact-arithmetic CCD/contact fallbacks, matching barrier/friction calculations, and colored-predictor path checks (default: true)");
        add_bool  ("ccd_exact_step_bound", ccd_exact_step_bound, false, "Experimental direct exact separating-plane gap bound before legacy backtracking; requires exact fallback with linear CCD and OGC disabled (default: false)");

        add_int   ("example",      example,       1,              "Scene to run: 1=twisting_cloth, 2=two_cylinder_twist, 3=cylinder_twist_untwist, 4=avatar_clothing, 5=rotating_tennis_racket, 6=rotating_space_tool, 7=twenty_polygon_stack, 8=two_bunny_spot_cube_gear_cycles_on_pinned_cloth, 9=dynamic_bolt_into_fixed_nut, 10=four_bunny_spot_cube_gear_rows_on_pinned_cloth, 11=cloth_unrolling_down_fixed_ramp, 12=oscillating_cloth_layers, 13=wrecking_ball, 14=cloth_cylinder_drop, 15=200_sheet_cylinder_drop, 16=chain_net_ball, 17=rigid_objects_in_box, 18=bike_chain");
        add_double("sheet_y",      sheet_y,       0.20,           "Midline y (m) for example 1");
        add_double("twist_rate",   twist_rate,    0.5,            "Relative twist rate in Hz for example 1 (turns/second; total turns = rate * duration)");
        add_int   ("twist_nx",     twist_nx,      99,             "Grid subdivisions along x for example 1 (vertices = (twist_nx+1)*(twist_ny+1))");
        add_int   ("twist_ny",     twist_ny,      99,             "Grid subdivisions along y for example 1");
        add_double("twist_size",   twist_size,    2.5,            "Edge length (m) of the square cloth in example 1");

        add_int   ("tcyl_n_strips",    tcyl_n_strips,    4,    "Number of cloth strips along x (along cylinder axis) for example 2");
        add_double("tcyl_strip_w",     tcyl_strip_w,     0.18, "Each strip's x-width (m, along beam axis) for example 2");
        add_double("tcyl_strip_span_z",tcyl_strip_span_z,1.20, "Total x-span over which strips are distributed (m, along cyl axis) for example 2");
        add_double("tcyl_cloth_h",     tcyl_cloth_h,     1.00, "Pin-to-pin y distance between cylinders (m) for example 2");
        add_int   ("tcyl_nx",          tcyl_nx,          16,   "Strip subdivisions along x for example 2 (cols = tcyl_nx+1)");
        add_int   ("tcyl_ny",          tcyl_ny,          180,  "Strip subdivisions along y for example 2 (rows = tcyl_ny+1)");
        add_double("tcyl_radius",      tcyl_radius,      0.06, "Cylinder radius (m) for example 2 (axis along x)");
        add_double("tcyl_length",      tcyl_length,      1.50, "Cylinder length along x (m) for example 2; should exceed tcyl_strip_span_z");
        add_int   ("tcyl_nu",          tcyl_nu,          64,   "Circumferential subdivisions for visual cylinder mesh in example 2");
        add_double("tcyl_visual_shrink", tcyl_visual_shrink, 0.005, "Render cylinders thinner than tcyl_radius by this many m (visual only; physics still uses tcyl_radius). With SDF active the cloth can't penetrate r, so a small shrink (≈ d_hat) hides only transient lag. Pass a larger value (e.g. 0.04) to recover the old skinny-cylinder look.");
        add_double("tcyl_twist_rate",  tcyl_twist_rate,  0.15, "Turns/second per cylinder in example 2 (top and bot rotate in opposite directions)");
        add_double("tcyl_settle_time", tcyl_settle_time, 0.2,  "Seconds with omega=0 so cloth settles under gravity in example 2");
        add_double("tcyl_ramp_time",   tcyl_ramp_time,   0.5,  "Seconds to linearly ramp omega between 0 and full in example 2");
        add_double("tcyl_max_turn",    tcyl_max_turn,    1.5,  "Per-cylinder rotation cap (turns) in example 2. 0 = no cap; 0.5=180°, 1.5=540°.");
        add_bool  ("tcyl_untwist",     tcyl_untwist,     true, "If true, after reaching tcyl_max_turn the cylinder smoothly reverses back to 0 (twist + untwist).");
        add_double("tcyl_hold_time",   tcyl_hold_time,   0.0,  "Seconds to dwell at peak twist before reversing (only with tcyl_untwist=true).");

        add_double("tu_size",          tu_size,          4.0,  "Cloth arc length (m) for example 3: drop + bottom wrap + drop");
        add_double("tu_width",         tu_width,         0.60, "Cloth width (m) along the cylinder axis for example 3");
        add_int   ("tu_nx",            tu_nx,            55,   "Cloth grid subdivisions along x for example 3 (along the cylinder axis)");
        add_int   ("tu_ny",            tu_ny,            180,  "Cloth grid subdivisions along the arc (back drop -> bottom wrap -> front drop) for example 3");
        add_double("tu_twist_rate",    tu_twist_rate,    0.15, "Cylinder yaw rate (turns/second) for example 3");
        add_double("tu_settle_time",   tu_settle_time,   0.2,  "Seconds with omega=0 so the cloth settles before the cylinder starts yawing in example 3");
        add_double("tu_ramp_time",     tu_ramp_time,     0.5,  "Seconds to linearly ramp omega between 0 and full in example 3");
        add_double("tu_max_turn",      tu_max_turn,      2.0,  "Cylinder yaw cap (turns) in example 3 (0 = no cap)");
        add_bool  ("tu_untwist",       tu_untwist,       true, "If true, after reaching tu_max_turn the cylinder reverses back to 0 (twist + untwist) in example 3");
        add_double("tu_hold_time",     tu_hold_time,     0.0,  "Seconds to dwell at peak twist before reversing in example 3");
        add_double("tu_cyl_radius",    tu_cyl_radius,    0.156,"Collider cylinder radius (m) for example 3");
        add_double("tu_cyl_length",    tu_cyl_length,    3.9,  "Collider cylinder length along x (m) for example 3");
        add_int   ("tu_cyl_nu",        tu_cyl_nu,        64,   "Circumferential subdivisions for the visual cylinder in example 3");
        add_double("tu_visual_shrink", tu_visual_shrink, 0.002,"Render cylinder this much thinner than the cloth's rest radius (m, visual only). Small offset (~2mm) keeps a visible sliver between cloth + cylinder and avoids SDF/barrier energy fighting at the contact");

        add_int   ("osc_nx",        osc_nx,        49,     "Grid subdivisions from fixed to driven edge in example 12");
        add_int   ("osc_nz",        osc_nz,        49,     "Grid subdivisions across each edge in example 12");
        add_double("osc_length",    osc_length,    1.50,   "Fixed-to-driven edge distance (m) in example 12");
        add_double("osc_width",     osc_width,     0.50,   "Width (m) of each cloth layer in example 12");
        add_double("osc_layer_gap", osc_layer_gap, 0.0022, "Vertical mid-surface gap (m) between layers in example 12");
        add_double("osc_amplitude", osc_amplitude, 0.02,   "Signed vertical sinusoid amplitude (m) of the driven edges in example 12");
        add_double("osc_frequency", osc_frequency, 0.50,   "Sinusoid frequency (Hz) of the driven edges in example 12");

        add_int   ("drop_stack_count", drop_stack_count, 50,   "Number of falling cloth sheets in examples 14 and 15");
        add_int   ("drop_cloth_nx",    drop_cloth_nx,    69,   "Grid subdivisions along x per sheet in examples 14 and 15 (vertices = subdivisions + 1)");
        add_int   ("drop_cloth_ny",    drop_cloth_ny,    69,   "Grid subdivisions along z per sheet in examples 14 and 15 (vertices = subdivisions + 1)");
        add_double("drop_first_y",     drop_first_y,     1.20, "Initial lowest-sheet height (m) in examples 14 and 15; must clear the padded cylinder top and eps_sdf");
        add_double("drop_spacing",     drop_spacing,     0.005,"Initial vertical spacing (m) between cloth sheets in examples 14 and 15");
        add_double("drop_cloth_w",     drop_cloth_w,     2.07, "Cloth width along x (m) in examples 14 and 15");
        add_double("drop_cloth_h",     drop_cloth_h,     2.07, "Cloth length along z (m) in examples 14 and 15");
        add_double("drop_cx",          drop_cx,          0.0,  "Cloth-stack center x (m) in examples 14 and 15, independent of the cylinder");
        add_double("drop_cz",          drop_cz,          0.0,  "Cloth-stack center z (m) in examples 14 and 15, independent of the cylinder");
        add_double("cyl_ground_size",  cyl_ground_size,  12.5, "Visual ground edge length (m) in examples 14 and 15");
        add_double("cyl_ground_cell_size", cyl_ground_cell_size, 0.25, "Maximum visual ground cell edge (m) in examples 14 and 15");
        add_int   ("cyl_nu",           cyl_nu,           64,   "Circumferential visual cylinder subdivisions in examples 14 and 15");
        add_int   ("cyl_cap_rings",    cyl_cap_rings,    12,   "Radial subdivisions per visual cylinder end cap in examples 14 and 15");
        add_double("cyl_radius",       cyl_radius,       0.35, "Visible cylinder radius (m) in examples 14 and 15");
        add_double("cyl_sdf_padding",  cyl_sdf_padding,  0.012,"Extra cylinder SDF radius (m) in examples 14 and 15 to keep coarse cloth faces outside the visible cylinder");
        add_double("cyl_length",       cyl_length,       4.0,  "Visual cylinder length along z (m) in examples 14 and 15; collision SDF is infinite");
        add_double("cyl_cx",           cyl_cx,          -0.70, "Cylinder center x (m) in examples 14 and 15; negative offsets it left of the cloth stack");
        add_double("cyl_cy",           cyl_cy,           0.60, "Cylinder center y (m) in examples 14 and 15");
        add_double("cyl_cz",           cyl_cz,           0.0,  "Cylinder center z (m) in examples 14 and 15");

        add_int("chain_net_rows", chain_net_rows, 17, "Example 16: compact net base-link rows (at least 3; total scene limited to 30,000 vertices)");
        add_int("chain_net_cols", chain_net_cols, 17, "Example 16: compact net base-link columns (at least 3; total scene limited to 30,000 vertices)");
        add_double("chain_net_link_scale", chain_net_link_scale, 0.9375, "Example 16: positive uniform scale of link geometry and center spacing; the default 17x17 grid retains the 24.75 x 24 m net span");
        add_bool("chain_net_mixed_objects", chain_net_mixed_objects, true, "Example 16: drop 12 gears, 12 cubes, 12 balls and 12 triangular/pentagonal/octagonal rigid prisms; false restores the single ball");
        add_double("chain_net_object_size", chain_net_object_size, 0.0, "Example 16 mixed drop: maximum object extent; 0 uses one eighth of the smaller net span, capped to fit the fixed net interior");
        add_double("chain_net_object_density", chain_net_object_density, 1000.0, "Example 16 mixed drop: volume density of all falling objects; rigid_density controls links");
        add_double("chain_net_ball_radius", chain_net_ball_radius, 0.0, "Example 16 single-ball mode: sphere radius; 0 uses one quarter of the smaller net span");
        add_double("chain_net_ball_gap", chain_net_ball_gap, 3.0, "Example 16: positive initial surface gap from highest net vertex to lowest falling object");
        add_double("chain_net_ball_density", chain_net_ball_density, 1000.0, "Example 16 single-ball mode: sphere volume density; rigid_density controls links");
        add_int("chain_net_ball_subdiv", chain_net_ball_subdiv, 2, "Example 16: subdivision level for each falling sphere (0-7; subject to the 30,000-vertex scene limit)");

        add_double("bike_chain_drive_speed", bike_chain_drive_speed, -25.0, "Example 18: soft-pin motor target speed about world z in degrees/second");
        add_double("bike_chain_motor_radius", bike_chain_motor_radius, 0.03, "Example 18: positive radius of the four moving soft-pin motor points in metres");
        add_double("bike_chain_shaft_span", bike_chain_shaft_span, 0.05, "Example 18: positive axial distance from each gear center to either stationary shaft support point in metres");

        add_string("outdir",       outdir,        "frames_sim3d", "Output directory");
        add_string("format",       format,        "geo",          "Output format: obj, geo, ply, or usd");
        add_int   ("restart_frame", restart_frame, -1,            "Frame to restart from (-1 = no restart)");
        add_string("datadir",      datadir,       "",             "Path to input data directory (e.g. meshes for example 4)");
    }

    bool parse(int argc, char** argv) {
        if (!ArgParser::parse(argc, argv)) return false;
        const auto supplied = [&](const char* flag) {
            for (int i = 1; i < argc; ++i)
                if (std::string(argv[i]) == flag) return true;
            return false;
        };
        if (example == 12) {
            if (!supplied("--osc_amplitude")) osc_amplitude = 0.01;
            if (!supplied("--osc_frequency")) osc_frequency = 6.0;
            if (!supplied("--osc_length")) osc_length = 1.0;
        }
        if (example == 15) {
            const auto preset = [&](const char* flag, auto& field, auto value) {
                if (!supplied(flag)) field = value;
            };
            // 140 cells give 141 vertices and 3 cm edges on each 4.2 m sheet.
            // Defaults use the stronger contact budget; the README begins
            // with an explicit 40x3 free-fall phase before restarting here.
            preset("--drop_stack_count", drop_stack_count, 200);
            preset("--drop_cloth_nx", drop_cloth_nx, 140);
            preset("--drop_cloth_ny", drop_cloth_ny, 140);
            preset("--drop_cloth_w", drop_cloth_w, 4.2);
            preset("--drop_cloth_h", drop_cloth_h, 4.2);
            preset("--drop_first_y", drop_first_y, 1.8);
            preset("--drop_cx", drop_cx, 0.8);
            preset("--cyl_radius", cyl_radius, 0.55);
            preset("--cyl_cy", cyl_cy, 0.65);
            preset("--cyl_length", cyl_length, 8.0);
            preset("--cyl_sdf_padding", cyl_sdf_padding, 0.006);
            preset("--cyl_ground_size", cyl_ground_size, 40.0);
            preset("--nu", nu, 0.25);
            preset("--substeps", substeps, 100);
            preset("--max_substep_iters", max_substep_iters, 4);
            preset("--node_box_update_count", node_box_update_count, max_substep_iters);
            preset("--fixed_iters", fixed_iters, true);
            preset("--use_basic_experimental", use_basic_experimental, true);
            preset("--use_simd", use_simd, true);
            preset("--eps_sdf", eps_sdf, 0.015);
            preset("--k_barrier", k_barrier, 10000.0);
            preset("--node_box_min", node_box_min, 0.0001);
            preset("--node_box_max", node_box_max, 0.003);
            preset("--num_frames", num_frames, 120);
        }
        if (example == 14 || example == 15) {
            // The general 5 mm contact band equals this scene's release gap.
            // Keep the default strictly smaller, while honoring explicit flags.
            if (!supplied("--d_hat")) d_hat = 0.0048;
        }
        if (example == 16) {
            const auto preset = [&](const char* flag, auto& field, auto value) {
                if (!supplied(flag)) field = value;
            };
            // Starting settings for the mesh-link adaptation, not the paper's
            // codimensional model or its adaptive Newton solver settings.
            preset("--fps", fps, 25.0);
            preset("--substeps", substeps, 4);
            preset("--num_frames", num_frames, 250);
            preset("--rigid_density", rigid_density, 1000.0);
            preset("--d_hat", d_hat, 0.005);
            preset("--k_barrier", k_barrier, 1.0e9);
            preset("--max_substep_iters", max_substep_iters, 100);
            preset("--fixed_iters", fixed_iters, true);
            preset("--use_basic_experimental", use_basic_experimental, true);
            preset("--use_simd", use_simd, true);
        }
        if (example == 17) {
            const auto preset = [&](const char* flag, auto& field, auto value) {
                if (!supplied(flag)) field = value;
            };
            // Starting budget for the 300-object rigid drop; not a measured
            // convergence guarantee. Explicit command-line values win.
            preset("--fps", fps, 30.0);
            preset("--num_frames", num_frames, 200);
            preset("--substeps", substeps, 20);
            preset("--max_substep_iters", max_substep_iters, 50);
            preset("--fixed_iters", fixed_iters, true);
            preset("--rigid_density", rigid_density, 1000.0);
            preset("--d_hat", d_hat, 0.001);
            preset("--k_barrier", k_barrier, 1.0e9);
            preset("--friction_coefficient", friction_coefficient, 0.1);
            preset("--use_parallel", use_parallel, true);
            preset("--use_basic_experimental", use_basic_experimental, true);
            preset("--use_simd", use_simd, true);
            preset("--exact_computation_fallback", exact_computation_fallback, true);
        }
        if (example == 18) {
            const auto preset = [&](const char* flag, auto& field, auto value) {
                if (!supplied(flag)) field = value;
            };
            preset("--num_frames", num_frames, 180);
            preset("--fps", fps, 30.0);
            preset("--gy", gy, 0.0);
            preset("--substeps", substeps, 19);
            preset("--max_substep_iters", max_substep_iters, 200);
            preset("--fixed_iters", fixed_iters, true);
            preset("--use_basic_experimental", use_basic_experimental, true);
            preset("--use_simd", use_simd, true);
            preset("--use_parallel", use_parallel, true);
            preset("--use_ccd", use_ccd, true);
            preset("--exact_computation_fallback", exact_computation_fallback, true);
            preset("--rigid_density", rigid_density, 1000.0);
            preset("--bike_chain_drive_speed", bike_chain_drive_speed, -25.0);
            preset("--bike_chain_motor_radius", bike_chain_motor_radius, 0.03);
            preset("--bike_chain_shaft_span", bike_chain_shaft_span, 0.05);
            preset("--d_hat", d_hat, 3.0e-5);
            preset("--k_barrier", k_barrier, 4000.0);
            preset("--kpin", kpin, 1.0e4);
            preset("--k_sdf", k_sdf, 0.0);
            preset("--friction_coefficient", friction_coefficient, 0.0);
            preset("--node_box_min", node_box_min, 2.0e-5);
            preset("--node_box_max", node_box_max, 1.0e-4);
            preset("--theta_box_min", theta_box_min, 4.0e-4);
            preset("--theta_box_max", theta_box_max, 8.0e-3);
            preset("--node_box_update_count", node_box_update_count, 25);
        }
        return true;
    }

    ExportFormat to_export_format() const {
        if (format == "geo") return ExportFormat::GEO;
        if (format == "ply") return ExportFormat::PLY;
        if (format == "usd") return ExportFormat::USD;
        return ExportFormat::OBJ;
    }

    SimParams to_sim_params() const {
        if (!std::isfinite(friction_coefficient)
            || friction_coefficient < 0.0) {
            throw std::invalid_argument(
                "friction_coefficient must be nonnegative and finite");
        }
        if (friction_coefficient > 0.0
            && (!std::isfinite(friction_velocity_epsilon)
                || friction_velocity_epsilon <= 0.0)) {
            throw std::invalid_argument(
                "friction_velocity_epsilon must be positive and finite");
        }
        SimParams p = SimParams::zeros();
        p.fps              = fps;
        p.substeps         = substeps;
        p.mu               = E * thickness / (2.0 * (1.0 + nu));
        p.lambda           = E * thickness * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        p.density          = density;
        p.thickness        = thickness;
        p.solid_mu         = solid_E / (2.0 * (1.0 + solid_nu));
        p.solid_lambda     = solid_E * solid_nu
            / ((1.0 + solid_nu) * (1.0 - 2.0 * solid_nu));
        p.solid_density    = solid_density;
        p.rigid_density    = rigid_density;
        p.kB               = kB;
        p.kpin             = kpin;
        p.gravity          = Vec3(gx, gy, gz);
        p.max_global_iters = max_substep_iters;
        p.tol_abs          = tol_abs;
        p.tol_rel          = tol_rel;
        p.d_hat            = d_hat;
        p.friction_coefficient = friction_coefficient;
        p.friction_velocity_epsilon = friction_velocity_epsilon;
        p.k_sdf            = k_sdf;
        p.eps_sdf          = eps_sdf;
        p.use_parallel     = use_parallel;
        p.use_basic_experimental = use_basic_experimental;
        p.use_basic_experimental_v2 = use_basic_experimental && use_simd;
        p.use_simd         = use_simd;
        p.use_jacobi_dat   = use_jacobi_dat;
        p.dat_gamma        = dat_gamma;
        p.dat_query_radius = dat_query_radius;
        p.dat_node_box_padding = dat_node_box_padding;
        p.use_cloth_grid   = use_cloth_grid;
        p.cloth_grid_auto_dx = cloth_grid_auto_dx;
        p.cloth_grid_dx    = cloth_grid_dx;
        p.write_substeps   = write_substeps;
        p.use_ccd          = use_ccd;
        p.use_ccd_guess    = use_ccd_guess;
        p.use_colored_ccd_guess = use_colored_ccd_guess;
        p.use_colored_ccd_guess_level_based = use_colored_ccd_guess_level_based;
        p.colored_ccd_guess_iters = colored_ccd_guess_iters;
        p.colored_ccd_guess_fraction = colored_ccd_guess_fraction;
        p.use_verlet_guess = use_verlet_guess;
        p.use_ogc = use_ogc;
        p.use_ogc_solver = use_ogc_solver;
        p.ogc_box_pad = ogc_box_pad;
        p.use_translation_guess = use_translation_guess;
        p.fixed_iters = fixed_iters;
        p.verbose = verbose;
        p.node_box_max            = node_box_max;
        p.node_box_min            = node_box_min;
        p.theta_box_min           = theta_box_min;
        p.theta_box_max           = theta_box_max;
        p.node_box_update_count   = node_box_update_count;
        p.k_barrier                   = k_barrier;
        p.damping                     = damping;
        p.use_ticcd = use_ticcd;
        p.exact_computation_fallback = exact_computation_fallback;
        p.exact_computation_fallback_2 = exact_computation_fallback;
        p.ccd_exact_step_bound = ccd_exact_step_bound;
        p.validate_jacobi_dat_parameters();
        p.validate_ccd_exact_step_bound_parameters();
        p.validate_colored_ccd_guess_parameters();
        p.validate_cloth_grid_parameters();
        return p;
    }
};
