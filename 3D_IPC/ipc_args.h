#pragma once

#include "args.h"
#include "physics.h"
#include "visualization.h"

// CLI-facing view of SimParams (with scene-placement knobs on top).
struct IPCArgs3D : ArgParser {

    // --- time integration ---
    double fps          = 30.0;
    int    substeps     = 3;
    int    num_frames   = 60;

    // --- physics ---
    double E            = 1000.0; // Pa  (Young's modulus)
    double nu           = 0.3;    // (Poisson ratio)
    double density      = 900.0;  // kg/m^3
    double thickness    = 0.001;  // m
    double kB           = 1e-3;   // J   (bending stiffness; 0 disables bending)
    double kpin         = 1e5;    // N/m (pin spring stiffness)
    double gx           = 0.0;    // m/s^2
    double gy           = -9.81;  // m/s^2
    double gz           = 0.0;    // m/s^2

    // --- solver ---
    int    max_substep_iters = 500;  // per-substep cap; per-frame total = max_substep_iters * substeps
    double tol_abs       = 1e-6;  // residual norm tolerance
    double tol_rel       = 1e-1;  // relative tolerance: stop when residual < tol_rel * initial
    double d_hat         = 0.01;  // barrier activation distance; 0 disables contact
    double k_sdf         = 1e5;   // SDF penalty stiffness; 0 disables the SDF term
    double eps_sdf       = 0.002; // SDF soft-barrier range (m); cloth rest at phi=eps_sdf. 0 = hard quadratic at the surface.
    bool   use_parallel  = true;
    bool   write_substeps = false;
    bool   use_ccd       = true;
    bool   use_ccd_guess = true;
    bool   use_ogc = false;
    bool   use_ogc_solver = false;
    double ogc_box_pad = 0.005;
    bool   fixed_iters = false;
    bool   use_gpu                 = false;
    double node_box_max            = 0.01;
    double node_box_min            = 0.001;
    double k_barrier                   = 1.0;
    bool   use_ticcd                   = false;    // true: Tight-Inclusion library | false: self-written linear CCD

    // --- scene selection ---
    int         example      = 1;   // 1=twisting_cloth, 2=two_cylinder_twist, 3=cloth_pile
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

    // example 3: small cloths piling onto a corner-pinned ground sheet
    int         pile_count       = 5;      // number of falling cloths
    int         pile_nx          = 16;     // each falling cloth's grid subdivisions along x
    int         pile_ny          = 16;     // each falling cloth's grid subdivisions along y
    double      pile_cloth_size  = 0.35;   // each falling cloth's edge length (m)
    double      pile_first_y     = 0.25;   // y of the lowest falling cloth at t=0 (m)
    double      pile_spacing     = 0.09;   // y-gap between successive falling cloths (m)
    double      pile_drop_speed  = 2.0;    // initial -y velocity for every falling vertex (m/s)
    int         pile_ground_nx   = 10;     // ground sheet grid subdivisions along x
    int         pile_ground_ny   = 10;     // ground sheet grid subdivisions along y
    double      pile_ground_size = 1.2;    // ground sheet edge length (m); pinned at all four corners

    // --- output / restart ---
    std::string outdir       = "frames_sim3d";
    std::string format       = "geo";
    int         restart_frame = -1;

    IPCArgs3D() {
        add_double("fps",         fps,         30.0,       "Output frames per second");
        add_int   ("substeps",    substeps,    3,          "Solver substeps per frame (solver_dt = 1/(fps*substeps))");
        add_int   ("num_frames",  num_frames,  60,        "Number of frames to simulate");

        add_double("E",           E,           1000.0,     "Young's modulus (Pa)");
        add_double("nu",          nu,          0.3,        "Poisson ratio");
        add_double("density",     density,     900.0,       "Mass density (kg/m^3)");
        add_double("thickness",   thickness,   0.001,       "Shell thickness (m)");
        add_double("kB",          kB,          1e-3,       "Bending stiffness (0 disables bending)");
        add_double("kpin",        kpin,        1e5,        "Pin spring stiffness");
        add_double("gx",          gx,          0.0,        "Gravity x-component");
        add_double("gy",          gy,          -9.81,      "Gravity y-component");
        add_double("gz",          gz,          0.0,        "Gravity z-component");

        add_int   ("max_substep_iters", max_substep_iters, 500, "Max Gauss-Seidel iterations per substep (per-frame total = max_substep_iters * substeps)");
        add_double("tol_abs",     tol_abs,     1e-6,       "Absolute convergence tolerance (residual force)");
        add_double("tol_rel",     tol_rel,     1e-1,       "Relative tolerance: stop when residual < tol_rel * initial_residual (0 disables)");
        add_double("d_hat",       d_hat,       0.01,       "Barrier activation distance (0 = off)");
        add_double("k_sdf",       k_sdf,       1e5,        "SDF penalty stiffness (0 = off). Penalty 0.5·k·(eps_sdf-phi)^2 for phi<eps_sdf; transient penetration depth ~ v·sqrt(m/k).");
        add_double("eps_sdf",     eps_sdf,     0.002,      "SDF soft-barrier range (m). Cloth's force-free rest is at phi=eps_sdf. 0 = hard quadratic at the surface.");
        add_bool  ("use_parallel",   use_parallel,   true,  "Use parallel Gauss-Seidel (requires coloring)");
        add_bool  ("write_substeps", write_substeps, false, "Write an output file after every substep (useful for visual debugging)");

        add_bool  ("use_ccd",      use_ccd,      true,   "Run CCD step clamping in per_vertex_safe_step");
        add_bool  ("use_ccd_guess",    use_ccd_guess,    true,  "Use ccd_initial_guess as the substep start point (ignored if use_ogc is on)");
        add_bool  ("use_ogc", use_ogc, false, "Use trust-region narrow phase instead of CCD for step clamping");
        add_bool  ("use_ogc_solver", use_ogc_solver, false, "Use the serial OGC Gauss-Seidel solver (rebuilds BVH per iter; partial leaf refit per move)");
        add_double("ogc_box_pad", ogc_box_pad, 0.005, "Padding on OGC node boxes / tri-edge unions for the per-iter BVH rebuild (floored to d_hat at use)");
        add_bool  ("fixed_iters",      fixed_iters,      false, "Run exactly max_substep_iters sweeps per substep with no tolerance / convergence check");
        add_bool  ("use_gpu",              use_gpu,              false, "Route the GS sweep through the GPU implementation (CPU stub when CUDA is unavailable)");
        add_double("node_box_max",         node_box_max,         0.01,  "Upper bound on node box half-extent used by the basic solver");
        add_double("node_box_min",         node_box_min,         0.001, "Lower bound on node box half-extent (floor when prev disp is near zero)");
        add_double("k_barrier",                k_barrier,                1.0,   "Barrier stiffness multiplier");
        add_bool  ("use_ticcd",                use_ticcd,                false, "CCD backend for *_only_one_node_moves: true=Tight-Inclusion library, false=self-written linear (default)");

        add_int   ("example",      example,       1,              "Scene to run: 1=twisting_cloth, 2=two_cylinder_twist, 3=cloth_pile");
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

        add_int   ("pile_count",       pile_count,       5,    "Number of falling cloths in example 3");
        add_int   ("pile_nx",          pile_nx,          16,   "Grid subdivisions along x for each falling cloth in example 3");
        add_int   ("pile_ny",          pile_ny,          16,   "Grid subdivisions along y for each falling cloth in example 3");
        add_double("pile_cloth_size",  pile_cloth_size,  0.35, "Edge length (m) of each falling cloth in example 3");
        add_double("pile_first_y",     pile_first_y,     0.25, "y-position (m) of the lowest falling cloth at t=0 in example 3");
        add_double("pile_spacing",     pile_spacing,     0.09, "y-gap (m) between successive falling cloths in example 3");
        add_double("pile_drop_speed",  pile_drop_speed,  2.0,  "Initial downward speed (m/s) of every falling-cloth vertex in example 3");
        add_int   ("pile_ground_nx",   pile_ground_nx,   10,   "Ground-sheet subdivisions along x in example 3");
        add_int   ("pile_ground_ny",   pile_ground_ny,   10,   "Ground-sheet subdivisions along y in example 3");
        add_double("pile_ground_size", pile_ground_size, 1.2,  "Ground-sheet edge length (m) in example 3 (corners pinned)");

        add_string("outdir",       outdir,        "frames_sim3d", "Output directory");
        add_string("format",       format,        "geo",          "Output format: obj, geo, ply, or usd");
        add_int   ("restart_frame", restart_frame, -1,            "Frame to restart from (-1 = no restart)");
    }

    ExportFormat to_export_format() const {
        if (format == "geo") return ExportFormat::GEO;
        if (format == "ply") return ExportFormat::PLY;
        if (format == "usd") return ExportFormat::USD;
        return ExportFormat::OBJ;
    }

    SimParams to_sim_params() const {
        SimParams p = SimParams::zeros();
        p.fps              = fps;
        p.substeps         = substeps;
        p.mu               = E / (2.0 * (1.0 + nu));
        p.lambda           = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        p.density          = density;
        p.thickness        = thickness;
        p.kB               = kB;
        p.kpin             = kpin;
        p.gravity          = Vec3(gx, gy, gz);
        p.max_global_iters = max_substep_iters;
        p.tol_abs          = tol_abs;
        p.tol_rel          = tol_rel;
        p.d_hat            = d_hat;
        p.k_sdf            = k_sdf;
        p.eps_sdf          = eps_sdf;
        p.use_parallel     = use_parallel;
        p.write_substeps   = write_substeps;
        p.use_ccd          = use_ccd;
        p.use_ccd_guess = use_ccd_guess;
        p.use_ogc = use_ogc;
        p.use_ogc_solver = use_ogc_solver;
        p.ogc_box_pad = ogc_box_pad;
        p.fixed_iters = fixed_iters;
        p.use_gpu                 = use_gpu;
        p.node_box_max            = node_box_max;
        p.node_box_min            = node_box_min;
        p.k_barrier                   = k_barrier;
        p.use_ticcd                   = use_ticcd;
        return p;
    }
};
