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
    double kpin         = 1e7;    // N/m (pin spring stiffness)
    double gx           = 0.0;    // m/s^2
    double gy           = -9.81;  // m/s^2
    double gz           = 0.0;    // m/s^2

    // --- solver ---
    int    max_substep_iters = 500;  // per-substep cap; per-frame total = max_substep_iters * substeps
    double tol_abs       = 1e-6;  // residual norm tolerance
    double tol_rel       = 1e-1;  // relative tolerance: stop when residual < tol_rel * initial
    double d_hat         = 0.01;  // barrier activation distance; 0 disables contact
    double k_sdf         = 1e2;   // SDF penalty stiffness; 0 disables the SDF term
    double eps_sdf       = 0.002; // SDF ramp-Heaviside transition-layer width
    bool   use_parallel  = true;
    bool   ccd_check     = false;
    bool   use_ccd_guess = true;
    bool   use_trust_region = false;
    bool   fixed_iters = false;
    bool   use_gpu                 = false;
    int    color_rebuild_interval  = 10;

    // --- mesh ---
    int    nx           = 31;     
    int    ny           = 31;     
    double width        = 1.0;    // m
    double height       = 1.0;    // m

    // --- scene placement ---
    double left_x       = -0.75;  // m
    double right_x      = 0.75;   // m
    double sheet_y      = 0.20;   // m
    double left_z       = 0.00;   // m
    double right_z      = 0.02;   // m

    // --- scene selection ---
    int         example      = 3;   // 1=two_sheets, 2=cloth_stack_low_res, 3=cloth_stack_high_res, 4=cloth_cylinder_drop, 5=twisting_cloth, 6=cloth_sphere_drop
    double      twist_rate   = 0.5; // relative twist rate in Hz for example 5 (full turns per second)
    int         twist_nx     = 99;  // grid subdivisions along x for example 5
    int         twist_ny     = 99;  // grid subdivisions along y for example 5
    double      twist_size   = 2.5; // edge length (m) of the square cloth in example 5
    int         drop_stack_count = 50; // number of falling cloths in example 4 stack
    int         drop_cloth_nx    = 16; // grid subdivisions along x of each falling cloth for example 4
    int         drop_cloth_ny    = 16; // grid subdivisions along y of each falling cloth for example 4
    double      drop_first_y     = 0.40; // y-coordinate of the lowest falling cloth at t=0 for examples 4 and 6
    double      drop_spacing     = 0.02; // vertical gap (m) between successive stacked cloths at t=0 for examples 4 and 6
    double      drop_cloth_w     = 0.70; // width (m, along x) of each falling cloth for example 4
    double      drop_cloth_h     = 0.70; // length (m, along z) of each falling cloth for example 4
    double      cyl_ground_size  = 4.0;  // edge length (m) of the square pinned ground in example 4
    int         cyl_nu        = 32;    // circumferential subdivisions for example 4 cylinder (axial row count auto-picked for iso triangles)
    double      cyl_radius    = 0.105; // cylinder collider radius (m) for example 4
    double      cyl_length    = 0.9;   // cylinder collider length (m) for example 4
    double      cyl_cx        = 0.0;   // cylinder center x for example 4
    double      cyl_cy        = 0.25;  // cylinder center y for example 4
    double      cyl_cz        = 0.0;   // cylinder center z for example 4
    double      sphere_radius = 0.10;  // sphere collider radius (m) in example 6
    double      sphere_cx     = 0.0;   // sphere center x for example 6
    double      sphere_cy     = 0.10;  // sphere center y for example 6; default y=r so sphere is tangent to ground
    double      sphere_cz     = 0.0;   // sphere center z for example 6
    int         sphere_subdiv = 2;     // icosphere subdivision level for example 6 (V = 10*4^n + 2)
    double      sphere_cloth_size = 0.70; // edge length (m) of each falling cloth in example 6
    double      sphere_ground_size = 4.0; // edge length (m) of the square pinned ground in example 6

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
        add_double("kpin",        kpin,        1e7,        "Pin spring stiffness");
        add_double("gx",          gx,          0.0,        "Gravity x-component");
        add_double("gy",          gy,          -9.81,      "Gravity y-component");
        add_double("gz",          gz,          0.0,        "Gravity z-component");

        add_int   ("max_substep_iters", max_substep_iters, 500, "Max Gauss-Seidel iterations per substep (per-frame total = max_substep_iters * substeps)");
        add_double("tol_abs",     tol_abs,     1e-6,       "Absolute convergence tolerance (residual force)");
        add_double("tol_rel",     tol_rel,     1e-1,       "Relative tolerance: stop when residual < tol_rel * initial_residual (0 disables)");
        add_double("d_hat",       d_hat,       0.01,       "Barrier activation distance (0 = off)");
        add_double("k_sdf",       k_sdf,       1e2,        "SDF penalty stiffness (0 = off; obstacles live on SimParams)");
        add_double("eps_sdf",     eps_sdf,     0.002,       "SDF penalty ramp-Heaviside transition-layer width");
        add_bool  ("use_parallel", use_parallel, true,     "Use parallel Gauss-Seidel (requires coloring)");

        add_bool  ("ccd_check",    ccd_check,    false,  "Run post-sweep CCD penetration check (serial + parallel)");
        add_bool  ("use_ccd_guess",    use_ccd_guess,    true,  "Use ccd_initial_guess as the substep start point (ignored if use_trust_region is on)");
        add_bool  ("use_trust_region", use_trust_region, false, "Use trust-region narrow phase instead of CCD for step clamping");
        add_bool  ("fixed_iters",      fixed_iters,      false, "Run exactly max_substep_iters sweeps per substep with no tolerance / convergence check");
        add_bool  ("use_gpu",              use_gpu,              false, "Route the GS sweep through the GPU implementation (CPU stub when CUDA is unavailable)");
        add_int   ("color_rebuild_interval", color_rebuild_interval, 10, "Parallel solver: recolor every N outer iterations (N<=0 treated as 1)");

        add_int   ("nx",          nx,          31,         "Mesh subdivisions in x");
        add_int   ("ny",          ny,          31,         "Mesh subdivisions in y");
        add_double("width",       width,       1.0,        "Mesh width");
        add_double("height",      height,      1.0,        "Mesh height");

        add_double("left_x",      left_x,      -0.75,      "Left sheet origin x");
        add_double("right_x",     right_x,     0.75,       "Right sheet origin x");
        add_double("sheet_y",     sheet_y,     0.20,       "Shared sheet origin y");
        add_double("left_z",      left_z,      0.00,       "Left sheet origin z");
        add_double("right_z",     right_z,     0.02,       "Right sheet origin z");

        add_int   ("example",      example,       3,              "Scene to run: 1=two_sheets, 2=cloth_stack_low_res, 3=cloth_stack_high_res, 4=cloth_cylinder_drop, 5=twisting_cloth, 6=cloth_sphere_drop");
        add_double("twist_rate",   twist_rate,    0.5,            "Relative twist rate in Hz for example 5 (turns/second; total turns = rate * duration)");
        add_int   ("twist_nx",     twist_nx,      99,             "Grid subdivisions along x for example 5 (vertices = (twist_nx+1)*(twist_ny+1))");
        add_int   ("twist_ny",     twist_ny,      99,             "Grid subdivisions along y for example 5");
        add_double("twist_size",   twist_size,    2.5,            "Edge length (m) of the square cloth in example 5");
        add_int   ("drop_stack_count", drop_stack_count, 50,       "Number of falling cloths in example 4 stack");
        add_int   ("drop_cloth_nx",    drop_cloth_nx,    16,       "Grid subdivisions along x of each falling cloth in example 4");
        add_int   ("drop_cloth_ny",    drop_cloth_ny,    16,       "Grid subdivisions along y of each falling cloth in example 4");
        add_double("drop_first_y",     drop_first_y,     0.40,     "Y-coordinate of the lowest falling cloth at t=0 in examples 4 and 6");
        add_double("drop_spacing",     drop_spacing,     0.02,     "Vertical gap (m) between successive stacked cloths at t=0 in examples 4 and 6");
        add_double("drop_cloth_w",     drop_cloth_w,     0.70,     "Width (m, along x) of each falling cloth in example 4");
        add_double("drop_cloth_h",     drop_cloth_h,     0.70,     "Length (m, along z) of each falling cloth in example 4");
        add_double("cyl_ground_size", cyl_ground_size,   4.0,      "Edge length (m) of the square pinned ground in example 4");
        add_int   ("cyl_nu",           cyl_nu,           32,       "Circumferential subdivisions for example 4 cylinder");
        add_double("cyl_radius",       cyl_radius,       0.105,    "Cylinder collider radius in meters for example 4");
        add_double("cyl_length",       cyl_length,       0.9,      "Cylinder collider length in meters for example 4");
        add_double("cyl_cx",           cyl_cx,           0.0,      "Cylinder center x for example 4");
        add_double("cyl_cy",           cyl_cy,           0.25,     "Cylinder center y for example 4");
        add_double("cyl_cz",           cyl_cz,           0.0,      "Cylinder center z for example 4");
        add_double("sphere_radius",    sphere_radius,    0.10,     "Sphere collider radius in meters in example 6");
        add_double("sphere_cx",        sphere_cx,        0.0,      "Sphere center x for example 6");
        add_double("sphere_cy",        sphere_cy,        0.10,     "Sphere center y for example 6; default y=r so sphere is tangent to ground");
        add_double("sphere_cz",        sphere_cz,        0.0,      "Sphere center z for example 6");
        add_int   ("sphere_subdiv",    sphere_subdiv,    2,        "Icosphere subdivision level in example 6 (V = 10*4^n + 2)");
        add_double("sphere_cloth_size", sphere_cloth_size, 0.70,   "Edge length in meters of each falling cloth in example 6");
        add_double("sphere_ground_size", sphere_ground_size, 4.0,  "Edge length in meters of the square pinned ground in example 6");

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
        p.ccd_check        = ccd_check;
        p.use_ccd_guess = use_ccd_guess;
        p.use_trust_region = use_trust_region;
        p.fixed_iters = fixed_iters;
        p.use_gpu                 = use_gpu;
        p.color_rebuild_interval  = color_rebuild_interval;
        return p;
    }
};
