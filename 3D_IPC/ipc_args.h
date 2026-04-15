#pragma once

#include "args.h"
#include "physics.h"
#include "visualization.h"

// CLI-facing view of SimParams (with scene-placement knobs on top).
struct IPCArgs3D : ArgParser {

    // --- time integration ---
    double fps          = 30.0;   
    int    substeps     = 3;      
    int    num_frames   = 120;    

    // --- physics ---
    double mu           = 10.0;   // Pa  (first Lame parameter / shear modulus)
    double lambda       = 10.0;   // Pa  (second Lame parameter)
    double density      = 1.0;    // kg/m^3
    double thickness    = 0.1;    // m
    double kB           = 0.0;    // J   (bending stiffness; 0 disables bending)
    double kpin         = 1e7;    // N/m (pin spring stiffness)
    double gx           = 0.0;    // m/s^2
    double gy           = -9.81;  // m/s^2
    double gz           = 0.0;    // m/s^2

    // --- solver ---
    int    max_iters     = 5000;
    double tol_abs       = 1e-6;  // residual norm tolerance
    double step_weight   = 1.0;
    double d_hat         = 0.01;  // barrier activation distance; 0 disables contact
    bool   use_parallel  = true;
    bool   ccd_check     = false;
    bool   use_trust_region = false;
    bool   use_incremental_refresh = false;

    // --- mesh ---
    int    nx           = 2;      
    int    ny           = 2;      
    double width        = 0.4;    // m
    double height       = 0.4;    // m

    // --- scene placement ---
    double left_x       = -0.75;  // m
    double right_x      = 0.75;   // m
    double sheet_y      = 0.20;   // m
    double left_z       = 0.00;   // m
    double right_z      = 0.02;   // m

    // --- scene selection ---
    int         example      = 3;   // 1=two_sheets, 2=cloth_stack_low_res, 3=cloth_stack_high_res

    // --- output / restart ---
    std::string outdir       = "frames_sim3d";
    std::string format       = "geo";
    int         restart_frame = -1;

    IPCArgs3D() {
        add_double("fps",         fps,         30.0,       "Output frames per second");
        add_int   ("substeps",    substeps,    3,          "Solver substeps per frame (solver_dt = 1/(fps*substeps))");
        add_int   ("num_frames",  num_frames,  120,        "Number of frames to simulate");

        add_double("mu",          mu,          10.0,       "First Lame parameter (shear modulus)");
        add_double("lambda",      lambda,      10.0,       "Second Lame parameter");
        add_double("density",     density,     1.0,        "Mass density");
        add_double("thickness",   thickness,   0.1,        "Shell thickness");
        add_double("kB",          kB,          1e-2,       "Bending stiffness (0 disables bending)");
        add_double("kpin",        kpin,        1e7,        "Pin spring stiffness");
        add_double("gx",          gx,          0.0,        "Gravity x-component");
        add_double("gy",          gy,          -9.81,      "Gravity y-component");
        add_double("gz",          gz,          0.0,        "Gravity z-component");

        add_int   ("max_iters",   max_iters,   5000,        "Max Gauss-Seidel iterations per frame");
        add_double("tol_abs",     tol_abs,     1e-6,       "Absolute convergence tolerance (residual force)");
        add_double("step_weight", step_weight, 1.0,        "Newton step damping factor");
        add_double("d_hat",       d_hat,       0.01,       "Barrier activation distance (0 = off)");
        add_bool  ("use_parallel", use_parallel, true,     "Use parallel Gauss-Seidel (requires coloring)");

        add_bool  ("ccd_check",    ccd_check,    false,  "Run post-sweep CCD penetration check (serial + parallel)");
        add_bool  ("use_trust_region", use_trust_region, false, "Use trust-region narrow phase instead of CCD for step clamping");
        add_bool  ("use_incremental_refresh", use_incremental_refresh, false, "Refresh broad-phase BVH per moved vertex during GS sweep (default off; enable for aggressive scenes)");

        add_int   ("nx",          nx,          10,         "Mesh subdivisions in x");
        add_int   ("ny",          ny,          10,         "Mesh subdivisions in y");
        add_double("width",       width,       1.0,        "Mesh width");
        add_double("height",      height,      1.0,        "Mesh height");

        add_double("left_x",      left_x,      -0.75,      "Left sheet origin x");
        add_double("right_x",     right_x,     0.75,       "Right sheet origin x");
        add_double("sheet_y",     sheet_y,     0.20,       "Shared sheet origin y");
        add_double("left_z",      left_z,      0.00,       "Left sheet origin z");
        add_double("right_z",     right_z,     0.02,       "Right sheet origin z");

        add_int   ("example",      example,       3,              "Scene to run: 1=two_sheets, 2=cloth_stack_low_res, 3=cloth_stack_high_res");

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
        SimParams p;
        p.fps              = fps;
        p.substeps         = substeps;
        p.mu               = mu;
        p.lambda           = lambda;
        p.density          = density;
        p.thickness        = thickness;
        p.kB               = kB;
        p.kpin             = kpin;
        p.gravity          = Vec3(gx, gy, gz);
        p.max_global_iters = max_iters;
        p.tol_abs          = tol_abs;
        p.step_weight      = step_weight;
        p.d_hat            = d_hat;
        p.restart_frame    = restart_frame;
        p.use_parallel     = use_parallel;
        p.ccd_check        = ccd_check;
        p.use_trust_region = use_trust_region;
        p.use_incremental_refresh = use_incremental_refresh;
        return p;
    }
};
