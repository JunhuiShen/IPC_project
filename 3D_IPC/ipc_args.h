#pragma once

#include "args.h"
#include "physics.h"
#include "visualization.h"

// ======================================================
// IPCArgs3D -- simulation parameters for 3D IPC
// ======================================================

struct IPCArgs3D : ArgParser {

    // --- time integration ---
    double fps          = 30.0;
    int    substeps     = 3;
    int    num_frames   = 120;

    // --- physics ---
    double mu           = 10.0;
    double lambda       = 10.0;
    double density      = 1.0;
    double thickness    = 0.1;
    double kpin         = 1e7;
    double gx           = 0.0;
    double gy           = -9.81;
    double gz           = 0.0;

    // --- solver ---
    int    max_iters     = 500;
    double tol_abs       = 1e-6;
    double step_weight   = 1.0;
    double d_hat         = 0.02;
    bool   use_parallel  = false;

    // --- mesh ---
    int    nx           = 2;
    int    ny           = 2;
    double width        = 0.4;
    double height       = 0.4;

    // --- scene placement ---
    double left_x       = -0.75;
    double right_x      = 0.75;
    double sheet_y      = 0.20;
    double left_z       = 0.00;
    double right_z      = 0.02;

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
        add_double("kpin",        kpin,        1e7,        "Pin stiffness");
        add_double("gx",          gx,          0.0,        "Gravity x-component");
        add_double("gy",          gy,          -9.81,      "Gravity y-component");
        add_double("gz",          gz,          0.0,        "Gravity z-component");

        add_int   ("max_iters",   max_iters,   500,        "Max Gauss-Seidel iterations per frame");
        add_double("tol_abs",     tol_abs,     1e-6,       "Absolute convergence tolerance");
        add_double("step_weight", step_weight, 1.0,        "Newton step damping factor");
        add_double("d_hat",       d_hat,       0.02,        "Barrier activation distance (0 = off)");
        add_bool  ("use_parallel", use_parallel, false,    "Use parallel Gauss-Seidel (requires coloring)");

        add_int   ("nx",          nx,          10,          "Mesh subdivisions in x");
        add_int   ("ny",          ny,          10,          "Mesh subdivisions in y");
        add_double("width",       width,       1.0,        "Mesh width");
        add_double("height",      height,      1.0,        "Mesh height");

        add_double("left_x",      left_x,      -0.75,      "Left sheet origin x");
        add_double("right_x",     right_x,     0.75,       "Right sheet origin x");
        add_double("sheet_y",     sheet_y,     0.20,       "Shared sheet origin y");
        add_double("left_z",      left_z,      0.00,       "Left sheet origin z");
        add_double("right_z",     right_z,     0.02,       "Right sheet origin z");

        add_int   ("example",      example,       3,              "Scene to run: 1=two_sheets, 2=cloth_stack_low_res, 3=cloth_stack_high_res");

        add_string("outdir",       outdir,        "frames_sim3d", "Output directory");
        add_string("format",       format,        "geo",          "Output format: obj, geo, or usd");
        add_int   ("restart_frame", restart_frame, -1,            "Frame to restart from (-1 = no restart)");
    }

    ExportFormat to_export_format() const {
        if (format == "geo") return ExportFormat::GEO;
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
        p.kpin             = kpin;
        p.gravity          = Vec3(gx, gy, gz);
        p.max_global_iters = max_iters;
        p.tol_abs          = tol_abs;
        p.step_weight      = step_weight;
        p.d_hat            = d_hat;
        p.restart_frame    = restart_frame;
        p.use_parallel     = use_parallel;
        return p;
    }
};
