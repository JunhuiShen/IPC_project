#pragma once

#include "args.h"
#include "physics.h"
#include "visualization.h"

// ======================================================
// IPCArgs3D — simulation parameters for 3D IPC
//
// All fields default to the values previously hard-coded
// in simulation.cpp.  Pass --help to see all options.
// ======================================================

struct IPCArgs3D : ArgParser {

    // --- time integration ---
    double fps          = 30.0;
    int    substeps     = 1;
    int    num_frames   = 100;

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
    int    max_iters    = 100;
    double tol_abs      = 1e-6;
    double step_weight  = 1.0;
    double d_hat        = 0.0;

    // --- mesh ---
    int    nx           = 10;
    int    ny           = 10;
    double width        = 2.0;
    double height       = 2.0;

    // --- output / restart ---
    std::string outdir       = "frames_sim3d";
    std::string format       = "geo";
    int         restart_frame = -1;

    IPCArgs3D() {
        add_double("fps",         fps,         30.0,      "Output frames per second");
        add_int   ("substeps",    substeps,    1,         "Solver substeps per frame (solver_dt = 1/(fps*substeps))");
        add_int   ("num_frames",  num_frames,  100,       "Number of frames to simulate");

        add_double("mu",          mu,          10.0,      "First Lame parameter (shear modulus)");
        add_double("lambda",      lambda,      10.0,      "Second Lame parameter");
        add_double("density",     density,     1.0,       "Mass density (kg/m^2)");
        add_double("thickness",   thickness,   0.1,       "Shell thickness");
        add_double("kpin",        kpin,        1e7,       "Pin stiffness");
        add_double("gx",          gx,          0.0,       "Gravity x-component (m/s^2)");
        add_double("gy",          gy,          -9.81,     "Gravity y-component (m/s^2)");
        add_double("gz",          gz,          0.0,       "Gravity z-component (m/s^2)");

        add_int   ("max_iters",   max_iters,   100,       "Max Gauss-Seidel iterations per frame");
        add_double("tol_abs",     tol_abs,     1e-6,      "Absolute convergence tolerance");
        add_double("step_weight", step_weight, 1.0,       "Newton step damping factor");
        add_double("d_hat",       d_hat,       0.0,       "Barrier activation distance (0 = off)");

        add_int   ("nx",          nx,          10,        "Mesh subdivisions in x");
        add_int   ("ny",          ny,          10,        "Mesh subdivisions in y");
        add_double("width",       width,       2.0,       "Mesh width");
        add_double("height",      height,      2.0,       "Mesh height");

        add_string("outdir",       outdir,        "frames_sim3d", "Output directory");
        add_string("format",       format,        "geo",          "Output format: obj or geo");
        add_int   ("restart_frame", restart_frame, -1,            "Frame to restart from (-1 = no restart)");
    }

    ExportFormat to_export_format() const {
        return (format == "geo") ? ExportFormat::GEO : ExportFormat::OBJ;
    }

    SimParams to_sim_params() const {
        SimParams p;
        p.fps             = fps;
        p.substeps        = substeps;
        p.mu              = mu;
        p.lambda          = lambda;
        p.density         = density;
        p.thickness       = thickness;
        p.kpin            = kpin;
        p.gravity         = Vec3(gx, gy, gz);
        p.max_global_iters = max_iters;
        p.tol_abs         = tol_abs;
        p.step_weight     = step_weight;
        p.d_hat           = d_hat;
        p.restart_frame   = restart_frame;
        return p;
    }
};
