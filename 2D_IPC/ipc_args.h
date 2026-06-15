#pragma once

#include "args.h"
#include "example.h"
#include "visualization.h"

#include <stdexcept>

// ======================================================
// IPCArgs — simulation parameters for 2D IPC
//
// All fields have sensible defaults matching the original
// hard-coded values.  Pass --help to see all options.
// ======================================================

struct IPCArgs : ArgParser {

    // --- time integration ---
    double dt           = 1.0 / 30.0;
    int    substeps     = 3;
    int    num_frames = 120;

    // --- physics ---
    double gx           = 0.0;
    double gy           = -9.81;
    double k_spring     = 1000.0;
    double k_sdf        = 1e5;
    double eps_sdf      = 0.002;
    double density = 900.0;
    double thickness = 0.001;

    // --- IPC ---
    double d_hat            = 0.005;
    double tol_abs          = 1e-6;
    int    max_substep_iters = 500;
    double eta              = 0.9;
    double k_barrier        = 100.0;
    bool   use_parallel     = true;
    double node_box_min     = 0.001;
    double node_box_max     = 0.01;
    int    node_box_update_count = 1;

    // --- geometry ---
    int number_of_nodes = 100;

    // --- output ---
    std::string outdir        = "frames_2d";
    std::string output_format = "geo";   // "obj" | "geo"
    bool        write_substeps = false;
    int         restart_frame = -1;

    // --- strategy (stored as strings, converted via getters) ---
    std::string example_type        = "1";
    std::string step_policy         = "ccd";  // "ccd" | "trust_region"
    std::string initial_guess_type  = "ccd";  // "ccd" | "trivial" | "affine"

    IPCArgs() {
        add_double("dt",              dt,              1.0/30.0,  "Timestep size (seconds)");
        add_int   ("substeps",        substeps,        3,         "Substeps per frame");
        add_int   ("num_frames",      num_frames,      120,       "Number of frames to simulate");

        add_double("gx",              gx,              0.0,       "Gravity x-component (m/s^2)");
        add_double("gy",              gy,              -9.81,     "Gravity y-component (m/s^2)");
        add_double("k_spring",        k_spring,        1000.0,    "Spring stiffness");
        add_double("k_sdf",           k_sdf,           1e5,       "SDF penalty stiffness");
        add_double("eps_sdf",         eps_sdf,         0.002,     "SDF transition band width");
        add_double("density",         density,         900.0,     "Mass density (kg/m^3)");
        add_double("thickness",       thickness,       0.001,     "Chain cross-section thickness (m)");

        add_double("d_hat",           d_hat,           0.005,     "IPC contact distance threshold");
        add_double("tol_abs",         tol_abs,         1e-6,      "Absolute convergence tolerance");
        add_int   ("max_substep_iters", max_substep_iters, 500,   "Max Gauss-Seidel iterations per substep");
        add_double("eta",             eta,             0.9,       "Step-size safety factor");
        add_double("k_barrier",       k_barrier,       100.0,     "IPC barrier stiffness multiplier");
        add_bool  ("use_parallel",    use_parallel,    true,      "Use color-parallel Gauss-Seidel updates");
        add_double("node_box_min",    node_box_min,    0.001,     "Lower bound on node box half-width");
        add_double("node_box_max",    node_box_max,    0.01,      "Upper bound on node box half-width");
        add_int   ("node_box_update_count", node_box_update_count, 1, "Gauss-Seidel iterations between node-box/contact recoloring rebuilds");

        add_int   ("nodes",           number_of_nodes, 100,       "Nodes per chain");

        add_string("outdir",          outdir,          "frames_2d", "Output directory");
        add_string("format",          output_format,   "geo",     "Output format: obj | geo");
        add_bool  ("write_substeps",  write_substeps,  false,     "Export every substep as substep_XXXX");
        add_int   ("restart_frame",   restart_frame,   -1,        "Resume from outdir/state_XXXX.bin; -1 disables restart");

        add_string("example",         example_type,    "1",       "Example scene: 1");
        add_string("step_policy",     step_policy,     "ccd",     "Step filter: ccd | trust_region");
        add_string("initial_guess",   initial_guess_type, "ccd",  "Initial guess: ccd | trivial | affine");
    }

    // --- typed getters ---

    ExampleType get_example_type() const {
        if (example_type == "1") return ExampleType::Example1;
        throw std::invalid_argument("Unknown example: " + example_type);
    }

    OutputFormat get_output_format() const {
        if (output_format == "obj") return OutputFormat::OBJ;
        if (output_format == "geo") return OutputFormat::GEO;
        throw std::invalid_argument("Unknown output format: " + output_format);
    }

    InitialGuessType get_initial_guess_type() const {
        if (initial_guess_type == "ccd")     return InitialGuessType::CCD;
        if (initial_guess_type == "trivial") return InitialGuessType::Trivial;
        if (initial_guess_type == "affine")  return InitialGuessType::Affine;
        throw std::invalid_argument("Unknown initial guess: " + initial_guess_type);
    }

    // Returns true if step policy is CCD (false = TrustRegion)
    bool use_ccd_step_policy() const {
        if (step_policy == "ccd") return true;
        if (step_policy == "trust_region") return false;
        throw std::invalid_argument("Unknown step policy: " + step_policy);
    }

    // Call after parse(). Throws if parameter combinations are invalid.
    void validate() const {
        if (!(eta > 0.0 && eta < 1.0)) throw std::invalid_argument("eta must be in (0, 1)");
        if (!(d_hat >= 0.0)) throw std::invalid_argument("d_hat must be nonnegative");
        if (!(k_sdf >= 0.0)) throw std::invalid_argument("k_sdf must be nonnegative");
        if (!(eps_sdf > 0.0)) throw std::invalid_argument("eps_sdf must be positive");
        if (!(thickness > 0.0)) throw std::invalid_argument("thickness must be positive");
        if (substeps <= 0) throw std::invalid_argument("substeps must be positive");
        if (max_substep_iters <= 0) throw std::invalid_argument("max_substep_iters must be positive");
        if (number_of_nodes < 2) throw std::invalid_argument("nodes must be at least 2");
        if (node_box_min <= 0.0) throw std::invalid_argument("node_box_min must be positive");
        if (node_box_max < node_box_min) throw std::invalid_argument("node_box_max must be >= node_box_min");
        if (node_box_update_count <= 0) throw std::invalid_argument("node_box_update_count must be positive");
        if (step_policy == "trust_region" && eta > 0.5) {
            throw std::invalid_argument("eta must be <= 0.5 when using trust_region");
        }
    }
};
