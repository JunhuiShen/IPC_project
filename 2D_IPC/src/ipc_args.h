#pragma once

#include "args.h"
#include "example.h"
#include "visualization.h"
#include "initial_guess/initial_guess.h"
#include <cassert>
#include <iostream>

// ======================================================
// IPCArgs — simulation parameters for 2D IPC
//
// All fields have sensible defaults matching the original
// hard-coded values.  Pass --help to see all options.
// ======================================================

struct IPCArgs : ArgParser {

    // --- time integration ---
    double dt           = 1.0 / 30.0;
    int    total_frames = 120;

    // --- physics ---
    double gx           = 0.0;
    double gy           = -9.81;
    double k_spring     = 1000.0;
    double mass_density = 1000.0;

    // --- IPC ---
    double dhat             = 0.1;
    double tol_abs          = 1e-6;
    int    max_global_iters = 10000;
    double eta              = 0.9;

    // --- geometry ---
    int number_of_nodes = 100;

    // --- output ---
    std::string outdir        = "frames_spring_IPC_bvh";
    std::string output_format = "geo";   // "obj" | "geo"

    // --- strategy (stored as strings, converted via getters) ---
    std::string example_type        = "1";    // "1" | "2"
    std::string step_policy         = "ccd";  // "ccd" | "trust_region"
    std::string initial_guess_type  = "ccd";  // "ccd" | "trust_region" | "trivial" | "affine"

    IPCArgs() {
        add_double("dt",              dt,              1.0/30.0,  "Timestep size (seconds)");
        add_int   ("total_frames",    total_frames,    120,       "Number of frames to simulate");

        add_double("gx",              gx,              0.0,       "Gravity x-component (m/s^2)");
        add_double("gy",              gy,              -9.81,     "Gravity y-component (m/s^2)");
        add_double("k_spring",        k_spring,        1000.0,    "Spring stiffness");
        add_double("mass_density",    mass_density,    1000.0,    "Mass density (kg/m^2)");

        add_double("dhat",            dhat,            0.1,       "IPC contact distance threshold");
        add_double("tol_abs",         tol_abs,         1e-6,      "Absolute convergence tolerance");
        add_int   ("max_iters",       max_global_iters,10000,     "Max global Newton iterations");
        add_double("eta",             eta,             0.9,       "Step-size safety factor");

        add_int   ("nodes",           number_of_nodes, 100,       "Nodes per chain");

        add_string("outdir",          outdir,          "frames_spring_IPC_bvh", "Output directory");
        add_string("format",          output_format,   "geo",     "Output format: obj | geo");

        add_string("example",         example_type,    "1",       "Example scene: 1 | 2");
        add_string("step_policy",     step_policy,     "ccd",     "Step filter: ccd | trust_region");
        add_string("initial_guess",   initial_guess_type, "ccd",  "Initial guess: ccd | trust_region | trivial | affine");
    }

    // --- typed getters ---

    ExampleType get_example_type() const {
        if (example_type == "2") return ExampleType::Example2;
        return ExampleType::Example1;
    }

    OutputFormat get_output_format() const {
        if (output_format == "obj") return OutputFormat::OBJ;
        return OutputFormat::GEO;
    }

    initial_guess::Type get_initial_guess_type() const {
        if (initial_guess_type == "trust_region") return initial_guess::Type::TrustRegion;
        if (initial_guess_type == "trivial")      return initial_guess::Type::Trivial;
        if (initial_guess_type == "affine")       return initial_guess::Type::Affine;
        return initial_guess::Type::CCD;
    }

    // Returns true if step policy is CCD (false = TrustRegion)
    bool use_ccd_step_policy() const {
        return step_policy != "trust_region";
    }

    // Call after parse(). Asserts parameter combinations are sensible.
    void validate() const {
        // eta must always be in (0, 1)
        assert(eta > 0.0 && eta < 1.0 && "eta must be in (0, 1)");

        // Trust region requires a conservative eta (originally 0.4).
        // If either trust_region flag is active, eta should be well below 1.
        const bool using_trust_region = (step_policy == "trust_region" ||
                                         initial_guess_type == "trust_region");
        if (using_trust_region) {
            if (eta > 0.5) {
                std::cerr << "Warning: trust_region is active but eta=" << eta
                          << " is high (recommended <= 0.5, original default was 0.4).\n";
            }
            assert(eta <= 0.5 && "eta must be <= 0.5 when using trust_region");
        }
    }
};
