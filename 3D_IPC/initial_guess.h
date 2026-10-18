#pragma once

#include "physics.h"

#include <vector>

class BroadPhase;

// Optional diagnostics for the actual predictor execution. Updates are recorded
// only after a complete parallel color group has joined, never as an invented
// serial order within that group. No trace storage or output is used by default.
enum class ColoredCCDGuessTraceStage { Initial = 0, ColorUpdate = 1, SweepEnd = 2, LevelsBuilt = 3 };

struct ColoredCCDGuessTraceStep {
    ColoredCCDGuessTraceStage stage = ColoredCCDGuessTraceStage::Initial;
    int iteration = 0; // 1-based committed sweep; 0 is setup/the initial state.
    int active_level = -1; // -1 for legacy colored sweeps / outside an update.
    int active_color = -1;
    std::vector<int> active_vertices; // Attempted updates, including clipped/no-op moves.
    std::vector<Vec3> positions;
};

struct ColoredCCDGuessTrace {
    std::vector<Vec3> targets;
    std::vector<int> color_ids, level_ids; // -1 until the corresponding schedule exists.
    // New predictor: movement in the discarded trial sweep, used to seed levels.
    // Legacy predictor: movement in its first committed sweep.
    std::vector<double> first_sweep_movement;
    std::vector<ColoredCCDGuessTraceStep> steps;
    bool probe_completed = false; // Trial-only sweep; never counted below.
    int completed_iterations = 0;
    bool reached_targets = false;
    bool stalled = false;
};

// Move toward xhat using a global mesh-CCD step with a 0.9 TOI safety factor.
std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, BroadPhase* scratch_broad_phase = nullptr);

// Move deformable nodes toward x + intended_displacement using colored linear CCD.
// Rigid proxies stay fixed; mesh contacts preserve a 1e-8 gap or an existing smaller gap.
// Rebuild swept contacts while reusing optional broad-phase and thread-local storage.
std::vector<Vec3> collision_colored_ccd_initial_guess(
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& intended_displacement,
    const RefMesh& ref_mesh, const SimParams& params, int ccd_iterations,
    BroadPhase* scratch_broad_phase = nullptr, ColoredCCDGuessTrace* trace = nullptr);

// One trial original-colored sweep on temporary positions builds fixed levels.
// In each connected component, all deformable nodes tied for the largest actual
// trial-sweep movement seed level 0. Subsequent levels are graph distances from
// those seeds. Discard trial positions; actual updates start again at x, ordered
// by level then color. Same-level/same-color nodes update in parallel; rigid
// proxies stay fixed. ccd_iterations caps actual updating sweeps, EXCLUDING the
// separate trial pass (0 skips both trial and updates).
// Reuse the trial's targets, contact graph and coloring. Each level/color group
// retains the original predictor's heavy-first scheduling in exact-fallback mode.
// Stop after a complete sweep when every deformable node is within 1e-14 of its
// original target, or no node moved. Uses the original CCD and gap safeguards.
std::vector<Vec3> collision_colored_ccd_initial_guess_level_based(
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& intended_displacement,
    const RefMesh& ref_mesh, const SimParams& params, int ccd_iterations,
    BroadPhase* scratch_broad_phase = nullptr, ColoredCCDGuessTrace* trace = nullptr);

// Add dt^2 * gravity to xhat, then apply the global CCD guess.
std::vector<Vec3> verlet_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const SimParams& params, BroadPhase* scratch_broad_phase = nullptr);

// Apply a uniform translation driven by inertia, gravity, pins, and SDF penalties.
std::vector<Vec3> translation_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params);
