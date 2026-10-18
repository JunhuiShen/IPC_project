#include "initial_guess.h"
#include "broad_phase.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr int count = 12;
constexpr double dt = 0.2, middle_y = 1.05;

struct Options {
    fs::path outdir;
    int iterations = 10;
    double gap = 0.05;
    bool exact = false;
    bool original = false;
};

const char* method_name(const Options& options) { return options.original ? "original" : "level"; }
const char* function_name(const Options& options) {
    return options.original ? "collision_colored_ccd_initial_guess" : "collision_colored_ccd_initial_guess_level_based";
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "Usage: colored_ccd_new_visual_test --outdir EMPTY_DIRECTORY\n"
                << "  [--method original|level] [--ccd_iterations 10] [--gap 0.05]\n"
                << "  [--exact_computation_fallback false]\n"
                << "Three squares free-fall in ONE dt=0.2 predictor step. Parallel execution enabled.\n"
                << "The middle square has side 2; the upper/lower squares have side 1.\n"
                << "--gap is the vertical spacing on each side of the middle square.\n"
                << "--method level (default): discard one color probe, then update by level/color.\n"
                << "--method original: update immediately by color; no probe or levels.\n"
                << "--ccd_iterations caps actual sweeps; the level method's probe is excluded.\n";
            std::exit(0);
        }
        if (++i == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[i];
        std::size_t consumed = 0;
        if (key == "--outdir") options.outdir = value;
        else if (key == "--method") {
            if (value != "original" && value != "level")
                throw std::invalid_argument("--method must be original or level");
            options.original = value == "original";
        }
        else if (key == "--exact_computation_fallback") {
            if (value != "true" && value != "false") throw std::invalid_argument("Expected true or false");
            options.exact = value == "true";
        } else if (key == "--ccd_iterations") {
            const long long n = std::stoll(value, &consumed);
            if (consumed != value.size() || n < 0 || n >= std::numeric_limits<int>::max())
                throw std::invalid_argument("Invalid CCD iteration count");
            options.iterations = static_cast<int>(n);
        } else if (key == "--gap") {
            options.gap = std::stod(value, &consumed);
            if (consumed != value.size()) throw std::invalid_argument("Invalid gap");
        } else throw std::invalid_argument("Unknown option " + key);
    }
    if (options.outdir.empty()) throw std::invalid_argument("--outdir is required");
    if (!std::isfinite(options.gap) || options.gap <= 0
        || !std::isfinite(middle_y - options.gap) || middle_y - options.gap == middle_y
        || !std::isfinite(middle_y + options.gap) || middle_y + options.gap == middle_y)
        throw std::invalid_argument("Gap must be positive, finite, and representable");
    return options;
}

std::ofstream output(const fs::path& path) {
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path);
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    return out;
}

std::string numbered(const char* prefix, std::size_t index) {
    std::ostringstream out;
    out << prefix << std::setfill('0') << std::setw(4) << index << ".geo";
    return out.str();
}

const char* stage_name(ColoredCCDGuessTraceStage stage) {
    switch (stage) {
    case ColoredCCDGuessTraceStage::Initial: return "Initial";
    case ColoredCCDGuessTraceStage::ColorUpdate: return "ColorUpdate";
    case ColoredCCDGuessTraceStage::SweepEnd: return "SweepEnd";
    case ColoredCCDGuessTraceStage::LevelsBuilt: return "LevelsBuilt";
    }
    throw std::invalid_argument("Unknown trace stage");
}

bool identical(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(a[i].data(), b[i].data(), 3 * sizeof(double))) return false;
    return true;
}

Vec3 display_color(int color) {
    static const std::array<Vec3, count> palette = {{
        Vec3(.90,.18,.15), Vec3(.15,.70,.25), Vec3(.15,.40,1), Vec3(1,.65,.05),
        Vec3(.65,.25,.90), Vec3(.05,.80,.85), Vec3(1,.35,.65), Vec3(.70,.80,.10),
        Vec3(.55,.30,.10), Vec3(.95,.65,.85), Vec3(.10,.45,.40), Vec3(.35,.30,.65)}};
    return color < 0 ? Vec3(.55,.55,.55) : palette.at(static_cast<std::size_t>(color));
}

struct Geometry {
    std::vector<Vec3> positions;
    std::vector<int> indices, node;
    int primitive_size = 0;
    bool colored = false;
};

Geometry surface_or_nodes(const std::vector<Vec3>& positions,
                          const std::vector<int>& triangles, bool colored) {
    Geometry result{positions, triangles, {}, triangles.empty() ? 0 : 3, colored};
    result.node.resize(positions.size());
    std::iota(result.node.begin(), result.node.end(), 0);
    return result;
}

void segment(Geometry& geo, int node, const Vec3& a, const Vec3& b) {
    const int base = static_cast<int>(geo.positions.size());
    geo.positions.push_back(a); geo.positions.push_back(b);
    geo.node.push_back(node); geo.node.push_back(node);
    geo.indices.push_back(base); geo.indices.push_back(base + 1);
    geo.primitive_size = 2; geo.colored = true;
}

Geometry arrows(const std::vector<Vec3>& starts, const std::vector<Vec3>& ends) {
    Geometry result;
    for (int node = 0; node < count; ++node) {
        segment(result, node, starts[node], ends[node]);
        const Vec3 delta = ends[node] - starts[node];
        const double length = delta.stableNorm();
        if (length == 0) continue; // Preserve the zero-length shaft and its attributes.
        const Vec3 direction = delta / length, wing = direction.unitOrthogonal();
        const double head = std::min(.03, .2 * length);
        segment(result, node, ends[node], ends[node] - head * direction + .4 * head * wing);
        segment(result, node, ends[node], ends[node] - head * direction - .4 * head * wing);
    }
    return result;
}

Geometry labels(const std::vector<Vec3>& positions, const std::vector<int>& colors,
                const std::vector<int>& levels) {
    // Separate seven-segment display glyphs in the XY plane, facing +Z.
    constexpr double ends[7][4] = {
        {0,1,1,1}, {1,1,1,.5}, {1,.5,1,0}, {0,0,1,0},
        {0,.5,0,0}, {0,1,0,.5}, {0,.5,1,.5}};
    constexpr unsigned masks[10] = {63,6,91,79,102,109,125,7,127,111};
    Geometry result;
    for (int node = 0; node < count; ++node) {
        const std::string label = "C" + std::to_string(colors[node])
            + " L" + std::to_string(levels[node]);
        const Vec3 anchor = positions[node] + Vec3(.035,.035,0);
        for (std::size_t digit = 0; digit < label.size(); ++digit) {
            const char glyph = label[digit];
            const unsigned mask = glyph == 'C' ? 57 : glyph == 'L' ? 56
                : glyph == ' ' ? 0 : glyph == '-' ? 64 : masks[glyph - '0'];
            for (int s = 0; s < 7; ++s) if (mask & (1u << s)) {
                const Vec3 a = anchor + Vec3(.04*(1.4*digit + ends[s][0]), .07*ends[s][1], 0);
                const Vec3 b = anchor + Vec3(.04*(1.4*digit + ends[s][2]), .07*ends[s][3], 0);
                segment(result, node, a, b);
            }
        }
    }
    return result;
}

struct Attributes {
    const std::vector<Vec3>& initial;
    const std::vector<Vec3>& intended;
    const std::vector<Vec3>& previous;
    const ColoredCCDGuessTrace& trace;
    const ColoredCCDGuessTraceStep& step;
    const std::vector<int>& colors;
    const std::vector<int>& levels;
    std::array<int, count> active{};
    int index = 0;
    bool levels_ready = false, probe_completed = false, first_sweep_complete = false;
    bool original = false;
};

void write_geo(const fs::path& path, const Geometry& geo, const Attributes& data) {
    auto out = output(path);
    const auto primitives = geo.primitive_size ? geo.indices.size() / geo.primitive_size : 0;
    out << "[\"fileversion\",\"18.5.408\",\"hasindex\",false,\"pointcount\"," << geo.positions.size()
        << ",\"vertexcount\"," << geo.indices.size() << ",\"primitivecount\"," << primitives
        << ",\"info\",{},\"topology\",[\"pointref\",[\"indices\",[";
    for (std::size_t i = 0; i < geo.indices.size(); ++i) out << (i ? "," : "") << geo.indices[i];
    out << "]]],\"attributes\",[\"pointattributes\",[";
    bool first = true;
    const auto attr = [&](const char* name, int size, bool integer, auto value) {
        const char* storage = integer ? "int32" : "fpreal64";
        if (!first) out << ',';
        first = false;
        out << "[[\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"" << name
            << "\",\"options\",{}],[\"size\"," << size << ",\"storage\",\"" << storage
            << "\",\"values\",[\"size\"," << size << ",\"storage\",\"" << storage << "\",\"tuples\",[";
        for (std::size_t i = 0; i < geo.positions.size(); ++i) {
            out << (i ? ",[" : "[");
            for (int axis = 0; axis < size; ++axis)
                out << (axis ? "," : "") << value(i, geo.node[i], axis);
            out << ']';
        }
        out << "]]]]";
    };
    attr("P", 3, false, [&](auto i, int, int a) { return geo.positions[i][a]; });
    if (geo.colored)
        attr("Cd", 3, false, [&](auto, int n, int a) { return display_color(data.colors[n])[a]; });
    attr("pscale", 1, false, [](auto, int, int) { return .018; });
    attr("id", 1, true, [](auto, int n, int) { return n; });
    attr("layer_id", 1, true, [](auto, int n, int) { return n / 4; });
    attr("color_id", 1, true, [&](auto, int n, int) { return data.colors[n]; });
    attr("level_id", 1, true, [&](auto, int n, int) { return data.levels[n]; });
    attr("active", 1, true, [&](auto, int n, int) { return data.active[n]; });
    attr("step_index", 1, true, [&](auto, int, int) { return data.index; });
    attr("iteration", 1, true, [&](auto, int, int) { return data.step.iteration; });
    attr("stage", 1, true, [&](auto, int, int) { return static_cast<int>(data.step.stage); });
    attr("active_level", 1, true, [&](auto, int, int) { return data.step.active_level; });
    attr("active_color", 1, true, [&](auto, int, int) { return data.step.active_color; });
    attr("levels_ready", 1, true, [&](auto, int, int) { return int(data.levels_ready); });
    attr("probe_completed", 1, true, [&](auto, int, int) { return int(data.probe_completed); });
    attr("first_sweep_complete", 1, true, [&](auto, int, int) { return int(data.first_sweep_complete); });
    attr("node_position", 3, false, [&](auto, int n, int a) { return data.step.positions[n][a]; });
    attr("initial_position", 3, false, [&](auto, int n, int a) { return data.initial[n][a]; });
    attr("intended_displacement", 3, false, [&](auto, int n, int a) { return data.intended[n][a]; });
    attr("target", 3, false, [&](auto, int n, int a) { return data.trace.targets[n][a]; });
    attr("remaining", 3, false, [&](auto, int n, int a) { return data.trace.targets[n][a] - data.step.positions[n][a]; });
    attr("accepted_displacement", 3, false, [&](auto, int n, int a) { return data.step.positions[n][a] - data.previous[n][a]; });
    attr("movement", 1, false, [&](auto, int n, int) { return (data.step.positions[n] - data.previous[n]).stableNorm(); });
    attr("remaining_distance", 1, false, [&](auto, int n, int) { return (data.trace.targets[n] - data.step.positions[n]).stableNorm(); });
    attr("first_sweep_movement", 1, false, [&](auto, int n, int) {
        return (data.original ? data.first_sweep_complete : data.probe_completed)
            && data.trace.first_sweep_movement.size() == count
            ? data.trace.first_sweep_movement[n] : -1.0;
    });
    attr("probe_movement", 1, false, [&](auto, int n, int) {
        return data.probe_completed && data.trace.first_sweep_movement.size() == count
            ? data.trace.first_sweep_movement[n] : -1.0;
    });
    out << "]],\"primitives\",[";
    if (primitives) out << "[[\"type\",\"Polygon_run\"],[\"startvertex\",0,\"nprimitives\","
        << primitives << ",\"nvertices_rle\",[" << geo.primitive_size << ',' << primitives << "]]]";
    out << "]]\n";
    out.close();
}

template<class T> void write_array(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}
void write_positions(std::ostream& out, const std::vector<Vec3>& positions) {
    out << '[';
    for (std::size_t i = 0; i < positions.size(); ++i)
        out << (i ? ",[" : "[") << positions[i].x() << ',' << positions[i].y() << ',' << positions[i].z() << ']';
    out << ']';
}

void write_trace(const fs::path& path, const Options& options, const std::vector<Vec3>& initial,
    const std::vector<Vec3>& intended, const std::vector<int>& triangles, const ColoredCCDGuessTrace& trace) {
    auto out = output(path);
    out << "{\n\"method\":\"" << method_name(options) << "\",\"function\":\"" << function_name(options) << "\","
        << "\"plain_vs_traced_bitwise_equal\":true,\"inputs_unchanged_bitwise\":true,\n"
        << "\"dt\":" << dt << ",\"gravity\":[0,-9.81,0],\"gap\":" << options.gap
        << ",\"d_hat\":0.005,\"use_parallel\":true,\"exact_computation_fallback\":" << (options.exact ? "true" : "false")
        << ",\"ccd_iterations_limit\":" << options.iterations << ",\"completed_iterations\":" << trace.completed_iterations
        << ",\"probe_completed\":" << (trace.probe_completed ? "true" : "false")
        << ",\"iteration_semantics\":\"" << (options.original
            ? "actual original-color sweeps; no probe or levels"
            : "actual level/color sweeps; excludes the discarded original-color probe") << "\""
        << ",\"first_sweep_movement_semantics\":\"" << (options.original
            ? "committed movement during actual color sweep 1; visible after its SweepEnd"
            : "probe distances used to select levels, not committed displacement") << "\""
        << ",\"level_semantics\":\"" << (options.original
            ? "not applicable; level IDs remain -1 throughout"
            : "BFS distance from maximum probe-movement seeds") << "\""
        << ",\"reached_targets\":" << (trace.reached_targets ? "true" : "false") << ",\"stalled\":" << (trace.stalled ? "true" : "false")
        << ",\n\"initial_positions\":"; write_positions(out, initial);
    out << ",\n\"intended_displacement\":"; write_positions(out, intended);
    out << ",\n\"triangles\":"; write_array(out, triangles);
    out << ",\n\"targets\":"; write_positions(out, trace.targets);
    out << ",\n\"color_ids\":"; write_array(out, trace.color_ids);
    out << ",\n\"level_ids\":"; write_array(out, trace.level_ids);
    out << ",\n\"first_sweep_movement\":"; write_array(out, trace.first_sweep_movement);
    out << ",\n\"steps\":[\n";
    bool probe_completed = false;
    for (std::size_t i = 0; i < trace.steps.size(); ++i) {
        const auto& step = trace.steps[i];
        if (step.stage == ColoredCCDGuessTraceStage::LevelsBuilt) probe_completed = true;
        out << (i ? ",\n{" : "{") << "\"step_index\":" << i << ",\"stage\":" << static_cast<int>(step.stage)
            << ",\"stage_name\":\"" << stage_name(step.stage) << "\",\"iteration\":" << step.iteration
            << ",\"probe_completed\":" << (probe_completed ? "true" : "false")
            << ",\"active_level\":" << step.active_level << ",\"active_color\":" << step.active_color << ",\"active_vertices\":";
        write_array(out, step.active_vertices);
        out << ",\"positions\":"; write_positions(out, step.positions);
        out << '}';
    }
    out << "\n]}\n";
    out.close();
}

void validate(const Options& options, const ColoredCCDGuessTrace& trace, const std::vector<Vec3>& initial,
              const std::vector<Vec3>& targets, const std::vector<Vec3>& result) {
    if (trace.steps.empty() || trace.steps.front().stage != ColoredCCDGuessTraceStage::Initial
        || !identical(trace.steps.front().positions, initial) || !identical(trace.steps.back().positions, result)
        || !identical(trace.targets, targets)) throw std::runtime_error("Trace endpoints/targets differ from the actual call");
    if ((!trace.color_ids.empty() && trace.color_ids.size() != count)
        || (!trace.level_ids.empty() && trace.level_ids.size() != count)
        || (!trace.first_sweep_movement.empty() && trace.first_sweep_movement.size() != count))
        throw std::runtime_error("Invalid trace attribute sizes");
    int completed = 0;
    bool probe_completed = false;
    int previous_level = -1, previous_color = -1;
    std::vector<Vec3> previous = initial;
    for (const auto& step : trace.steps) {
        if (step.positions.size() != count) throw std::runtime_error("Trace node count changed");
        for (const auto& p : step.positions) if (!p.allFinite()) throw std::runtime_error("Nonfinite trace position");
        std::array<bool, count> seen{};
        for (int node : step.active_vertices) {
            if (node < 0 || node >= count || seen[node]) throw std::runtime_error("Invalid active trace vertices");
            seen[node] = true;
        }
        if (step.stage == ColoredCCDGuessTraceStage::LevelsBuilt) {
            if (options.original || probe_completed || completed != 0 || step.iteration != 0
                || trace.level_ids.size() != count || !identical(step.positions, initial)
                || !step.active_vertices.empty() || step.active_level != -1 || step.active_color != -1)
                throw std::runtime_error("LevelsBuilt must record only probe metadata, with original positions unchanged");
            probe_completed = true;
        } else if (step.stage == ColoredCCDGuessTraceStage::ColorUpdate) {
            if (step.iteration != completed + 1 || step.active_color < 0
                || (options.original ? step.active_level != -1 : (!probe_completed || step.active_level < 0)))
                throw std::runtime_error("Color update does not belong to the selected method's actual sweep");
            if (step.active_level < previous_level
                || (step.active_level == previous_level && step.active_color <= previous_color))
                throw std::runtime_error("Update groups are not in increasing color or level/color order");
            previous_level = step.active_level;
            previous_color = step.active_color;
            for (int node : step.active_vertices)
                if (trace.color_ids.at(node) != step.active_color
                    || (!options.original && trace.level_ids.at(node) != step.active_level))
                    throw std::runtime_error("Active vertex does not match its group's actual color/level");
            for (int n = 0; n < count; ++n)
                if (!seen[n] && std::memcmp(previous[n].data(), step.positions[n].data(), 3 * sizeof(double)))
                    throw std::runtime_error("An inactive node moved in a color update");
        } else if (step.stage == ColoredCCDGuessTraceStage::SweepEnd) {
            if ((!options.original && !probe_completed)
                || step.iteration != completed + 1 || !identical(step.positions, previous))
                throw std::runtime_error("Invalid real-sweep endpoint");
            ++completed;
            previous_level = previous_color = -1;
        } else if (step.iteration != 0 || !identical(step.positions, initial)) {
            throw std::runtime_error("Invalid initial snapshot");
        }
        previous = step.positions;
    }
    if (completed != trace.completed_iterations) throw std::runtime_error("SweepEnd count disagrees with completed iterations");
    if (probe_completed != trace.probe_completed) throw std::runtime_error("Probe status disagrees with LevelsBuilt event");
    for (int color : trace.color_ids) if (color < -1 || color >= count) throw std::runtime_error("Invalid color ID");
    if (options.original) for (int level : trace.level_ids)
        if (level != -1) throw std::runtime_error("Original colored CCD must not assign levels");
}
int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        if (fs::exists(options.outdir) && (!fs::is_directory(options.outdir) || !fs::is_empty(options.outdir)))
            throw std::invalid_argument("Output directory must be new or empty; choose a new --outdir");
        RefMesh mesh;
        mesh.num_positions = count; mesh.mass.assign(count, 1.0);
        // Preserve the existing sheets/IDs and append the new small upper sheet.
        mesh.tris = {0,2,1,0,3,2,4,6,5,4,7,6,8,10,9,8,11,10};
        const double lower = middle_y - options.gap, upper = middle_y + options.gap;
        const std::vector<Vec3> initial = {
            Vec3(-1,middle_y,-1), Vec3(1,middle_y,-1), Vec3(1,middle_y,1), Vec3(-1,middle_y,1),
            Vec3(-.5,lower,-.5), Vec3(.5,lower,-.5), Vec3(.5,lower,.5), Vec3(-.5,lower,.5),
            Vec3(-.5,upper,-.5), Vec3(.5,upper,-.5), Vec3(.5,upper,.5), Vec3(-.5,upper,.5)};
        SimParams params = SimParams::zeros();
        params.fps = 1.0/dt; params.substeps = 1; params.gravity = Vec3(0,-9.81,0);
        params.d_hat = .005; params.use_parallel = true; params.use_ticcd = false;
        params.exact_computation_fallback = options.exact;
        const std::vector<Vec3> intended(count, dt*dt*params.gravity);
        auto x = initial, displacement = intended, targets = initial;
        for (int n = 0; n < count; ++n) targets[n] += intended[n];
        BroadPhase plain_scratch, traced_scratch;
        const auto guess = options.original ? collision_colored_ccd_initial_guess : collision_colored_ccd_initial_guess_level_based;
        const auto plain = guess(x, displacement, mesh, params, options.iterations, &plain_scratch, nullptr);
        ColoredCCDGuessTrace trace;
        const auto traced = guess(x, displacement, mesh, params, options.iterations, &traced_scratch, &trace);
        if (!identical(plain, traced)) throw std::runtime_error("Plain/traced results differ bitwise");
        if (!identical(x, initial) || !identical(displacement, intended)) throw std::runtime_error("An input was modified");
        validate(options, trace, initial, targets, traced);
        fs::create_directories(options.outdir);
        write_trace(options.outdir/"trace.json", options, initial, intended, mesh.tris, trace);
        auto table = output(options.outdir/"steps.tsv");
        const char* header = "step_index\titeration\tstage\tactive_level\tactive_color\tactive_vertices\tlevels_ready\tprobe_completed\tmax_movement\tmax_remaining\n";
        std::cout << "Method: " << method_name(options) << " (" << function_name(options) << ")\n"
            << (options.original ? "Original color sweeps commit motion immediately; levels are not applicable (-1).\n"
                                 : "A discarded color probe selects levels; only subsequent level/color sweeps commit motion.\n");
        table << header; std::cout << std::setprecision(17) << header;
        std::vector<Vec3> previous = initial;
        std::vector<int> colors = trace.color_ids, levels(count, -1);
        if (colors.empty()) colors.assign(count, -1);
        bool levels_ready = false, first_sweep_complete = false;
        for (std::size_t i = 0; i < trace.steps.size(); ++i) {
            const auto& step = trace.steps[i];
            if (step.stage == ColoredCCDGuessTraceStage::LevelsBuilt) {
                levels = trace.level_ids; levels_ready = true;
            }
            if (step.stage == ColoredCCDGuessTraceStage::SweepEnd && step.iteration >= 1) first_sweep_complete = true;
            Attributes data{initial, intended, previous, trace, step, colors, levels};
            data.index = static_cast<int>(i); data.levels_ready = levels_ready; data.first_sweep_complete = first_sweep_complete;
            data.probe_completed = levels_ready && trace.probe_completed;
            data.original = options.original;
            for (int n : step.active_vertices) data.active[n] = 1;
            write_geo(options.outdir/numbered("cloth_",i), surface_or_nodes(step.positions,mesh.tris,false), data);
            write_geo(options.outdir/numbered("nodes_",i), surface_or_nodes(step.positions,{},true), data);
            write_geo(options.outdir/numbered("displacement_",i), arrows(step.positions,targets), data);
            write_geo(options.outdir/numbered("accepted_displacement_",i), arrows(previous,step.positions), data);
            write_geo(options.outdir/numbered("vertex_labels_",i), labels(step.positions,colors,levels), data);
            if (i == 0) write_geo(options.outdir/"target_0000.geo", surface_or_nodes(targets,mesh.tris,false), data);
            double moved = 0, remaining = 0;
            for (int n = 0; n < count; ++n) {
                moved = std::max(moved,(step.positions[n]-previous[n]).stableNorm());
                remaining = std::max(remaining,(targets[n]-step.positions[n]).stableNorm());
            }
            std::ostringstream row; row.imbue(std::locale::classic());
            row << std::setprecision(17) << i << '\t' << step.iteration << '\t' << stage_name(step.stage)
                << '\t' << step.active_level << '\t' << step.active_color << '\t';
            for (std::size_t j = 0; j < step.active_vertices.size(); ++j) row << (j ? "," : "") << step.active_vertices[j];
            row << '\t' << levels_ready << '\t' << data.probe_completed << '\t' << moved << '\t' << remaining << '\n';
            table << row.str(); std::cout << row.str();
            previous = step.positions;
        }
        table.close();
        auto guide = output(options.outdir/"usage.txt");
        guide << (options.original ? "THREE FREE-FALLING SQUARES: ORIGINAL COLORED CCD\n\n"
                                  : "THREE FREE-FALLING SQUARES: DISCARDED COLOR PROBE, THEN BFS LEVEL + COLOR\n\n")
            << "Producer: " << function_name(options) << ", optional function-level trace.\n"
            << "Plain/traced outputs are bitwise identical; input positions/displacements are bitwise unchanged.\n"
            << "Middle (previous top): id 0..3, x/z +/-1, y=" << middle_y << ".\n"
            << "Lower (unchanged): id 4..7, x/z +/-0.5, y=" << lower << ".\n"
            << "Upper (new): id 8..11, x/z +/-0.5, y=" << upper << ".\n"
            << "All three layers are deformable and free falling; none is fixed or rigid.\n"
            << "dt=" << dt << ", gravity=(0,-9.81,0), intended y displacement=" << intended[0].y()
            << ", gap=" << options.gap << ", d_hat=0.005, parallel=true, exact_fallback=" << options.exact << ".\n"
            << "Maximum actual " << (options.original ? "color" : "level/color") << " sweeps="
            << options.iterations << "; completed=" << trace.completed_iterations
            << "; reached_targets=" << trace.reached_targets << "; stalled=" << trace.stalled << ".\n\n"
            << "These snapshots are events of ONE physical dt=0.2 step, not successive time steps.\n"
            << "Gravity is applied once; every update retries the remaining displacement toward the SAME target.\n"
            << "ColorUpdate follows completion of a whole independent color group. It does not imply any\n"
            << "serial order among its parallel vertices. No snapshots are invented after early termination.\n";
        if (options.original)
            guide << "The ORIGINAL function runs ordinary color sweeps directly on the committed positions.\n"
                << "Sweep 1 already moves vertices. There is NO discarded probe and NO level construction.\n"
                << "LevelsBuilt never occurs; levels_ready/probe_completed remain false; all level IDs are -1.\n"
                << "Here L-1 means NOT APPLICABLE, not an uncomputed level. With limit 0, no sweep runs.\n\n";
        else
            guide << "An original-color trial sweep runs on a temporary copy solely to select BFS levels.\n"
                << "Its trial positions are discarded: LevelsBuilt is iteration 0 at the unchanged initial positions.\n"
                << "Only level/color sweeps commit motion, numbered 1..maximum. The probe does NOT consume\n"
                << "this limit. No trial ColorUpdate/SweepEnd events are exported. With limit 0, no probe runs.\n"
                << "Probe completed=" << trace.probe_completed << ".\n\n";
        guide << "HOUDINI: set timeline 0.." << trace.steps.size()-1 << ". Load separate File SOPs:\n";
        for (const char* pattern : {"cloth_$F4.geo","nodes_$F4.geo","vertex_labels_$F4.geo",
                                    "displacement_$F4.geo","accepted_displacement_$F4.geo"})
            guide << "  " << (fs::absolute(options.outdir)/pattern).string() << '\n';
        guide << "Merge the desired display layers. Cloth is neutral; nodes/arrows use fixed actual CCD colors.\n"
            << "nodes is a 12-point cloud: enable point display or Copy to Points a small sphere using pscale.\n"
            << "vertex_labels displays C<color_id> L<level_id> beside each vertex, e.g. C6 L1.\n"
            << "C means color number; L means level number ("
            << (options.original ? "-1 throughout: no levels in the original method" : "-1 until levels are built") << ").\n"
            << "The separate line glyphs lie in the world XY plane, facing +Z.\n"
            << "target_0000.geo is an optional static surface at the original target.\n"
            << "displacement arrows: CURRENT to TARGET. accepted_displacement arrows: PREVIOUS EVENT to CURRENT.\n"
            << "Both arrow sequences include all 12 nodes of all three layers; zero-length shafts retain attributes.\n\n"
            << "POINT ATTRIBUTES (Geometry Spreadsheet)\n"
            << "id: original vertex (repeated at arrow/label points); layer_id: 0 middle, 1 lower, 2 upper.\n"
            << "color_id/Cd: actual fixed CCD color; gray/-1 means no color was built.\n"
            << (options.original ? "level_id/active_level: always -1 (not applicable). levels_ready/probe_completed: always false.\n"
                                : "level_id: -1 before LevelsBuilt, then actual fixed BFS level; 0 denotes a seed.\n"
                                  "levels_ready latches at LevelsBuilt; final levels are never shown in earlier snapshots.\n"
                                  "probe_completed latches at LevelsBuilt: the probe has selected levels, without committing motion.\n")
            << "active: member of this color-update group, including members with zero accepted motion.\n"
            << "step_index: chronological file number; iteration: actual " << (options.original ? "color" : "level/color")
            << " sweep (0 initial" << (options.original ? "" : "/LevelsBuilt") << ").\n"
            << "stage: 0 Initial, 1 ColorUpdate, 2 SweepEnd, 3 LevelsBuilt.\n"
            << "active_color: updating color group, -1 when inapplicable.\n"
            << "P: display point; node_position: exact solver vertex; initial_position/intended_displacement/target: fixed vectors.\n"
            << "remaining: target-current vector; remaining_distance: its magnitude.\n"
            << "accepted_displacement/movement: vector/distance since preceding trace event.\n"
            << (options.original ? "first_sweep_movement: committed movement in sweep 1; -1 until SweepEnd of iteration 1.\n"
                                  "probe_movement: always -1; the original method has no probe.\n"
                                : "probe_movement/first_sweep_movement: the SAME discarded-probe distance used to select seeds,\n"
                                  "-1 until LevelsBuilt. It is not displacement committed by an actual sweep.\n")
            << "first_sweep_complete: whether the first actual sweep has finished.\n"
            << "pscale: optional marker radius. All floats use fpreal64 and 17 significant digits.\n\n"
            << "steps.tsv/console table provide event order; SweepEnd rows identify complete-sweep endpoints.\n"
            << "trace.json preserves the full function trace, original targets, connectivity, final levels, and settings.\n";
        guide.close();
        std::cout << "Exported " << trace.steps.size() << " events to " << fs::absolute(options.outdir)
            << "\nDiscarded original-color probe: "
            << (options.original ? "not applicable" : trace.probe_completed ? "completed" : "not run")
            << "\nCompleted actual " << (options.original ? "color" : "level/color")
            << " sweeps: " << trace.completed_iterations << "; reached_targets=" << trace.reached_targets
            << "; stalled=" << trace.stalled << "; plain/traced bitwise check: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "colored_ccd_new_visual_test: " << error.what() << '\n';
        return 1;
    }
}
