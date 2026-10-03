#include "initial_guess.h"
#include "broad_phase.h"
#include "output.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

struct Options {
    fs::path outdir = "outputs/ccd_two_squares";
    int iterations = 10;
    double dt = 0.2, gap = 0.05, d_hat = 0.005, gravity = 9.81;
};

constexpr double upper_height = 1.05;

void usage() {
    std::cout << "Two deformable squares: one gravity predictor, repeated colored CCD sweeps.\n"
        << "Options: --outdir DIR --ccd_iterations N --dt SECONDS --gap DISTANCE\n"
        << "         --d_hat DISTANCE --gravity POSITIVE_MAGNITUDE --help\n"
        << "Defaults: outputs/ccd_two_squares, 10 sweeps, dt=0.2, gap=0.05,\n"
        << "          d_hat=0.005, gravity=9.81. Output directory must be empty.\n";
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { usage(); std::exit(0); }
        if (++i == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--outdir") { options.outdir = value; continue; }
        std::size_t consumed = 0;
        if (key == "--ccd_iterations") {
            const long long count = std::stoll(value, &consumed);
            if (count < 0 || count >= std::numeric_limits<int>::max())
                throw std::invalid_argument("invalid CCD iteration count");
            options.iterations = static_cast<int>(count);
        } else {
            const double number = std::stod(value, &consumed);
            if (!std::isfinite(number)) throw std::invalid_argument(key + " must be finite");
            if (key == "--dt") options.dt = number;
            else if (key == "--gap") options.gap = number;
            else if (key == "--d_hat") options.d_hat = number;
            else if (key == "--gravity") options.gravity = number;
            else throw std::invalid_argument("unknown option " + key);
        }
        if (consumed != value.size()) throw std::invalid_argument("invalid value for " + key);
    }
    if (options.outdir.empty() || options.dt <= 0.0 || options.gap <= 0.0
        || options.d_hat < 0.0 || options.gravity < 0.0)
        throw std::invalid_argument("require nonempty outdir, dt/gap > 0, d_hat/gravity >= 0");
    if (!std::isfinite(1.0 / options.dt) || !std::isfinite(options.dt * options.dt)
        || !std::isfinite(options.dt * options.dt * options.gravity)
        || !std::isfinite(upper_height - options.gap)
        || upper_height - options.gap == upper_height)
        throw std::invalid_argument("dt, gravity, or gap exceeds representable geometry");
    return options;
}

std::ofstream open_output(const fs::path& path) {
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path);
    out << std::setprecision(17);
    return out;
}

std::string numbered(const std::string& prefix, int iteration, const char* suffix) {
    std::ostringstream name;
    name << prefix << std::setfill('0') << std::setw(4) << iteration << suffix;
    return name.str();
}

Vec3 vertex_color(int color) {
    // At most eight collision colors in this eight-vertex demo. This mapping is
    // by the actual CCD color ID, never by which square contains the vertex.
    static const Vec3 palette[] = {
        Vec3(0.90, 0.18, 0.15), Vec3(0.15, 0.70, 0.25),
        Vec3(0.15, 0.40, 1.00), Vec3(1.00, 0.65, 0.05),
        Vec3(0.65, 0.25, 0.90), Vec3(0.05, 0.80, 0.85),
        Vec3(1.00, 0.35, 0.65), Vec3(0.70, 0.80, 0.10)};
    if (color < 0 || color >= 8)
        throw std::invalid_argument("invalid collision color in two-square demo");
    return palette[color];
}

// Same Houdini GEO schema as export_geo, with double-precision vector attributes.
// Surface files have no Cd: only the separate point cloud is colored.
void write_geo(const fs::path& path, const std::vector<Vec3>& positions,
    const std::vector<int>& triangles, const std::vector<Vec3>& initial,
    const std::vector<Vec3>& displacement, const std::vector<Vec3>& targets,
    const std::vector<Vec3>& previous, const std::vector<int>& colors, int iteration,
    bool color_points = false, int update_index = -1, int updated_color = -1) {
    if (color_points && !triangles.empty())
        throw std::invalid_argument("vertex color export must contain only points");
    auto out = open_output(path);
    out << "[\"fileversion\",\"18.5.408\",\"hasindex\",false,\"pointcount\","
        << positions.size() << ",\"vertexcount\"," << triangles.size()
        << ",\"primitivecount\"," << triangles.size() / 3
        << ",\"info\",{},\"topology\",[\"pointref\",[\"indices\",[";
    for (std::size_t i = 0; i < triangles.size(); ++i)
        out << (i ? "," : "") << triangles[i];
    out << "]]],\"attributes\",[\"pointattributes\",[";
    bool first = true;
    auto attribute = [&](const char* name, int size, bool integer, auto value) {
        const char* storage = integer ? "int32" : "fpreal64";
        if (!first) out << ',';
        first = false;
        out << "[[\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"" << name
            << "\",\"options\",{}],[\"size\"," << size << ",\"storage\",\"" << storage
            << "\",\"values\",[\"size\"," << size << ",\"storage\",\"" << storage
            << "\",\"tuples\",[";
        for (std::size_t point = 0; point < positions.size(); ++point) {
            out << (point ? ",[" : "[");
            for (int axis = 0; axis < size; ++axis)
                out << (axis ? "," : "") << value(point, axis);
            out << ']';
        }
        out << "]]]]";
    };
    attribute("P", 3, false, [&](auto i, int a) { return positions[i][a]; });
    if (color_points) {
        attribute("Cd", 3, false, [&](auto i, int a) { return vertex_color(colors[i])[a]; });
        attribute("pscale", 1, false, [](auto, int) { return 0.008; });
    }
    attribute("initial_position", 3, false, [&](auto i, int a) { return initial[i][a]; });
    attribute("intended_displacement", 3, false, [&](auto i, int a) { return displacement[i][a]; });
    attribute("remaining_displacement", 3, false, [&](auto i, int a) { return targets[i][a] - positions[i][a]; });
    attribute("accepted_displacement", 3, false, [&](auto i, int a) { return positions[i][a] - previous[i][a]; });
    attribute("target", 3, false, [&](auto i, int a) { return targets[i][a]; });
    attribute("pointid", 1, true, [](auto i, int) { return static_cast<int>(i); });
    attribute("square_id", 1, true, [](auto i, int) { return static_cast<int>(i / 4); });
    attribute("color_id", 1, true, [&](auto i, int) { return colors[i]; });
    attribute("iteration", 1, true, [&](auto, int) { return iteration; });
    if (update_index >= 0) {
        attribute("update_index", 1, true, [&](auto, int) { return update_index; });
        attribute("updated_color", 1, true, [&](auto, int) { return updated_color; });
        attribute("updated", 1, true, [&](auto i, int) {
            return static_cast<int>(updated_color >= 0 && colors[i] == updated_color);
        });
    }
    out << "]],\"primitives\",[";
    if (!triangles.empty())
        out << "[[\"type\",\"Polygon_run\"],[\"startvertex\",0,"
            << "\"nprimitives\"," << triangles.size() / 3 << ",\"nvertices_rle\",[3,"
            << triangles.size() / 3 << "]]]";
    out << "]]\n";
    out.close();
}

void write_arrows(const fs::path& path, const std::vector<Vec3>& starts,
    const std::vector<Vec3>& ends, std::size_t first = 0,
    std::size_t last = std::numeric_limits<std::size_t>::max()) {
    if (last == std::numeric_limits<std::size_t>::max()) last = starts.size();
    if (starts.size() != ends.size() || first > last || last > starts.size())
        throw std::invalid_argument("invalid arrow vertex range");
    auto out = open_output(path);
    out << "# Each arrow runs from its supplied start position to its end position.\n";
    for (std::size_t i = first; i < last; ++i) {
        const Vec3 difference = ends[i] - starts[i];
        const double length = difference.stableNorm();
        const Vec3 direction = length > 0.0 ? Vec3(difference / length) : Vec3::Zero();
        const Vec3 wing = length > 0.0 ? Vec3(direction.unitOrthogonal()) : Vec3::Zero();
        const double head = std::min(0.03, 0.2 * length);
        const Vec3 left = ends[i] - head * direction + 0.4 * head * wing;
        const Vec3 right = ends[i] - head * direction - 0.4 * head * wing;
        out << "g vertex_" << i << '\n';
        for (const Vec3& position : {starts[i], ends[i], left, right})
            out << "v " << position.x() << ' ' << position.y() << ' ' << position.z() << '\n';
        const std::size_t base = 4 * (i - first) + 1;
        out << "l " << base << ' ' << base + 1 << "\nl " << base + 2 << ' '
            << base + 1 << ' ' << base + 3 << '\n';
    }
    out.close();
}

void checked_boxes(const fs::path& path, const std::vector<AABB>& boxes) {
    export_aabb_list(path.string(), boxes);
    std::ifstream check(path);
    std::string line;
    std::size_t points = 0, edges = 0;
    while (std::getline(check, line)) {
        if (line.compare(0, 2, "v ") == 0) ++points;
        if (line.compare(0, 2, "l ") == 0) ++edges;
    }
    if (!check.eof() || points != 8 * boxes.size() || edges != 12 * boxes.size())
        throw std::runtime_error("failed writing boxes: " + path.string());
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        if (fs::exists(options.outdir) && (!fs::is_directory(options.outdir) || !fs::is_empty(options.outdir)))
            throw std::invalid_argument("output directory is not empty; choose a new --outdir");
        fs::create_directories(options.outdir);
        RefMesh mesh;
        mesh.num_positions = 8;
        mesh.mass.assign(8, 1.0);
        mesh.tris = {0, 2, 1, 0, 3, 2, 4, 6, 5, 4, 7, 6};
        std::vector<Vec3> initial;
        for (int square = 0; square < 2; ++square) {
            const double half_side = square == 0 ? 1.0 : 0.5;
            const double y = square == 0 ? upper_height : upper_height - options.gap;
            initial.emplace_back(-half_side, y, -half_side);
            initial.emplace_back( half_side, y, -half_side);
            initial.emplace_back( half_side, y,  half_side);
            initial.emplace_back(-half_side, y,  half_side);
        }
        SimParams params = SimParams::zeros();
        params.fps = 1.0 / options.dt;
        params.substeps = 1;
        params.gravity = Vec3(0.0, -options.gravity, 0.0);
        params.d_hat = options.d_hat;
        params.use_parallel = true;
        const std::vector<Vec3> displacement(8, params.dt2() * params.gravity);
        std::vector<Vec3> previous = initial;
        std::vector<Vec3> previous_update = initial;
        const fs::path updates_dir = options.outdir / "vertex_updates";
        fs::create_directories(updates_dir);
        auto update_log = open_output(updates_dir / "updates.txt");
        update_log << "Actual CCD color updates in execution order. Update 0 is the initial state.\n"
            << "Each update completes one color; singleton colors update one vertex.\n"
            << "Zero-motion updates are included. No gravity is reapplied between updates.\n";
        int update_index = 0;
        std::size_t num_colors = 0;
        auto write_update = [&](int iteration, int updated_color,
            const std::vector<Vec3>& positions, const std::vector<Vec3>& targets,
            const std::vector<int>& colors) {
            write_geo(updates_dir / numbered("mesh_", update_index, ".geo"), positions,
                mesh.tris, initial, displacement, targets, previous_update, colors, iteration,
                false, update_index, updated_color);
            write_geo(updates_dir / numbered("vertex_colors_", update_index, ".geo"), positions,
                {}, initial, displacement, targets, previous_update, colors, iteration,
                true, update_index, updated_color);
            write_arrows(updates_dir / numbered("remaining_displacement_", update_index, ".obj"), positions, targets);
            write_arrows(updates_dir / numbered("accepted_displacement_", update_index, ".obj"), previous_update, positions);
            write_arrows(updates_dir / numbered("bottom_remaining_displacement_", update_index, ".obj"), positions, targets, 4, 8);
            write_arrows(updates_dir / numbered("bottom_accepted_displacement_", update_index, ".obj"), previous_update, positions, 4, 8);
            previous_update = positions;
        };
        auto readme = open_output(options.outdir / "README.txt");
        readme << "TWO DEFORMABLE SQUARES: SINGLE FREE-FALL PREDICTOR + COLORED CCD\n"
            << "Upper: vertices 0..3, square_id=0, side 2, center (0," << upper_height << ",0).\n"
            << "Lower: vertices 4..7, square_id=1, side 1, center (0,"
            << upper_height - options.gap << ",0). Each has two triangles.\n"
            << "dt=" << params.dt() << ", gravity=(0,-" << options.gravity << ",0), d_hat=" << params.d_hat
            << ", CCD sweeps=" << options.iterations << ". Both start with zero velocity.\n"
            << "The repository's semi-implicit predictor uses v_new=dt*gravity and\n"
            << "displacement=dt*v_new=dt^2*gravity,\n"
            << "not the analytical 0.5*dt^2*gravity trajectory. Intended y displacement=" << displacement[0].y() << ".\n"
            << "Frame 0 is the original state; frame k is the actual state after k complete\n"
            << "CCD sweeps. These are iterations of ONE physical step, NOT successive time steps.\n"
            << "Gravity is applied once; every sweep retries target-current toward the same target.\n"
            << "Both squares have identical intended translation, so simultaneous physical fall\n"
            << "preserves their gap. Any partial-step clipping is from sequential color updates\n"
            << "with other colors held still, not a relative-velocity impact. No rigid bodies,\n"
            << "elasticity, bending, ground, or SDF are involved. CCD safety is 0.9.\n\n"
            << "The upper square is numbered first to expose this Gauss-Seidel clipping in the\n"
            << "default coloring order. Numbering the lower square first can let both squares\n"
            << "reach their targets within the first sweep despite identical physical inputs.\n\n"
            << "HOUDINI: Load mesh_sweep_$F4.geo in a File SOP; set playback to 0.." << options.iterations << ".\n"
            << "The cloth surfaces (including targets.geo) have NO Cd attribute and use the\n"
            << "default neutral shading; they are not colored by layer or by vertex group.\n"
            << "VERTEX COLORING: Load vertex_colors_$F4.geo in a separate File SOP. It contains\n"
            << "only the eight moving vertices, with Cd from their actual CCD color_id. Enable\n"
            << "point display to see them, or Copy to Points a unit-radius Sphere onto this\n"
            << "point cloud (pscale=0.008) to make visible colored vertex markers. Display those\n"
            << "alongside the neutral mesh; do not transfer their Cd onto the cloth triangles.\n"
            << "If merging colored markers with the cloth, first assign a neutral gray Color SOP\n"
            << "to the cloth branch to avoid missing Cd becoming black during the merge.\n"
            << "pointid gives the original vertex ID; square_id identifies the layer. color_id\n"
            << "is the collision-only scheduling group, not an elastic/material/layer color.\n"
            << "Load remaining_displacement_$F4.obj in another File SOP: arrows begin at each\n"
            << "current position and end at its original target; arrow length shrinks as it moves.\n"
            << "Alternatively add a vector Visualizer for point attribute remaining_displacement,\n"
            << "scale 1. intended_displacement is the ORIGINAL constant vector, not the remainder.\n"
            << "accepted_displacement is motion since the preceding sweep; target is the fixed\n"
            << "target point. original_intended_displacement.obj shows original arrows; targets.geo\n"
            << "shows the target surfaces. No visualization scaling changes the simulation.\n"
            << "A zero remaining arrow means that vertex already reached its target, not that\n"
            << "gravity is disabled. Both squares receive the same nonzero gravity displacement.\n"
            << "BOTTOM LAYER ONLY: Load bottom_intended_displacement.obj for the four original\n"
            << "gravity arrows, anchored at the INITIAL bottom positions; these remain visible\n"
            << "even after the bottom reaches its target. Load bottom_remaining_displacement_$F4.obj\n"
            << "for bottom current-to-target arrows, or bottom_accepted_displacement_$F4.obj for\n"
            << "bottom motion during each sweep (previous-to-current arrows; zero at sweep 0).\n"
            << "Merge any of these with mesh_sweep_$F4.geo. OBJ groups vertex_4..vertex_7\n"
            << "preserve the original bottom vertex IDs. Zero-length arrows are not artificially\n"
            << "enlarged; with default settings the bottom reaches its target in sweep 1.\n"
            << "Load blue_node_boxes.obj, green_triangle_boxes.obj, green_edge_boxes.obj, and\n"
            << "red_edge_boxes.obj separately and assign blue/green/red Color SOPs. These are\n"
            << "the actual fixed broad-phase boxes, exported once, not recomputed visual boxes.\n"
            << "Node boxes have 20% total extent padding (10% each side), plus numerical minima.\n"
            << "Green boxes additionally use d_hat; that padding is NOT a CCD clearance constraint.\n\n"
            << "VERTEX-BY-VERTEX PLAYBACK: Load vertex_updates/mesh_$F4.geo and\n"
            << "vertex_updates/vertex_colors_$F4.geo. Update frame 0 is the initial state;\n"
            << "each subsequent frame is the ACTUAL state immediately after one complete color.\n"
            << "With the default eight singleton colors, this is one vertex update per frame,\n"
            << "in color order. No replays or changes to the solver's update order are used.\n"
            << "For other settings, a color may contain multiple vertices updated in parallel;\n"
            << "those appear together in one snapshot, not as a fictitious serial ordering.\n"
            << "Point attribute updated=1 identifies the vertices just processed, even if no\n"
            << "motion was left. iteration records the CCD sweep, update_index the playback\n"
            << "frame, and updated_color the completed color (-1 on initial frame).\n"
            << "vertex_updates/remaining_displacement_$F4.obj shows current-to-target arrows;\n"
            << "vertex_updates/accepted_displacement_$F4.obj shows only the last update's motion.\n"
            << "Bottom-only versions are also in vertex_updates. Original intended arrows in\n"
            << "the parent directory remain fixed reference arrows for both playback sequences.\n"
            << "vertex_updates/updates.txt and the terminal print each processed vertex's\n"
            << "before/after position, target, accepted displacement, and remaining displacement.\n";
        int snapshots = 0;
        collision_colored_ccd_initial_guess(initial, displacement, mesh, params, options.iterations,
            [&](int iteration, const std::vector<Vec3>& positions, const std::vector<Vec3>& targets,
                const BroadPhase& broad_phase, const std::vector<std::vector<int>>& groups) {
                std::vector<int> colors(8, -1);
                for (std::size_t color = 0; color < groups.size(); ++color)
                    for (int vertex : groups[color]) colors[vertex] = static_cast<int>(color);
                if (iteration == 0) {
                    num_colors = groups.size();
                    write_update(0, -1, positions, targets, colors);
                    const auto& cache = broad_phase.cache();
                    checked_boxes(options.outdir / "blue_node_boxes.obj", cache.node_boxes);
                    checked_boxes(options.outdir / "green_triangle_boxes.obj", cache.tri_boxes);
                    checked_boxes(options.outdir / "green_edge_boxes.obj", cache.edge_boxes);
                    checked_boxes(options.outdir / "red_edge_boxes.obj", cache.red_edge_boxes);
                    write_arrows(options.outdir / "original_intended_displacement.obj", initial, targets);
                    write_arrows(options.outdir / "bottom_intended_displacement.obj", initial, targets, 4, 8);
                    write_geo(options.outdir / "targets.geo", targets, mesh.tris, initial,
                        displacement, targets, targets, colors, -1);
                    readme << "\nBroad phase: NT pairs=" << cache.nt_pairs.size() << ", SS pairs=" << cache.ss_pairs.size()
                        << ", collision colors=" << groups.size() << ".\n";
                    for (std::size_t color = 0; color < groups.size(); ++color) {
                        const Vec3 rgb = vertex_color(static_cast<int>(color));
                        readme << "Color " << color << " (RGB " << rgb.x() << ',' << rgb.y()
                            << ',' << rgb.z() << "):";
                        for (int vertex : groups[color]) readme << ' ' << vertex;
                        readme << '\n';
                    }
                }
                write_geo(options.outdir / numbered("mesh_sweep_", iteration, ".geo"), positions,
                    mesh.tris, initial, displacement, targets, previous, colors, iteration);
                write_geo(options.outdir / numbered("vertex_colors_", iteration, ".geo"), positions,
                    {}, initial, displacement, targets, previous, colors, iteration, true);
                write_arrows(options.outdir / numbered("remaining_displacement_", iteration, ".obj"), positions, targets);
                write_arrows(options.outdir / numbered("bottom_remaining_displacement_", iteration, ".obj"), positions, targets, 4, 8);
                write_arrows(options.outdir / numbered("bottom_accepted_displacement_", iteration, ".obj"), previous, positions, 4, 8);
                double max_remaining = 0.0;
                for (std::size_t i = 0; i < positions.size(); ++i)
                    max_remaining = std::max(max_remaining, (targets[i] - positions[i]).stableNorm());
                std::cout << "CCD sweep " << iteration << ": max remaining displacement=" << max_remaining << '\n';
                previous = positions;
                ++snapshots;
            },
            [&](int iteration, int color, const std::vector<Vec3>& positions,
                const std::vector<Vec3>& targets, const BroadPhase&,
                const std::vector<std::vector<int>>& groups) {
                if (update_index == std::numeric_limits<int>::max())
                    throw std::runtime_error("too many color-update snapshots");
                ++update_index;
                std::vector<int> colors(positions.size(), -1);
                for (std::size_t group = 0; group < groups.size(); ++group)
                    for (int vertex : groups[group]) colors[vertex] = static_cast<int>(group);
                for (int vertex : groups.at(static_cast<std::size_t>(color))) {
                    std::ostringstream line;
                    line << std::setprecision(17) << "update=" << update_index
                        << " iteration=" << iteration << " color=" << color
                        << " vertex=" << vertex << " layer=" << (vertex < 4 ? "top" : "bottom");
                    auto vector_value = [&](const char* name, const Vec3& value) {
                        line << ' ' << name << "=(" << value.x() << ',' << value.y() << ',' << value.z() << ')';
                    };
                    vector_value("before", previous_update[vertex]);
                    vector_value("after", positions[vertex]);
                    vector_value("target", targets[vertex]);
                    vector_value("accepted", positions[vertex] - previous_update[vertex]);
                    vector_value("remaining", targets[vertex] - positions[vertex]);
                    update_log << line.str() << '\n';
                    std::cout << line.str() << '\n';
                }
                write_update(iteration, color, positions, targets, colors);
            });
        if (static_cast<std::size_t>(update_index)
            != static_cast<std::size_t>(options.iterations) * num_colors)
            throw std::runtime_error("observer did not emit every color update");
        readme << "\nVertex-update playback range: 0.." << update_index
            << ". Colors per CCD sweep: " << num_colors << ".\n";
        update_log.close();
        readme.close();
        if (snapshots != options.iterations + 1)
            throw std::runtime_error("observer did not emit every requested sweep");
        std::cout << "Wrote " << snapshots << " snapshots to " << fs::absolute(options.outdir) << '\n';
        std::cout << "Also wrote initial state + " << update_index
            << " color updates to " << fs::absolute(updates_dir) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ccd_initial_guess_visualize: " << error.what() << '\n';
        return 1;
    }
}
