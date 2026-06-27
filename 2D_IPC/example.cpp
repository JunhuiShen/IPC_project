#include "example.h"
#include "make_shape.h"

#include <stdexcept>

static void pin_node(Chain& chain, int i) {
    chain.pins.push_back({i, get_xi(chain.deformed_positions, i)});
}

ExampleScene build_example(
    ExampleType example_type, int number_of_nodes, double density, double thickness) {
    ExampleScene scene;
    switch (example_type) {
        case ExampleType::Example1: {
            Chain upper = make_chain({-2.4, 2.8}, {1.2, 1.35}, number_of_nodes, density, thickness);
            Chain lower = make_chain({-1.85, 2.05}, {1.1, -2.9}, number_of_nodes, density, thickness);
            pin_node(upper, 0);
            pin_node(lower, 0);
            assemble_chains({upper, lower}, scene.state, scene.ref_mesh, scene.pins);
            break;
        }
        // Command line: ./build/simulation --example 2 --num_frames 500 --outdir frames_hexagon
        case ExampleType::Example2: {
            append_rigid_polygon(int(6), scene.state, scene.ref_mesh, {double(0),double(10)}, double(.5),double(1), double(0.001), {double(0),double(0)}, double(0),double(1));
            scene.sdf_grounds.push_back({double(0)});
            scene.static_positions = {{double(-4), double(0)}, {double(4), double(0)}};
            scene.static_edges = {{0, 1}};
            break;
        }
        // Command line: ./build/simulation --example 3 --num_frames 500 --outdir frames_two_hex
        case ExampleType::Example3: {
            append_rigid_polygon(6, scene.state, scene.ref_mesh, {-5.0, 0.0}, 0.5, 1.0, 0.001, {3.0, 0.0}, 0.0, 0.0);
            append_rigid_polygon(6, scene.state, scene.ref_mesh, {5.0, 0.0}, 0.5, 1.0, 0.001, {-3.0, 0.0}, 0.0, 0.0);
            break;
        }
        default:
            throw std::invalid_argument("Unknown example type");
    }
    return scene;
}
