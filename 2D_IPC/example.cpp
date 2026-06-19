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
            Chain upper =
                make_chain({-2.4, 2.8}, {1.2, 1.35}, number_of_nodes, density, thickness);
            Chain lower =
                make_chain({-1.85, 2.05}, {1.1, -2.9}, number_of_nodes, density, thickness);
            pin_node(upper, 0);
            pin_node(lower, 0);
            assemble_chains({upper, lower}, scene.state, scene.ref_mesh, scene.pins);
            break;
        }
        default:
            throw std::invalid_argument("Unknown example type");
    }
    return scene;
}
