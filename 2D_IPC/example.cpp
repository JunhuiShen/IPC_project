#include "example.h"
#include "chain.h"

#include <stdexcept>

static void pin_node(Chain& chain, int i) {
    chain.is_pinned[i] = 1;
    set_xi(chain.xpin, i, get_xi(chain.x, i));
}

ExampleScene build_example(
    ExampleType example_type, int number_of_nodes, double density) {
    ExampleScene scene;
    switch (example_type) {
        case ExampleType::Example1: {
            Chain upper =
                make_chain({-2.4, 2.8}, {1.2, 1.35}, number_of_nodes, density);
            Chain lower =
                make_chain({-1.85, 2.05}, {1.1, -2.9}, number_of_nodes, density);
            pin_node(upper, 0);
            pin_node(lower, 0);
            assemble_chains({upper, lower}, scene.state, scene.ref_mesh);
            break;
        }
        default:
            throw std::invalid_argument("Unknown example type");
    }
    return scene;
}
