#include "example.h"
#include "chain.h"

#include <stdexcept>

// ------------------------------------------------------
// Utility helpers
// ------------------------------------------------------

static void set_uniform_velocity(Chain& chain, const Vec2& v){
    for (int i = 0; i < chain.N; ++i)
        set_xi(chain.v, i, v);
}

static void pin_node(Chain& chain, int i){
    chain.is_pinned[i] = 1;
    set_xi(chain.xpin, i, get_xi(chain.x, i));
}

// ------------------------------------------------------
// Scene builder
// ------------------------------------------------------

ExampleScene build_example(ExampleType example_type, int number_of_nodes, double density) {
    ExampleScene scene;
    std::vector<Chain> chains;

    switch (example_type) {

        // --------------------------------------------------
        // Example 1: two chains moving toward each other
        // --------------------------------------------------
        case ExampleType::Example1:
        {

            Chain chain1 = make_chain({-0.1, 3.0}, {-0.1, -3.0}, number_of_nodes, density);
            Chain chain2 = make_chain({ 0.1, 3.0}, { 0.1, -3.0}, number_of_nodes, density);

            pin_node(chain1, 0);
            pin_node(chain2, 0);

            set_uniform_velocity(chain1, {-6.0, 0.0});
            set_uniform_velocity(chain2, { 6.0, 0.0});

            chains.push_back(chain1);
            chains.push_back(chain2);
            break;
        }

        // --------------------------------------------------
        // Example 2: two pinned chains swinging into each other
        // --------------------------------------------------
        case ExampleType::Example2:
        {

            Chain upper = make_chain(
                    {-2.4, 2.8}, {1.2, 1.35}, number_of_nodes, density);
            Chain lower = make_chain(
                    {-1.85, 2.05}, {1.1, -2.9}, number_of_nodes, density);

            pin_node(upper, 0);
            pin_node(lower, 0);

            chains.push_back(upper);
            chains.push_back(lower);
            break;
        }

        default:
            throw std::runtime_error("Unknown ExampleType");
    }

    assemble_chains(chains, scene.state, scene.ref_mesh);
    return scene;
}
