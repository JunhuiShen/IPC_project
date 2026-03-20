#include "example.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

ExampleScene build_example(ExampleType example_type, int number_of_nodes, double mass_density) {
    ExampleScene scene;
    scene.total_frames = 120;

    switch (example_type) {

        // --------------------------------------------------
        // Example 1: two chains moving toward each other
        // --------------------------------------------------
        case ExampleType::Example1:
        {

            Chain chain1 = make_chain({-0.1, 3.0}, {-0.1, -3.0}, number_of_nodes, mass_density);
            Chain chain2 = make_chain({ 0.1, 3.0}, { 0.1, -3.0}, number_of_nodes, mass_density);

            pin_node(chain1, 0);
            pin_node(chain2, 0);

            set_uniform_velocity(chain1, {-6.0, 0.0});
            set_uniform_velocity(chain2, { 6.0, 0.0});

            scene.chains.push_back(chain1);
            scene.chains.push_back(chain2);
            break;
        }

        // --------------------------------------------------
        // Example 2: two chains falling onto ground
        // --------------------------------------------------
        case ExampleType::Example2:
        {

            Chain chain1 = make_chain({ 2.5, -0.5}, {-1.8,  1.7}, number_of_nodes, mass_density);
            Chain chain2 = make_chain({ 3.0,  0.3}, {-1.3,  2.5}, number_of_nodes, mass_density);

            Chain ground = make_chain({-3.0, -1.8}, { 3.0, -1.8}, 2, 1.0);

            pin_node(ground, 0);
            pin_node(ground, 1);

            set_uniform_velocity(chain1, {0.0, 0.0});
            set_uniform_velocity(chain2, {0.0, 0.0});
            set_uniform_velocity(ground, {0.0, 0.0});

            scene.chains.push_back(chain1);
            scene.chains.push_back(chain2);
            scene.chains.push_back(ground);
            break;
        }

        default:
            throw std::runtime_error("Unknown ExampleType");
    }

    return scene;
}