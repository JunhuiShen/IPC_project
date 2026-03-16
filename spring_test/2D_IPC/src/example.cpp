#include "example.h"
#include <stdexcept>

ExampleScene build_example(ExampleType example_type, int number_of_nodes) {
    ExampleScene scene;

    // Example 1: two vertical chains moving toward each other
    if (example_type == ExampleType::Example1) {
        scene.total_frames = 150;

        Chain chain1 = make_chain({-0.1,  1.5}, {-0.1, -1.5}, number_of_nodes, 0.05);
        Chain chain2 = make_chain({ 0.1,  1.5}, { 0.1, -1.5}, number_of_nodes, 0.05);

        chain1.is_pinned[0] = 1;
        chain2.is_pinned[0] = 1;

        set_xi(chain1.xpin, 0, get_xi(chain1.x, 0));
        set_xi(chain2.xpin, 0, get_xi(chain2.x, 0));

        for (int i = 0; i < chain1.N; ++i) {
            set_xi(chain1.v, i, {-6.0, 0.0});
        }

        for (int i = 0; i < chain2.N; ++i) {
            set_xi(chain2.v, i, {6.0, 0.0});
        }

        scene.chains.push_back(chain1);
        scene.chains.push_back(chain2);
    }
    // Example 2: multiple chains fall onto a pinned ground segment
    else if (example_type == ExampleType::Example2) {
        scene.total_frames = 60;

        Chain chain1 = make_chain({-0.8, 1.2}, { 1.6, 0.0}, number_of_nodes, 0.05);
        Chain chain2 = make_chain({-0.4, 2.0}, { 2.0, 0.8}, number_of_nodes, 0.05);
        Chain chain3 = make_chain({ 0.0, 2.8}, { 2.4, 1.6}, number_of_nodes, 0.05);
        Chain ground = make_chain({-2.0, -1.8}, { 2.0, -1.8}, 2, 1.0);

        ground.is_pinned[0] = 1;
        ground.is_pinned[1] = 1;

        set_xi(ground.xpin, 0, get_xi(ground.x, 0));
        set_xi(ground.xpin, 1, get_xi(ground.x, 1));

        for (int i = 0; i < chain1.N; ++i) {
            set_xi(chain1.v, i, {0.0, 0.0});
        }

        for (int i = 0; i < chain2.N; ++i) {
            set_xi(chain2.v, i, {0.0, 0.0});
        }

        for (int i = 0; i < chain3.N; ++i) {
            set_xi(chain3.v, i, {0.0, 0.0});
        }

        for (int i = 0; i < ground.N; ++i) {
            set_xi(ground.v, i, {0.0, 0.0});
        }

        scene.chains.push_back(chain1);
        scene.chains.push_back(chain2);
        scene.chains.push_back(chain3);
        scene.chains.push_back(ground);
    }
    // Example 3: an upper free chain falls onto a lower pinned chain
    else if (example_type == ExampleType::Example3) {
            total_frame = 60;

            Chain lower = chain_model::make_chain({-1.2, 0.2}, {1.2, 0.0}, number_of_nodes, 0.05);
            Chain upper = chain_model::make_chain({-0.4, 1.8}, {1.6, 1.0}, number_of_nodes, 0.05);

            // Pin both ends of the lower chain so it acts like a suspended obstacle
            lower.is_pinned[0] = 1;
            lower.is_pinned[lower.N - 1] = 1;

            set_xi(lower.xpin, 0, get_xi(lower.x, 0));
            set_xi(lower.xpin, lower.N - 1, get_xi(lower.x, lower.N - 1));

            // Initial velocities
            for (int i = 0; i < lower.N; ++i)
                set_xi(lower.v, i, {0.0, 0.0});

            for (int i = 0; i < upper.N; ++i)
                set_xi(upper.v, i, {0.0, 0.0});

            chains.push_back(lower);
            chains.push_back(upper);
        }  
    else {
        throw std::runtime_error("Unknown ExampleType");
    }

    return scene;
}
