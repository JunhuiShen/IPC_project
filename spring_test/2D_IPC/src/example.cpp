#include "example.h"
#include <stdexcept>

ExampleScene build_example(ExampleType example_type, int number_of_nodes) {
    ExampleScene scene;

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
    else {
        throw std::runtime_error("Unknown ExampleType");
    }

    return scene;
}