#include "example.h"
#include <stdexcept>
#include <cmath>
#include <utility>
#include <algorithm>

// Stretch a segment about its midpoint by a factor s
std::pair<Vec2, Vec2> stretch_about_midpoint(const Vec2& a, const Vec2& b, double s) {
    Vec2 mid{0.5 * (a.x + b.x), 0.5 * (a.y + b.y)};
    Vec2 half{0.5 * (b.x - a.x), 0.5 * (b.y - a.y)};
    half.x *= s;
    half.y *= s;
    return {
            Vec2{mid.x - half.x, mid.y - half.y},
            Vec2{mid.x + half.x, mid.y + half.y}
    };
}

static void set_uniform_velocity(Chain& chain, const Vec2& v){
    for (int i = 0; i < chain.N; ++i)
        set_xi(chain.v, i, v);
}

static void pin_node(Chain& chain, int i){
    chain.is_pinned[i] = 1;
    set_xi(chain.xpin, i, get_xi(chain.x, i));
}

ExampleScene build_example(ExampleType example_type, int number_of_nodes){
    ExampleScene scene;
    scene.total_frames = 60;

    const double stretch = std::sqrt(std::max(1.0, double(number_of_nodes - 1) / 10.0));

    switch (example_type) {

        // Example 1: two chains moving toward each other
        case ExampleType::Example1:{
            auto [a1, b1] = stretch_about_midpoint({-0.1, 1.5}, {-0.1, -1.5}, stretch);
            auto [a2, b2] = stretch_about_midpoint({ 0.1, 1.5}, { 0.1, -1.5}, stretch);

            Chain chain1 = make_chain(a1, b1, number_of_nodes, 0.05);
            Chain chain2 = make_chain(a2, b2, number_of_nodes, 0.05);

            pin_node(chain1, 0);
            pin_node(chain2, 0);

            set_uniform_velocity(chain1, {-2.0, 0.0});
            set_uniform_velocity(chain2, { 2.0, 0.0});

            scene.chains.push_back(chain1);
            scene.chains.push_back(chain2);
            break;
        }

        // Example 2: two chains falling onto ground
        case ExampleType::Example2:{
            auto [c1a, c1b] = stretch_about_midpoint({-0.8, 1.2}, {1.6, 0.0}, stretch);
            auto [c2a, c2b] = stretch_about_midpoint({ 0.0, 2.8}, {2.4, 1.6}, stretch);

            Chain chain1 = make_chain(c1a, c1b, number_of_nodes, 0.05);
            Chain chain2 = make_chain(c2a, c2b, number_of_nodes, 0.05);

            double ground_scale = std::max(1.0, 0.8 * stretch);
            auto [ga, gb] = stretch_about_midpoint({-2.0, -1.8}, {2.0, -1.8}, ground_scale);

            Chain ground = make_chain(ga, gb, 2, 1.0);

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
        }

        default:
            throw std::runtime_error("Unknown ExampleType");
    }

    return scene;
}
