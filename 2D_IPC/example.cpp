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
            append_rigid_polygon(int(6), scene.state, scene.ref_mesh, {double(0),double(10)}, double(.5), density, thickness, {double(0),double(0)}, double(0),double(1));
            scene.sdf_grounds.push_back({double(0)});
            scene.static_positions = {{double(-4), double(0)}, {double(4), double(0)}};
            scene.static_edges = {{0, 1}};
            break;
        }
        // Command line: ./build/simulation --example 3 --num_frames 100 --outdir frames_hexagons_collide --eps_sdf .001 --max_substep_iters 5000 --substeps 25 --gy 0
        case ExampleType::Example3: {
            append_rigid_polygon(6, scene.state, scene.ref_mesh, {-5.0, 0.0}, 0.5, density, thickness, {3.0, 0.0}, 0.0, 4.0);
            append_rigid_polygon(6, scene.state, scene.ref_mesh, {5.0, 0.0}, 0.5, density, thickness, {-3.0, 0.0}, 0.0, 4.0);
            break;
        }
        // Command line: ./build/simulation --example 4 --num_frames 300 --outdir frames_box --max_substep_iters 5000 --substeps 50
        case ExampleType::Example4: {
            // Open-top box: ground at y=0, left wall at x=-4, right wall at x=4
            scene.sdf_grounds.push_back({0.0});
            scene.sdf_planes.push_back({{ 1.0, 0.0}, -4.0}); // left wall:  phi = x.x - (-4) = x.x + 4, > 0 when x.x > -4
            scene.sdf_planes.push_back({{-1.0, 0.0}, -4.0}); // right wall: phi = -x.x - (-4) = 4 - x.x, > 0 when x.x < 4

            // Visualize the box outline
            scene.static_positions = {
                {-4.0, 0.0}, {4.0, 0.0},   // bottom ground
                {-4.0, 0.0}, {-4.0, 8.0},  // left wall
                { 4.0, 0.0}, { 4.0, 8.0},  // right wall
            };
            scene.static_edges = {{0,1}, {2,3}, {4,5}};

            // 200 polygons in a 10x20 grid, cycling through shapes 3-8 sides
            constexpr int cols = 10, rows = 15;
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    const int    idx   = row * cols + col;
                    const int    sides = 3 + (idx % 6);              // cycles 3,4,5,6,7,8
                    const double x     = -3.6 + col * 0.8;           // evenly across [-3.6, 3.6]
                    const double y     = 6.0  + row * 1.0;           // stack upward from y=6
                    const double theta = (idx % 7) * (M_PI / 7.0);   // varied initial orientation
                    append_rigid_polygon(sides, scene.state, scene.ref_mesh, {x, y}, 0.3, density, thickness, {0.0, 0.0}, theta, 0.0);
                }
            }
            break;
        }
        default:
            throw std::invalid_argument("Unknown example type");
    }
    return scene;
}
