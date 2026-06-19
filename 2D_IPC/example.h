#pragma once

#include "physics.h"

enum class ExampleType {
    Example1
};

struct ExampleScene {
    DeformedState state;
    RefMesh ref_mesh;
    std::vector<Pin> pins;
    std::vector<GroundSDF> sdf_grounds;
    std::vector<CircleSDF> sdf_circles;
    Vec static_positions;
    std::vector<std::pair<int, int>> static_edges;
};

ExampleScene build_example(ExampleType example_type, int number_of_nodes, double density, double thickness);
