#pragma once

#include "physics.h"

enum class ExampleType {
    Example1
};

struct ExampleScene {
    DeformedState state;
    RefMesh ref_mesh;
    std::vector<Pin> pins;
};

ExampleScene build_example(
    ExampleType example_type, int number_of_nodes, double density, double thickness);
