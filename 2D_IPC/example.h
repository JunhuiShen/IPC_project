#pragma once

#include "mesh.h"
#include "state.h"

enum class ExampleType {
    Example1,
    Example2
};

struct ExampleScene {
    State2D state;
    RefMesh ref_mesh;
};

ExampleScene build_example(ExampleType example_type, int number_of_nodes, double density);
