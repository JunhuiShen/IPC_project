#pragma once

#include "chain.h"
#include <vector>

enum class ExampleType {
    Example1,
    Example2,
    Example3
};

struct ExampleScene {
    std::vector<Chain> chains;
    int total_frames = 60;
};

ExampleScene build_example(ExampleType example_type, int number_of_nodes);
