#include "make_shape.h"
#include <gtest/gtest.h>

TEST(BuildIncidentTriangleMap, BasicExample) {
    // [0,1,2, 1,2,5] — two triangles
    std::vector<int> indices = {0, 1, 2, 1, 2, 5};
    auto map = build_incident_triangle_map(indices);

    EXPECT_EQ(map[0], (std::vector<int>{0}));
    EXPECT_EQ(map[1], (std::vector<int>{1, 3}));
    EXPECT_EQ(map[2], (std::vector<int>{2, 4}));
    EXPECT_EQ(map[5], (std::vector<int>{5}));
    EXPECT_EQ(map.size(), 4u);
}

TEST(BuildIncidentTriangleMap, EmptyInput) {
    std::vector<int> indices = {};
    auto map = build_incident_triangle_map(indices);
    EXPECT_TRUE(map.empty());
}

TEST(BuildIncidentTriangleMap, SingleTriangle) {
    std::vector<int> indices = {0, 1, 2};
    auto map = build_incident_triangle_map(indices);

    EXPECT_EQ(map[0], (std::vector<int>{0}));
    EXPECT_EQ(map[1], (std::vector<int>{1}));
    EXPECT_EQ(map[2], (std::vector<int>{2}));
    EXPECT_EQ(map.size(), 3u);
}
