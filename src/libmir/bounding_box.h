#ifndef _LIBMIR_BOUNDING_BOX_H_
#define _LIBMIR_BOUNDING_BOX_H_

#include <array>

#include "mat.h"

struct BoundingBox
{
    BoundingBox();

    explicit BoundingBox(const BoundingBox& other);

    BoundingBox(BoundingBox&& other);

    BoundingBox(const std::initializer_list<std::array<float, 2>>& corners);

    BoundingBox(const Mat& image);

    BoundingBox& operator=(const BoundingBox& other);

    BoundingBox& operator=(BoundingBox&& other);

    int ceiling_width() const;

    int ceiling_height() const;

    std::array<float, 2> left_top;
    std::array<float, 2> right_bottom;
};

BoundingBox bounding_box_transform(const BoundingBox& bb,
                                   const float* homography_ptr);

BoundingBox bounding_box_intersect(const BoundingBox& bb_a,
                                   const BoundingBox& bb_b);

#endif
