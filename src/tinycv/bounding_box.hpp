#ifndef _TINYCV_BOUNDING_BOX_HPP_
#define _TINYCV_BOUNDING_BOX_HPP_

#include "bounding_box.h"
#include "mat.h"

using Matrix3fRowMajor = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

BoundingBox::BoundingBox()
{
}

BoundingBox::BoundingBox(const BoundingBox& other)
    : left_top(other.left_top)
    , right_bottom(other.right_bottom)
{
}

BoundingBox::BoundingBox(BoundingBox&& other)
    : left_top(other.left_top)
    , right_bottom(other.right_bottom)
{
}

BoundingBox::BoundingBox(
    const std::initializer_list<std::array<float, 2>>& corners)
    : left_top(*corners.begin())
    , right_bottom(*(corners.end() - 1))
{
}

BoundingBox::BoundingBox(const Mat& image)
    : left_top({0, 0})
    , right_bottom({static_cast<float>(image.cols - 1),
                    static_cast<float>(image.rows - 1)})
{
}

BoundingBox& BoundingBox::operator=(BoundingBox&& other)
{
    left_top     = other.left_top;
    right_bottom = other.right_bottom;

    return *this;
}

BoundingBox& BoundingBox::operator=(const BoundingBox& other)
{
    left_top     = other.left_top;
    right_bottom = other.right_bottom;

    return *this;
}

int BoundingBox::ceiling_width() const
{
    return static_cast<int>(std::ceil(right_bottom[0] - left_top[0] + 1.f));
}

int BoundingBox::ceiling_height() const
{
    return static_cast<int>(std::ceil(right_bottom[1] - left_top[1] + 1.f));
}

BoundingBox bounding_box_intersect(const BoundingBox& bb_a,
                                   const BoundingBox& bb_b)
{
    using std::max;
    using std::min;

    return BoundingBox({{max(bb_a.left_top[0], bb_b.left_top[0]),
                         max(bb_a.left_top[1], bb_b.left_top[1])},
                        {min(bb_a.right_bottom[0], bb_b.right_bottom[0]),
                         min(bb_a.right_bottom[1], bb_b.right_bottom[1])}});
}

#endif
