#include "bounding_box.h"
#include "mat.h"

namespace tinycv
{

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

BoundingBox::BoundingBox(const std::initializer_list<Point<float>>& corners)
    : left_top(*corners.begin())
    , right_bottom(*(corners.end() - 1))
{
    // This method expects two corners: The left-top and the right-bottom
    assert(corners.size() == 2);
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
    return static_cast<int>(std::ceil(right_bottom.x - left_top.x + 1.f));
}

int BoundingBox::ceiling_height() const
{
    return static_cast<int>(std::ceil(right_bottom.y - left_top.y + 1.f));
}

BoundingBox bounding_box_intersect(const BoundingBox& bb_a,
                                   const BoundingBox& bb_b)
{
    using std::max;
    using std::min;

    return BoundingBox({{max(bb_a.left_top.x, bb_b.left_top.x),
                         max(bb_a.left_top.y, bb_b.left_top.y)},
                        {min(bb_a.right_bottom.x, bb_b.right_bottom.x),
                         min(bb_a.right_bottom.y, bb_b.right_bottom.y)}});
}
}
