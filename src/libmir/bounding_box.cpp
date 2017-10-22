#include <Eigen/Eigen>

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

int BoundingBox::flooring_width() const
{
    return static_cast<int>(right_bottom[0] - left_top[0]) + 1;
}

int BoundingBox::flooring_height() const
{
    return static_cast<int>(right_bottom[1] - left_top[1]) + 1;
}

BoundingBox bounding_box_transform(const BoundingBox& bb,
                                   const float* homography_ptr)
{
    using Eigen::Vector3f;
    using std::vector;

    Eigen::Map<const Matrix3fRowMajor> homography(homography_ptr);

    // clang-format off
    const vector<Vector3f> image_corner{
        {bb.left_top[0],     bb.left_top[1],     1.f},
        {bb.right_bottom[0], bb.left_top[1],     1.f},
        {bb.right_bottom[0], bb.right_bottom[1], 1.f},
        {bb.left_top[0],     bb.right_bottom[1], 1.f}
    };

    BoundingBox output_bb{
        {std::numeric_limits<float>::max(),
         std::numeric_limits<float>::max()},
        {std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::lowest()}
    };
    // clang-format on

    vector<Vector3f> transformed_corners(4);
    for (size_t i = 0; i < image_corner.size(); ++i) {
        transformed_corners[i] = homography * image_corner[i];
        transformed_corners[i] /= transformed_corners[i][2];

        // Update bounding box
        for (int c = 0; c < 2; ++c) {
            output_bb.left_top[c] =
                std::min(output_bb.left_top[c], transformed_corners[i][c]);
            output_bb.right_bottom[c] =
                std::max(output_bb.right_bottom[c], transformed_corners[i][c]);
        }
    }

    return output_bb;
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
