#ifndef _TINYCV_BOUNDING_BOX_H_
#define _TINYCV_BOUNDING_BOX_H_

#include <array>

#include "third_party/eigen/Eigen"

#include "mat.h"
#include "math.h"


namespace tinycv
{

struct BoundingBox
{
    BoundingBox();

    explicit BoundingBox(const BoundingBox& other);

    BoundingBox(BoundingBox&& other);

    BoundingBox(const std::initializer_list<Point<float>>& corners);

    BoundingBox(const Mat& image);

    BoundingBox& operator=(const BoundingBox& other);

    BoundingBox& operator=(BoundingBox&& other);

    int ceiling_width() const;

    int ceiling_height() const;

    Point<float> left_top;
    Point<float> right_bottom;
};

template <typename TransformClass>
BoundingBox bounding_box_transform(const BoundingBox& bb,
                                   const Mat& transform_parameters)
{
    // clang-format off
    const std::array<Point<float>, 4> image_corners{{
        {bb.left_top.x,     bb.left_top.y},
        {bb.right_bottom.x, bb.left_top.y},
        {bb.right_bottom.x, bb.right_bottom.y},
        {bb.left_top.x,     bb.right_bottom.y}
    }};

    BoundingBox output_bb{
        {std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()},
        {std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::lowest()}
    };
    // clang-format on

    for (size_t i = 0; i < image_corners.size(); ++i) {
        const Point<float> transformed_corner =
            TransformClass::transform(image_corners[i], transform_parameters);

        // Update bounding box
        const float* transformed_ptr = transformed_corner.cptr();
        for (int c = 0; c < 2; ++c) {
            output_bb.left_top.ptr()[c] =
                std::min(output_bb.left_top.ptr()[c], transformed_ptr[c]);
            output_bb.right_bottom.ptr()[c] =
                std::max(output_bb.right_bottom.ptr()[c], transformed_ptr[c]);
        }
    }

    return output_bb;
}

BoundingBox bounding_box_intersect(const BoundingBox& bb_a,
                                   const BoundingBox& bb_b);

template <typename PixelType>
Mat image_crop(const Mat& image, const BoundingBox& crop_bb)
{
    assert(image.type() == Mat::get_type_enum<PixelType>());

    Mat output;

    assert(crop_bb.left_top.x >= -1.f);
    assert(crop_bb.left_top.y >= -1.f);

    assert(crop_bb.left_top.x <= crop_bb.right_bottom.x);
    assert(crop_bb.left_top.y <= crop_bb.right_bottom.y);

    assert(crop_bb.right_bottom.x <= image.cols);
    assert(crop_bb.right_bottom.y <= image.rows);

    output.create_from_buffer<PixelType>(
        static_cast<PixelType*>(image.data) +
            static_cast<int>(crop_bb.left_top.y) * image.row_stride() +
            static_cast<int>(crop_bb.left_top.x) * image.channels(),
        crop_bb.ceiling_height(),
        crop_bb.ceiling_width(),
        image.channels(),
        image.row_stride());
    output.data_mgr_ = image.data_mgr_;

    return output;
}
}

#endif
