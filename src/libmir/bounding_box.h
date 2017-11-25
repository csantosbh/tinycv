#ifndef _LIBMIR_BOUNDING_BOX_H_
#define _LIBMIR_BOUNDING_BOX_H_

#include <array>

#include <Eigen/Eigen>

#include "mat.h"
#include "math.h"

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

template <typename TransformClass>
BoundingBox bounding_box_transform(const BoundingBox& bb,
                                   const Mat& transform_parameters)
{
    // clang-format off
    const std::array<Point<float>, 4> image_corners{{
        {bb.left_top[0],     bb.left_top[1]},
        {bb.right_bottom[0], bb.left_top[1]},
        {bb.right_bottom[0], bb.right_bottom[1]},
        {bb.left_top[0],     bb.right_bottom[1]}
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
        const float* transformed_ptr = transformed_corner.ptr();
        for (int c = 0; c < 2; ++c) {
            output_bb.left_top[c] =
                std::min(output_bb.left_top[c], transformed_ptr[c]);
            output_bb.right_bottom[c] =
                std::max(output_bb.right_bottom[c], transformed_ptr[c]);
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

    assert(crop_bb.left_top[0] >= -1.f);
    assert(crop_bb.left_top[1] >= -1.f);

    assert(crop_bb.left_top[0] <= crop_bb.right_bottom[0]);
    assert(crop_bb.left_top[1] <= crop_bb.right_bottom[1]);

    assert(crop_bb.right_bottom[0] <= image.cols);
    assert(crop_bb.right_bottom[1] <= image.rows);

    output.create_from_buffer<PixelType>(
        static_cast<PixelType*>(image.data) +
            static_cast<int>(crop_bb.left_top[1]) * image.row_stride() +
            static_cast<int>(crop_bb.left_top[0]) * image.channels(),
        crop_bb.ceiling_height(),
        crop_bb.ceiling_width(),
        image.channels(),
        image.row_stride());
    output.data_mgr_ = image.data_mgr_;

    return output;
}

#endif
