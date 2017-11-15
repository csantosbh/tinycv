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
