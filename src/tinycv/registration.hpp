#ifndef _TINYCV_REGISTRATION_HPP_
#define _TINYCV_REGISTRATION_HPP_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "bounding_box.h"
#include "derivative.h"
#include "histogram.h"
#include "interpolation.h"
#include "math.h"
#include "mutual_information.h"
#include "registration.h"
#include "sat.h"
#include "test.h"
#include "transform.h"


inline bool register_homography(const Mat& reference,
                                const Mat& tracked,
                                Mat& transform_homography)
{
    /*
    small_homog.for_each<Mat::Iterator<uint8_t>>(
        [](Mat::Iterator<uint8_t>& it, int y, int x, int c) {
            int qx = (int)(x < it.m.cols / 2 ? 0 : 255);
            int qy = (int)(y < it.m.rows / 2 ? 0 : 255);
            if (x > it.m.cols * 0.25 && x < it.m.cols * 0.75 &&
                y > it.m.rows * 0.25 && y < it.m.rows * 0.75) {
                it(y, x, c) = (uint8_t)(qx + qy > 255 ? 0 : qx + qy);
            }
        });
        */

    Mat reference_preprocessed;
    Mat tracked_preprocessed;

    preprocess_image<uint8_t>(reference, reference_preprocessed);
    preprocess_image<uint8_t>(tracked, tracked_preprocessed);

    /*
    generate_mi_space("mi.txt", reference_preprocessed);
    generate_mi_derivative_space(
        "dmi.txt", reference_preprocessed, reference_preprocessed);
    generate_mi_hessian_space(
        "d2mi.txt", reference_preprocessed, reference_preprocessed);
        */

    Mat initial_guess_affine;
    initial_guess_affine.create<float>(
        1, AffineTransform<float>::number_parameters, 1);
    initial_guess_affine << std::initializer_list<float>{0, 0, 0, 0, 0, 0};

    /*
    const float k = 35;
    Mat test_r = image_crop<float>(reference_preprocessed, BoundingBox{{k, k},
    {400, 250}}); Mat test_t = image_crop<float>(tracked_preprocessed,
    BoundingBox{{0, 0}, {400-k, 250-k}});
    */

    Mat transform_affine;
    register_impl<float, float, DerivativeNaive<1>, AffineTransform<float>>(
        reference_preprocessed,
        tracked_preprocessed,
        initial_guess_affine,
        transform_affine);
    Mat::ConstIterator<float> transform_affine_it(transform_affine);

    Mat initial_guess_homography;
    initial_guess_homography
        .create<float>(1, HomographyTransform<float>::number_parameters, 1)
        .fill<float>(0);

    Mat::Iterator<float> initial_guess_homog_it(initial_guess_homography);
    for (int i = 0; i < transform_affine.cols; ++i) {
        initial_guess_homog_it(0, i, 0) = transform_affine_it(0, i, 0);
    }

    register_impl<float, float, DerivativeNaive<1>, HomographyTransform<float>>(
        reference_preprocessed,
        tracked_preprocessed,
        initial_guess_homography,
        transform_homography);

    /*
    // clang-format off
    const float xy[]{0.0f, 0.0f};
    const float H[]{
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
        0.f, 0.f, 1.f,
    };
    // clang-format on

    homography_derivative(xy, H, dh);
    */

    const int gradient_border        = DerivativeNaive<1>::border_crop_size();
    const int preprocess_blur_border = 6;
    Mat crop_mat;
    Mat uncrop_mat;
    crop_mat.create<float>(1, 8, 1);
    uncrop_mat.create<float>(1, 8, 1);

    // clang-format off
    crop_mat << std::initializer_list<float>{
        0, 0, -2 * gradient_border - preprocess_blur_border,
        0, 0, -2 * gradient_border - preprocess_blur_border,
        0, 0
    };
    uncrop_mat << std::initializer_list<float>{
        0, 0, 2 * gradient_border + preprocess_blur_border,
        0, 0, 2 * gradient_border + preprocess_blur_border,
        0, 0
    };
    // clang-format on

    Mat transf_test;
    Mat transf_mask;
    image_transform<float,
                    1,
                    HomographyTransform<float>,
                    bilinear_sample<float, 1>>(
        tracked_preprocessed,
        transform_homography,
        BoundingBox(tracked_preprocessed),
        transf_test,
        transf_mask);

    HomographyTransform<float>::compose(
        transform_homography, crop_mat, transform_homography);
    HomographyTransform<float>::compose(
        uncrop_mat, transform_homography, transform_homography);

    image_transform<float,
                    1,
                    HomographyTransform<float>,
                    bilinear_sample<float, 1>>(
        tracked_preprocessed,
        transform_homography,
        BoundingBox(tracked_preprocessed),
        transf_test,
        transf_mask);

    HomographyTransform<float>::scale(
        static_cast<float>(tracked.cols) /
            static_cast<float>(tracked_preprocessed.cols +
                               2 * preprocess_blur_border),
        static_cast<float>(tracked.rows) /
            static_cast<float>(tracked_preprocessed.rows +
                               2 * preprocess_blur_border),
        transform_homography);
    return true;
}

#endif
