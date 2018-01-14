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

Mat escalada[2];
inline bool register_homography(const Mat& reference,
                                const Mat& tracked,
                                Mat& transform_homography)
{
    Mat reference_preprocessed;
    Mat tracked_preprocessed;

    const float work_scale           = 0.3f;
    const int preprocess_blur_border = 6;
    const float preprocess_blur_std  = 2.0;

    preprocess_image<uint8_t>(work_scale,
                              preprocess_blur_border,
                              preprocess_blur_std,
                              reference,
                              reference_preprocessed);

    preprocess_image<uint8_t>(work_scale,
                              preprocess_blur_border,
                              preprocess_blur_std,
                              tracked,
                              tracked_preprocessed);

    Mat initial_guess_affine;
    initial_guess_affine.create<float>(
        1, AffineTransform<float>::number_parameters, 1);
    initial_guess_affine << std::initializer_list<float>{0, 0, 0, 0, 0, 0};

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



    // tst0
    Mat msk, prep_alinhada;

    image_transform<float,
                    1,
                    HomographyTransform<float>,
                    bilinear_sample<float, 1>>(tracked_preprocessed,
                                               transform_homography,
                                               BoundingBox(reference_preprocessed),
                                               prep_alinhada,
                                               msk);
    // tst0





    HomographyTransform<float>::change_position(
        {static_cast<float>(preprocess_blur_border),
         static_cast<float>(preprocess_blur_border)},
        transform_homography,
        transform_homography);


    // tst
    {


        Mat referencia=escalada[0];
        Mat alinhada;
        image_transform<uint8_t,
                        1,
                        HomographyTransform<float>,
                        bilinear_sample<uint8_t, 1>>(escalada[1],
                                                   transform_homography,
                                                   BoundingBox(referencia),
                                                   alinhada,
                                                   msk);
        int breakpoint;
        breakpoint=12;
    }


    const Point<float> work_scale_to_full_scale{1.f / work_scale,
                                                1.f / work_scale};

    HomographyTransform<float>::change_scale(
        work_scale_to_full_scale, transform_homography, transform_homography);
    return true;
}

#endif
