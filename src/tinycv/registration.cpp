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


namespace tinycv
{

using DerivativeMethod = DerivativeNaive<1>;
using PixelType        = float;
using GradPixelType    = float;
using BinningMethod    = BSpline4;

// TODO use initial guess.. how to convert it from homography to affine while
// preserving as much fidelity as possible?
bool NonLinearRegistration::register_homography(const Mat& tracked,
                                               const Mat& initial_guess,
                                               Mat& transform_homography)
{
    Mat tracked_preprocessed;

    preprocess_image<uint8_t>(work_scale_,
                              preprocess_blur_border_,
                              preprocess_blur_std_,
                              tracked,
                              tracked_preprocessed);

    Mat initial_guess_affine;
    initial_guess_affine.create<float>(
        1, AffineTransform<float>::number_parameters, 1);
    initial_guess_affine << std::initializer_list<float>{0, 0, 0, 0, 0, 0};

    Mat transform_affine;
    register_impl<PixelType,
                  GradPixelType,
                  DerivativeMethod,
                  AffineTransform<float>,
                  BinningMethod>(reference_preprocessed_,
                                 tracked_preprocessed,
                                 initial_guess_affine,
                                 steepest_gradient_r_["affine"],
                                 mi_hessian_["affine"],
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

    register_impl<PixelType,
                  GradPixelType,
                  DerivativeMethod,
                  HomographyTransform<float>,
                  BinningMethod>(reference_preprocessed_,
                                 tracked_preprocessed,
                                 initial_guess_homography,
                                 steepest_gradient_r_["homography"],
                                 mi_hessian_["homography"],
                                 transform_homography);

    HomographyTransform<float>::change_position(
        {preprocess_blur_border_, preprocess_blur_border_},
        transform_homography,
        transform_homography);

    const Point<float> work_to_input_scale{1.f / work_scale_,
                                           1.f / work_scale_};

    HomographyTransform<float>::change_scale(
        work_to_input_scale, transform_homography, transform_homography);

    return true;
}

void NonLinearRegistration::set_reference(const Mat& reference)
{
    // Generate scaled reference image
    preprocess_image<uint8_t>(work_scale_,
                              preprocess_blur_border_,
                              preprocess_blur_std_,
                              reference,
                              reference_preprocessed_);

    // Declare gradient images
    Mat grad_x_reference;
    Mat grad_y_reference;

    // Generate gradients
    DerivativeMethod::template derivative<PixelType, GradPixelType>(
        reference_preprocessed_, ImageDerivativeAxis::dX, grad_x_reference);

    DerivativeMethod::template derivative<PixelType, GradPixelType>(
        reference_preprocessed_, ImageDerivativeAxis::dY, grad_y_reference);

    ///
    // Crop image borders to match their sizes with the second derivative
    const int gradient_border = DerivativeMethod::border_crop_size();

    BoundingBox border_bb_1{
        {{gradient_border, gradient_border}},
        {{static_cast<float>(grad_x_reference.cols - gradient_border - 1),
          static_cast<float>(grad_x_reference.rows - gradient_border - 1)}}};

    Mat cropped_grad_x =
        image_crop<GradPixelType>(grad_x_reference, border_bb_1);
    Mat cropped_grad_y =
        image_crop<GradPixelType>(grad_y_reference, border_bb_1);

    // Generate steepest gradient images
    generate_steepest_gradient<GradPixelType, AffineTransform<float>>(
        cropped_grad_x, cropped_grad_y, steepest_gradient_r_["affine"]);

    generate_steepest_gradient<GradPixelType, HomographyTransform<float>>(
        cropped_grad_x, cropped_grad_y, steepest_gradient_r_["homography"]);

    // Generate Mutual Information Hessian of reference image with itself at
    // origin
    generate_self_ic_hessian<PixelType,
                             GradPixelType,
                             BinningMethod,
                             AffineTransform<float>,
                             DerivativeMethod>(reference_preprocessed_,
                                               grad_x_reference,
                                               grad_y_reference,
                                               steepest_gradient_r_["affine"],
                                               mi_hessian_["affine"]);

    generate_self_ic_hessian<PixelType,
                             GradPixelType,
                             BinningMethod,
                             HomographyTransform<float>,
                             DerivativeMethod>(
        reference_preprocessed_,
        grad_x_reference,
        grad_y_reference,
        steepest_gradient_r_["homography"],
        mi_hessian_["homography"]);
}
}
