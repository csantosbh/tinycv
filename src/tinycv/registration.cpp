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

template <typename TransformClass>
bool NonLinearRegistration<TransformClass>::register_image(
    const Mat& tracked,
    const Mat& initial_guess_input,
    Mat& transform)
{
    Mat tracked_preprocessed;

    // Preprocess tracked image
    preprocess_image<uint8_t>(work_scale_,
                              preprocess_blur_border_,
                              preprocess_blur_std_,
                              tracked,
                              tracked_preprocessed);

    // Convert homography from input space to cropped/scaled preprocessed space
    Mat initial_guess_preprocessed;
    TransformClass::change_position(
        {-preprocess_blur_border_, -preprocess_blur_border_},
        initial_guess_input,
        initial_guess_preprocessed);

    const Point<float> input_to_work_scale{work_scale_, work_scale_};

    TransformClass::change_scale(input_to_work_scale,
                                 initial_guess_preprocessed,
                                 initial_guess_preprocessed);

    // Perform registration
    register_impl<PixelType,
                  GradPixelType,
                  DerivativeMethod,
                  TransformClass,
                  BinningMethod>(reference_preprocessed_,
                                 tracked_preprocessed,
                                 initial_guess_preprocessed,
                                 steepest_gradient_r_,
                                 mi_hessian_,
                                 number_max_iterations_,
                                 transform);

    // Convert homography from cropped/scaled preprocessed space to input space
    TransformClass::change_position(
        {preprocess_blur_border_, preprocess_blur_border_},
        transform,
        transform);

    const Point<float> work_to_input_scale{1.f / work_scale_,
                                           1.f / work_scale_};

    TransformClass::change_scale(work_to_input_scale, transform, transform);

    return true;
}

template <typename TransformClass>
void NonLinearRegistration<TransformClass>::set_reference(const Mat& reference)
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
    generate_steepest_gradient<GradPixelType, TransformClass>(
        cropped_grad_x, cropped_grad_y, steepest_gradient_r_);

    // Generate Mutual Information Hessian of reference image with itself at
    // origin
    generate_self_ic_hessian<PixelType,
                             GradPixelType,
                             BinningMethod,
                             TransformClass,
                             DerivativeMethod>(reference_preprocessed_,
                                               grad_x_reference,
                                               grad_y_reference,
                                               steepest_gradient_r_,
                                               mi_hessian_);
}
}
