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
    preprocess_image<uint8_t>(
        register_pyr_levels_["homography"].work_scale,
        register_pyr_levels_["homography"].preprocess_blur_border,
        register_pyr_levels_["homography"].preprocess_blur_std,
        tracked,
        tracked_preprocessed);

    // Convert homography from input space to cropped/scaled preprocessed space
    Mat initial_guess_preprocessed;
    TransformClass::change_position(
        {-register_pyr_levels_["homography"].preprocess_blur_border,
         -register_pyr_levels_["homography"].preprocess_blur_border},
        initial_guess_input,
        initial_guess_preprocessed);

    const Point<float> input_to_work_scale{
        register_pyr_levels_["homography"].work_scale,
        register_pyr_levels_["homography"].work_scale};

    TransformClass::change_scale(input_to_work_scale,
                                 initial_guess_preprocessed,
                                 initial_guess_preprocessed);

    /////////// Affine transform
    // Transform tracked image by initial guess
    Mat tracked_pre_transformed;
    Mat mask_pre_transformed;
    BoundingBox bb_pre_transformed = bounding_box_transform<TransformClass>(
        BoundingBox(tracked_preprocessed), initial_guess_preprocessed);

    image_transform<PixelType,
                    1,
                    TransformClass,
                    bilinear_sample<PixelType, 1>>(tracked_preprocessed,
                                                   initial_guess_preprocessed,
                                                   bb_pre_transformed,
                                                   tracked_pre_transformed,
                                                   mask_pre_transformed);

    Mat affine_initial_guess;
    TranslationTransform<float>::identity(affine_initial_guess);

    // Include crop term to initial affine transform
    TranslationTransform<float>::compose(bb_pre_transformed.left_top,
                                    affine_initial_guess,
                                    affine_initial_guess);
    Mat affine_transform;

    register_impl<PixelType,
                  GradPixelType,
                  DerivativeMethod,
                  TranslationTransform<float>,
                  BinningMethod>(
        register_pyr_levels_["affine"].reference_preprocessed,
        tracked_pre_transformed,
        affine_initial_guess,
        register_pyr_levels_["affine"].steepest_gradient_r,
        register_pyr_levels_["affine"].mi_hessian,
        register_pyr_levels_["affine"].number_max_iterations,
        affine_transform);

    // Remove crop term from output affine transform
    TranslationTransform<float>::compose(-bb_pre_transformed.left_top,
                                    affine_transform,
                                    affine_transform);

    // Update homography initial guess
    Mat affine_to_homog;
    TranslationTransform<float>::to_homography(affine_transform,
                                          affine_to_homog);
    HomographyTransform<float>::compose(affine_to_homog,
                                        initial_guess_preprocessed,
                                        initial_guess_preprocessed);

    /////////// Homography transform
    // Perform registration
    register_impl<PixelType,
                  GradPixelType,
                  DerivativeMethod,
                  TransformClass,
                  BinningMethod>(
        register_pyr_levels_["homography"].reference_preprocessed,
        tracked_preprocessed,
        initial_guess_preprocessed,
        register_pyr_levels_["homography"].steepest_gradient_r,
        register_pyr_levels_["homography"].mi_hessian,
        register_pyr_levels_["homography"].number_max_iterations,
        transform);

    // Convert homography from cropped/scaled preprocessed space to input space
    TransformClass::change_position(
        {register_pyr_levels_["homography"].preprocess_blur_border,
         register_pyr_levels_["homography"].preprocess_blur_border},
        transform,
        transform);

    const Point<float> work_to_input_scale{
        1.f / register_pyr_levels_["homography"].work_scale,
        1.f / register_pyr_levels_["homography"].work_scale};

    TransformClass::change_scale(work_to_input_scale, transform, transform);

    return true;
}

/**
 * <reference> can be released after this function has been called.
 */
template <typename TransformClass>
void NonLinearRegistration<TransformClass>::set_reference(const Mat& reference)
{
    RegisterPyramidLevel& homography_level = register_pyr_levels_["homography"];
    RegisterPyramidLevel& affine_level     = register_pyr_levels_["affine"];

    // Generate scaled reference image
    preprocess_image<uint8_t>(homography_level.work_scale,
                              homography_level.preprocess_blur_border,
                              homography_level.preprocess_blur_std,
                              reference,
                              homography_level.reference_preprocessed);

    affine_level.reference_preprocessed =
        homography_level.reference_preprocessed;

    // Declare gradient images
    Mat grad_x_reference;
    Mat grad_y_reference;

    // Generate gradients
    DerivativeMethod::template derivative<PixelType, GradPixelType>(
        homography_level.reference_preprocessed,
        ImageDerivativeAxis::dX,
        grad_x_reference);

    DerivativeMethod::template derivative<PixelType, GradPixelType>(
        homography_level.reference_preprocessed,
        ImageDerivativeAxis::dY,
        grad_y_reference);

    ///
    // Crop image borders to match their sizes with the second derivative
    const int gradient_border = DerivativeMethod::border_crop_size();

    BoundingBox border_bb_1{
        {Point<float>{gradient_border, gradient_border}},
        {Point<float>{
            static_cast<float>(grad_x_reference.cols - gradient_border - 1),
            static_cast<float>(grad_x_reference.rows - gradient_border - 1)}}};

    Mat cropped_grad_x =
        image_crop<GradPixelType>(grad_x_reference, border_bb_1);
    Mat cropped_grad_y =
        image_crop<GradPixelType>(grad_y_reference, border_bb_1);

    // Generate steepest gradient images
    generate_steepest_gradient<GradPixelType, TransformClass>(
        cropped_grad_x, cropped_grad_y, homography_level.steepest_gradient_r);

    generate_steepest_gradient<GradPixelType, TranslationTransform<float>>(
        cropped_grad_x, cropped_grad_y, affine_level.steepest_gradient_r);

    // Generate Mutual Information Hessian of reference image with itself at
    // origin
    generate_self_ic_hessian<PixelType,
                             GradPixelType,
                             BinningMethod,
                             TransformClass,
                             DerivativeMethod>(
        homography_level.reference_preprocessed,
        grad_x_reference,
        grad_y_reference,
        homography_level.steepest_gradient_r,
        homography_level.mi_hessian);

    generate_self_ic_hessian<PixelType,
                             GradPixelType,
                             BinningMethod,
                             TranslationTransform<float>,
                             DerivativeMethod>(
        affine_level.reference_preprocessed,
        grad_x_reference,
        grad_y_reference,
        affine_level.steepest_gradient_r,
        affine_level.mi_hessian);
}
}
