#ifndef _TINYCV_REGISTRATION_H_
#define _TINYCV_REGISTRATION_H_

#include "derivative.h"
#include "histogram.h"
#include "interpolation.h"
#include "mat.h"
#include "mutual_information.h"
#include "transform.h"


template <typename GradPixelType, typename TransformClass>
void generate_steepest_gradient(const Mat& grad_x,
                                const Mat& grad_y,
                                Mat& steepest_grad)
{
    const int number_transform_params = TransformClass::number_parameters;

    // All gradient images must have the same dimensions
    assert(grad_x.cols == grad_y.cols);
    assert(grad_x.rows == grad_y.rows);

    // All gradient images must have a single channel
    assert(grad_x.channels() == 1);
    assert(grad_y.channels() == 1);

    // All gradient images must have the same pixel type: GradPixelType
    assert(grad_x.type() == Mat::get_type_enum<GradPixelType>());
    assert(grad_x.type() == grad_y.type());

    // The transform must use the same pixel type as the gradients
    assert(Mat::get_type_enum<typename TransformClass::ElementType>() ==
           Mat::get_type_enum<GradPixelType>());

    if (steepest_grad.empty()) {
        steepest_grad.create<GradPixelType>(
            grad_x.rows, grad_x.cols, number_transform_params);
    } else {
        // The output steepest must have the same dimensions as the input
        // gradients
        assert(steepest_grad.cols == grad_x.cols);
        assert(steepest_grad.rows == grad_x.rows);

        // Make sure the output has the correct number of channels
        assert(steepest_grad.channels() == number_transform_params);

        // Make sure the output has the correct type
        assert(steepest_grad.type() == Mat::get_type_enum<GradPixelType>());
    }

    Mat transform_jacobian;
    transform_jacobian.create<GradPixelType>(2, number_transform_params, 1);

    Mat::ConstIterator<GradPixelType> grad_x_it(grad_x);
    Mat::ConstIterator<GradPixelType> grad_y_it(grad_y);
    Mat::Iterator<GradPixelType> transform_jacob_it(transform_jacobian);
    Mat::Iterator<GradPixelType> steepest_it(steepest_grad);

    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            GradPixelType grad_x_pixel = grad_x_it(y, x, 0);
            GradPixelType grad_y_pixel = grad_y_it(y, x, 0);

            TransformClass::jacobian_origin(static_cast<GradPixelType>(x),
                                            static_cast<GradPixelType>(y),
                                            transform_jacobian);

            for (int param = 0; param < number_transform_params; ++param) {
                steepest_it(y, x, param) =
                    grad_x_pixel * transform_jacob_it(0, param, 0) +
                    grad_y_pixel * transform_jacob_it(1, param, 0);
            }
        }
    }
}

template <typename GradPixelType, typename TransformClass>
void generate_steepest_hessian(const Mat& grad_x,
                               const Mat& grad_y,
                               const Mat& grad_xx,
                               const Mat& grad_xy,
                               const Mat& grad_yx,
                               const Mat& grad_yy,
                               Mat& steepest_hess)
{
    const int number_transform_params = TransformClass::number_parameters;

    // All gradient images must have the same dimensions
    assert(grad_x.cols == grad_y.cols && grad_x.rows == grad_y.rows);
    assert(grad_x.cols == grad_xx.cols && grad_x.rows == grad_xx.rows);
    assert(grad_x.cols == grad_xy.cols && grad_x.rows == grad_xy.rows);
    assert(grad_x.cols == grad_yx.cols && grad_x.rows == grad_yx.rows);
    assert(grad_x.cols == grad_yy.cols && grad_x.rows == grad_yy.rows);

    // All gradient images must have a single channel
    assert(grad_x.channels() == 1);
    assert(grad_y.channels() == 1);
    assert(grad_xx.channels() == 1);
    assert(grad_xy.channels() == 1);
    assert(grad_yx.channels() == 1);
    assert(grad_yy.channels() == 1);

    // All gradient images must have the same pixel type: GradPixelType
    assert(grad_x.type() == Mat::get_type_enum<GradPixelType>());
    assert(grad_x.type() == grad_y.type());
    assert(grad_x.type() == grad_xx.type());
    assert(grad_x.type() == grad_xy.type());
    assert(grad_x.type() == grad_yx.type());
    assert(grad_x.type() == grad_yy.type());

    // The transform must use the same pixel type as the gradients
    assert(Mat::get_type_enum<typename TransformClass::ElementType>() ==
           Mat::get_type_enum<GradPixelType>());

    // clang-format off
    using SteepestColType  = Eigen::Matrix<GradPixelType,
                                           number_transform_params,
                                           1>;
    using TransformRowType = Eigen::Matrix<GradPixelType,
                                           1,
                                           number_transform_params>;
    using HessianMatType   = Eigen::Matrix<GradPixelType,
                                           number_transform_params,
                                           number_transform_params,
                                           Eigen::RowMajor>;
    // clang-format on

    ///
    // Allocate output
    if (steepest_hess.empty()) {
        steepest_hess.create<GradPixelType>(grad_x.rows,
                                            grad_x.cols,
                                            number_transform_params *
                                                number_transform_params);
    } else {
        // The output steepest must have the same dimensions as the input
        // gradients
        assert(grad_x.cols == steepest_hess.cols &&
               grad_x.rows == steepest_hess.rows);

        // Make sure the output has the correct number of channels
        assert(steepest_hess.channels() ==
               number_transform_params * number_transform_params);

        // Make sure the output has the correct type
        assert(steepest_hess.type() == Mat::get_type_enum<GradPixelType>());
    }

    ///
    // Generate second order steepest images
    Mat steepest_grad_x;
    Mat steepest_grad_y;

    generate_steepest_gradient<GradPixelType, TransformClass>(
        grad_xx, grad_xy, steepest_grad_x);

    generate_steepest_gradient<GradPixelType, TransformClass>(
        grad_yx, grad_yy, steepest_grad_y);

    ///
    // Allocate buffer for transform jacobian
    Mat transform_jacobian;
    transform_jacobian.create<GradPixelType>(2, number_transform_params, 1);

    ///
    // Allocate buffer for transform hessians
    Mat transform_hessian_x;
    Mat transform_hessian_y;
    transform_hessian_x.create<GradPixelType>(
        number_transform_params, number_transform_params, 1);
    transform_hessian_y.create<GradPixelType>(
        number_transform_params, number_transform_params, 1);

    ///
    // Create iterators
    Mat::Iterator<GradPixelType> steepest_x_it(steepest_grad_x);
    Mat::Iterator<GradPixelType> steepest_y_it(steepest_grad_y);
    Mat::Iterator<GradPixelType> img_hessian_it(steepest_hess);
    Mat::ConstIterator<GradPixelType> grad_x_it(grad_x);
    Mat::ConstIterator<GradPixelType> grad_y_it(grad_y);
    Mat::Iterator<GradPixelType> transf_jacobian_it(transform_jacobian);

    ///
    // Compute image hessian
    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            ///
            // Generate transform jacobian
            TransformClass::jacobian_origin(static_cast<GradPixelType>(x),
                                            static_cast<GradPixelType>(y),
                                            transform_jacobian);

            ///
            // Generate transform hessians
            TransformClass::hessian_x_origin(static_cast<GradPixelType>(x),
                                             static_cast<GradPixelType>(y),
                                             transform_hessian_x);
            TransformClass::hessian_y_origin(static_cast<GradPixelType>(x),
                                             static_cast<GradPixelType>(y),
                                             transform_hessian_y);

            ///
            // Compute pixel output value, which is a hessian matrix in itself
            Eigen::Map<SteepestColType> steepest_x_col(&steepest_x_it(y, x, 0));
            Eigen::Map<SteepestColType> steepest_y_col(&steepest_y_it(y, x, 0));

            Eigen::Map<TransformRowType> transform_x_row(
                &transf_jacobian_it(0, 0, 0));
            Eigen::Map<TransformRowType> transform_y_row(
                &transf_jacobian_it(1, 0, 0));

            Eigen::Map<HessianMatType> pixel_hessian(&img_hessian_it(y, x, 0));

            Eigen::Map<HessianMatType> transf_hessian_x_mat(
                static_cast<GradPixelType*>(transform_hessian_x.data));
            Eigen::Map<HessianMatType> transf_hessian_y_mat(
                static_cast<GradPixelType*>(transform_hessian_y.data));

            pixel_hessian = steepest_x_col * transform_x_row +
                            steepest_y_col * transform_y_row +
                            grad_x_it(y, x, 0) * transf_hessian_x_mat +
                            grad_y_it(y, x, 0) * transf_hessian_y_mat;
        }
    }

    return;
}

/**
 * Generate inverse compositional Hessian of the Mutual Information between
 * reference image and itself
 * TODO this should be the private method of the Newton class/namespace
 */
template <typename PixelType,
          typename GradPixelType,
          typename BinningMethod,
          typename TransformClass,
          typename DerivativeMethod>
void generate_self_ic_hessian(const Mat& img_reference,
                              const Mat& grad_x_reference,
                              const Mat& grad_y_reference,
                              const Mat& steepest_gradient_r,
                              Mat& mi_hessian)
{
    // Declare gradient images
    Mat grad_xx_reference;
    Mat grad_xy_reference;
    Mat grad_yy_reference;

    // Generate second gradient images
    DerivativeMethod::template derivative<GradPixelType, GradPixelType>(
        grad_x_reference, ImageDerivativeAxis::dX, grad_xx_reference);

    DerivativeMethod::template derivative<GradPixelType, GradPixelType>(
        grad_x_reference, ImageDerivativeAxis::dY, grad_xy_reference);

    DerivativeMethod::template derivative<GradPixelType, GradPixelType>(
        grad_y_reference, ImageDerivativeAxis::dY, grad_yy_reference);

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

    BoundingBox border_bb_2{
        {{gradient_border * 2, gradient_border * 2}},
        {{static_cast<float>(img_reference.cols - gradient_border * 2 - 1),
          static_cast<float>(img_reference.rows - gradient_border * 2 - 1)}}};
    Mat cropped_reference = image_crop<PixelType>(img_reference, border_bb_2);

    ///
    // Generate steepest hessian image
    Mat steepest_hessian_r;
    generate_steepest_hessian<GradPixelType, TransformClass>(
        cropped_grad_x,
        cropped_grad_y,
        grad_xx_reference,
        grad_xy_reference,
        grad_xy_reference,
        grad_yy_reference,
        steepest_hessian_r);

    // Declare histogram matrices
    Mat histogram_r;
    Mat histogram_rt;
    Mat histogram_rt_grad;

    // Compute Mutual Information gradient
    double histogram_r_sum  = 0.0;
    double histogram_rt_sum = 0.0;
    joint_hist_gradient<PixelType,
                        GradPixelType,
                        BinningMethod,
                        TransformClass,
                        PositiveMaskIterator,
                        PositiveMaskIterator>(cropped_reference,
                                              {},
                                              steepest_gradient_r,
                                              cropped_reference,
                                              {},
                                              histogram_r_sum,
                                              histogram_rt_sum,
                                              histogram_r,
                                              histogram_rt,
                                              histogram_rt_grad);

    // Compute MI Hessian
    mutual_information_hessian<PixelType,
                               GradPixelType,
                               BinningMethod,
                               TransformClass,
                               PositiveMaskIterator>(cropped_reference,
                                                     steepest_gradient_r,
                                                     steepest_hessian_r,
                                                     cropped_reference,
                                                     {},
                                                     histogram_r,
                                                     histogram_rt,
                                                     histogram_rt_grad,
                                                     histogram_r_sum,
                                                     histogram_rt_sum,
                                                     mi_hessian);
}

template <typename InputPixelType>
void preprocess_image(const float scale,
                      const int blur_kernel_border,
                      const float blur_std,
                      const Mat& input,
                      Mat& output)
{
    using TransformClass  = HomographyTransform<float>;
    using OutputPixelType = float;

    assert(input.channels() == 1);

    Mat scale_params;
    scale_params.create<float>(1, 8, 1);

    // clang-format off
    scale_params << std::initializer_list<float>{
        scale - 1, 0, 0,
        0, scale - 1, 0,
        0, 0
    };
    // clang-format on

    Mat output_scaled;
    Mat output_scaled_mask;

    image_transform<InputPixelType,
                    1,
                    TransformClass,
                    bilinear_sample<InputPixelType, 1>>(
        input,
        scale_params,
        bounding_box_transform<TransformClass>(BoundingBox(input),
                                               scale_params),
        output_scaled,
        output_scaled_mask);

    Mat output_blurred;
    gaussian_blur<InputPixelType, InputPixelType, 1>(
        output_scaled, blur_kernel_border, blur_std, output_blurred);

    output = image_remap_histogram<InputPixelType,
                                   OutputPixelType,
                                   PositiveMaskIterator>(output_blurred, {});


    return;
}

template <typename PixelType,
          typename GradPixelType,
          typename DerivativeMethod,
          typename TransformClass>
bool register_impl(const Mat& img_reference,
                   const Mat& img_tracked,
                   const Mat& initial_guess,
                   Mat& composed_p)
{
    const int number_max_iterations   = 30;
    const int number_transform_params = TransformClass::number_parameters;
    const float convergence_threshold = 1e-3f;

    using BinningMethod = BSpline4;
    using TransformColType =
        Eigen::Matrix<GradPixelType, number_transform_params, 1>;
    using HessianMatType = Eigen::Matrix<GradPixelType,
                                         number_transform_params,
                                         number_transform_params,
                                         Eigen::RowMajor>;

    // Declare gradient images
    Mat grad_x_reference;
    Mat grad_y_reference;

    // Generate gradients
    DerivativeMethod::template derivative<PixelType, GradPixelType>(
        img_reference, ImageDerivativeAxis::dX, grad_x_reference);

    DerivativeMethod::template derivative<PixelType, GradPixelType>(
        img_reference, ImageDerivativeAxis::dY, grad_y_reference);

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

    BoundingBox border_bb_2{
        {{gradient_border * 2, gradient_border * 2}},
        {{static_cast<float>(img_reference.cols - gradient_border * 2 - 1),
          static_cast<float>(img_reference.rows - gradient_border * 2 - 1)}}};
    Mat cropped_reference = image_crop<PixelType>(img_reference, border_bb_2);
    Mat cropped_tracked   = image_crop<PixelType>(img_tracked, border_bb_2);

    // Generate steepest gradient image
    Mat steepest_gradient_r;
    generate_steepest_gradient<GradPixelType, TransformClass>(
        cropped_grad_x, cropped_grad_y, steepest_gradient_r);

    // Generate Mutual Information Hessian of reference image with itself at
    // origin
    Mat mi_hessian;
    generate_self_ic_hessian<PixelType,
                             GradPixelType,
                             BinningMethod,
                             TransformClass,
                             DerivativeMethod>(img_reference,
                                               grad_x_reference,
                                               grad_y_reference,
                                               steepest_gradient_r,
                                               mi_hessian);
    Eigen::Map<HessianMatType> mi_hessian_mat(
        static_cast<GradPixelType*>(mi_hessian.data));
    HessianMatType mi_hessian_inv = mi_hessian_mat.inverse();

    // Bounding box of the input reference image
    const BoundingBox cropped_ref_bb(cropped_reference);

    // Accumulated transform parameter
    if (composed_p.empty()) {
        composed_p.create<GradPixelType>(1, number_transform_params, 1);
    } else {
        // TODO assert
    }

    // Transform parameter step matrix and its inverse
    Mat delta_p;
    delta_p.create<GradPixelType>(1, number_transform_params, 1);
    Mat delta_p_inv;
    delta_p_inv.create<GradPixelType>(1, number_transform_params, 1);

    // Initialize transform with initial guess
    composed_p << initial_guess;

    // Convert initial guess from input image space to cropped work space
    Mat composed_p_homog;

    if (std::is_same<TransformClass, HomographyTransform<PixelType>>::value) {
        composed_p_homog = composed_p;
    } else {
        TransformClass::to_homography(composed_p, composed_p_homog);
    }

    HomographyTransform<PixelType>::change_position(
        {-2 * gradient_border, -2 * gradient_border},
        composed_p_homog,
        composed_p_homog);

    if (!std::is_same<TransformClass, HomographyTransform<PixelType>>::value) {
        TransformClass::from_homography(composed_p_homog, composed_p);
    }

    // Mutual information gradient, to be computed at each iteration
    Mat mi_gradient;

    // Reference image, tracked image and its mask for each iteration
    Mat local_reference;
    Mat local_tracked;
    Mat local_mask_t;

    // Run iterative Newton-Dame algorithm
    bool converged = false;
    for (int iteration = 1;; ++iteration) {
        // Compute transform bounding box
        BoundingBox interest_bb = bounding_box_intersect(
            bounding_box_transform<TransformClass>(cropped_ref_bb, composed_p),
            cropped_ref_bb);

        // Generate cropped reference and steepest grad images
        local_reference = image_crop<PixelType>(cropped_reference, interest_bb);
        Mat local_steepest_grad_r =
            image_crop<PixelType>(steepest_gradient_r, interest_bb);

        // Transform tracked image
        image_transform<PixelType,
                        1,
                        TransformClass,
                        bilinear_sample<PixelType, 1>>(cropped_tracked,
                                                       composed_p,
                                                       interest_bb,
                                                       local_tracked,
                                                       local_mask_t);

        // Compute gradient
        mutual_information_gradient<PixelType,
                                    GradPixelType,
                                    BinningMethod,
                                    TransformClass>(local_reference,
                                                    local_steepest_grad_r,
                                                    local_tracked,
                                                    local_mask_t,
                                                    mi_gradient);

        // Compute step
        Eigen::Map<TransformColType> delta_p_mat(
            static_cast<GradPixelType*>(delta_p.data));
        Eigen::Map<TransformColType> mi_gradient_mat(
            static_cast<GradPixelType*>(mi_gradient.data));

        std::cout << mutual_information<PixelType, BinningMethod>(
                         local_reference,
                         local_tracked,
                         Mat::ConstIterator<MaskPixelType>(local_mask_t))
                  << " " << mi_gradient_mat.transpose() << std::endl;

        delta_p_mat = mi_hessian_inv * mi_gradient_mat;

        // Check for convergence
        if (delta_p_mat.norm() < convergence_threshold) {
            converged = true;
            break;
        } else if (iteration == number_max_iterations) {
            break;
        }

        // Invert delta_p
        TransformClass::inverse(delta_p, delta_p_inv);

        // Update composed_p
        TransformClass::compose(composed_p, delta_p_inv, composed_p);
    }

    std::cout << "\n"
              << Eigen::Map<TransformColType>(
                     static_cast<GradPixelType*>(composed_p.data))
                     .transpose()
              << std::endl;

    if (converged) {
        // Convert transform from cropped work space to un-cropped input image
        // space
        Mat composed_p_homog;

        if (std::is_same<TransformClass,
                         HomographyTransform<PixelType>>::value) {
            composed_p_homog = composed_p;
        } else {
            TransformClass::to_homography(composed_p, composed_p_homog);
        }

        HomographyTransform<PixelType>::change_position(
            {2 * gradient_border, 2 * gradient_border},
            composed_p_homog,
            composed_p_homog);

        if (!std::is_same<TransformClass,
                          HomographyTransform<PixelType>>::value) {
            TransformClass::from_homography(composed_p_homog, composed_p);
        }
    }

    return converged;
}

bool register_homography(const Mat& reference,
                         const Mat& tracked,
                         Mat& transform_homography);

#endif
