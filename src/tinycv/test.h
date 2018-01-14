#ifndef _TINYCV_TEST_H_
#define _TINYCV_TEST_H_

#include <fstream>

#include "bounding_box.h"
#include "derivative.h"
#include "histogram.h"
#include "interpolation.h"
#include "mat.h"
#include "mutual_information.h"
#include "registration.h"
#include "transform.h"


namespace tinycv
{

static const float TEST_ALPHA_RANGE   = 1.f;
static const float TEST_DELTA_ALPHA   = 0.5f;
static const int TEST_ALPHA_MODEL_IDX = 2;

void visualize_steepest_gradient(const Mat& steepest_grad)
{
    Mat steepest0;
    Mat steepest1;
    Mat steepest2;
    Mat steepest3;
    Mat steepest4;
    Mat steepest5;
    Mat steepest6;
    Mat steepest7;

    steepest0.create<float>(steepest_grad.rows, steepest_grad.cols, 1);
    steepest1.create<float>(steepest_grad.rows, steepest_grad.cols, 1);
    steepest2.create<float>(steepest_grad.rows, steepest_grad.cols, 1);
    steepest3.create<float>(steepest_grad.rows, steepest_grad.cols, 1);
    steepest4.create<float>(steepest_grad.rows, steepest_grad.cols, 1);
    steepest5.create<float>(steepest_grad.rows, steepest_grad.cols, 1);
    steepest6.create<float>(steepest_grad.rows, steepest_grad.cols, 1);
    steepest7.create<float>(steepest_grad.rows, steepest_grad.cols, 1);

    Mat::Iterator<float> steepest0_it(steepest0);
    Mat::Iterator<float> steepest1_it(steepest1);
    Mat::Iterator<float> steepest2_it(steepest2);
    Mat::Iterator<float> steepest3_it(steepest3);
    Mat::Iterator<float> steepest4_it(steepest4);
    Mat::Iterator<float> steepest5_it(steepest5);
    Mat::Iterator<float> steepest6_it(steepest6);
    Mat::Iterator<float> steepest7_it(steepest7);

    Mat::ConstIterator<float> steepest_it(steepest_grad);

    for (int y = 0; y < steepest0.rows; ++y) {
        for (int x = 0; x < steepest0.cols; ++x) {
            steepest0_it(y, x, 0) = steepest_it(y, x, 0);
            steepest1_it(y, x, 0) = steepest_it(y, x, 1);
            steepest2_it(y, x, 0) = steepest_it(y, x, 2);
            steepest3_it(y, x, 0) = steepest_it(y, x, 3);
            steepest4_it(y, x, 0) = steepest_it(y, x, 4);
            steepest5_it(y, x, 0) = steepest_it(y, x, 5);
            steepest6_it(y, x, 0) = steepest_it(y, x, 6);
            steepest7_it(y, x, 0) = steepest_it(y, x, 7);
        }
    }

    return;
}

void visualize_steepest_hessian(const Mat& steepest_hess)
{
    Mat steephess0;
    Mat steephess1;
    Mat steephess2;
    Mat steephess3;
    Mat steephess4;
    Mat steephess5;
    Mat steephess6;
    Mat steephess7;

    const int num_params = 8;

    steephess0.create<float>(steepest_hess.rows, steepest_hess.cols, 1);
    steephess1.create<float>(steepest_hess.rows, steepest_hess.cols, 1);
    steephess2.create<float>(steepest_hess.rows, steepest_hess.cols, 1);
    steephess3.create<float>(steepest_hess.rows, steepest_hess.cols, 1);
    steephess4.create<float>(steepest_hess.rows, steepest_hess.cols, 1);
    steephess5.create<float>(steepest_hess.rows, steepest_hess.cols, 1);
    steephess6.create<float>(steepest_hess.rows, steepest_hess.cols, 1);
    steephess7.create<float>(steepest_hess.rows, steepest_hess.cols, 1);

    Mat::Iterator<float> steephess0_it(steephess0);
    Mat::Iterator<float> steephess1_it(steephess1);
    Mat::Iterator<float> steephess2_it(steephess2);
    Mat::Iterator<float> steephess3_it(steephess3);
    Mat::Iterator<float> steephess4_it(steephess4);
    Mat::Iterator<float> steephess5_it(steephess5);
    Mat::Iterator<float> steephess6_it(steephess6);
    Mat::Iterator<float> steephess7_it(steephess7);

    Mat::ConstIterator<float> steepest_it(steepest_hess);

    for (int y = 0; y < steephess0.rows; ++y) {
        for (int x = 0; x < steephess0.cols; ++x) {
            steephess0_it(y, x, 0) = steepest_it(y, x, 0);
            steephess1_it(y, x, 0) = steepest_it(y, x, 1 * num_params + 1);
            steephess2_it(y, x, 0) = steepest_it(y, x, 2 * num_params + 2);
            steephess3_it(y, x, 0) = steepest_it(y, x, 3 * num_params + 3);
            steephess4_it(y, x, 0) = steepest_it(y, x, 4 * num_params + 4);
            steephess5_it(y, x, 0) = steepest_it(y, x, 5 * num_params + 5);
            steephess6_it(y, x, 0) = steepest_it(y, x, 6 * num_params + 6);
            steephess7_it(y, x, 0) = steepest_it(y, x, 7 * num_params + 7);
        }
    }

    return;
}

void test_steepest_descent_imgs(const Mat& source)
{
    using GradPixelType    = float;
    using DerivativeMethod = DerivativeHoloborodko<1, FilterOrder::Fifth>;

    Mat grad_x;
    Mat grad_y;
    DerivativeMethod::derivative<uint8_t, GradPixelType>(
        source, ImageDerivativeAxis::dX, grad_x);

    DerivativeMethod::derivative<uint8_t, GradPixelType>(
        source, ImageDerivativeAxis::dY, grad_y);

    Mat steepest_img;
    generate_steepest_gradient<GradPixelType,
                               HomographyTransform<GradPixelType>>(
        grad_x, grad_y, steepest_img);

    visualize_steepest_gradient(steepest_img);

    return;
}

void test_image_derivative(const Mat& source)
{
    Mat dx;
    Mat dy;
    Mat dxy;

    using DerivativeMethod = DerivativeHoloborodko<1, FilterOrder::Fifth>;
    // const int cropped_border = DerivativeMethod::border_crop_size();

    DerivativeMethod::derivative<uint8_t, float>(
        source, ImageDerivativeAxis::dX, dx);

    DerivativeMethod::derivative<uint8_t, float>(
        source, ImageDerivativeAxis::dY, dy);

    DerivativeMethod::derivative<float, float>(
        dx, ImageDerivativeAxis::dY, dxy);

    Mat naive_dx;
    Mat naive_dy;
    DerivativeNaive<1>::derivative<uint8_t, float>(
        source, ImageDerivativeAxis::dX, naive_dx);
    DerivativeNaive<1>::derivative<uint8_t, float>(
        source, ImageDerivativeAxis::dX, naive_dy);
    const int cropped_border = DerivativeNaive<1>::border_crop_size();

    Mat cropped = image_crop<uint8_t>(
        source,
        BoundingBox({{cropped_border, cropped_border},
                     {static_cast<float>(source.cols - cropped_border - 1),
                      static_cast<float>(source.rows - cropped_border - 1)}}));

    return;
}

void test_bspline_4()
{
    const float LIM = 3.0f;
    using std::vector;

    const float eps = 1e-3f;

    /*
    for (float i = -LIM; i <= LIM; i += 0.01f) {
        std::cout << (BSpline4::histogram_bin_function(i + eps) -
                      BSpline4::histogram_bin_function(i)) /
                         eps
                  << " " << BSpline4::hbf_derivative(i) << std::endl;
    }
    */

    for (float i = -LIM; i <= LIM; i += 0.01f) {
        std::cout << (BSpline4::hbf_derivative(i + eps) -
                      BSpline4::hbf_derivative(i)) /
                         eps
                  << " " << BSpline4::hbf_second_derivative(i) << std::endl;
    }
}

void generate_mi_space(const std::string& file_name, const Mat& source)
{
    using BinningMethod  = BSpline4;
    using PixelType      = float;
    using TransformClass = HomographyTransform<float>;

    // Open output file
    std::ofstream file_handle;
    file_handle.open(file_name, std::ios::out);

    Mat source_small;

    ///
    // Generate the scaled-down destination image
    const float scale = 1.0f;

    Mat scale_params;
    scale_params.create<float>(1, 8, 1);
    // clang-format off
    scale_params << std::initializer_list<float>{
        scale - 1, 0, 0,
        0, scale - 1, 0,
        0, 0
    };
    // clang-format on

    {
        Mat tmp_mask;

        image_transform<PixelType,
                        1,
                        TransformClass,
                        bilinear_sample<PixelType, 1>>(
            source,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(source),
                                                   scale_params),
            source_small,
            tmp_mask);
    }

    /// Translation
    // clang-format off
    std::vector<float> transform_data {
        0.f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 0.f
    };
    // clang-format on

    Mat transform_params;
    transform_params.create_from_buffer<float>(
        transform_data.data(), 1, 8, 1, 8);
    for (float y = 0; y <= 0; y += 0.1f) {
        for (float x = -TEST_ALPHA_RANGE; x <= TEST_ALPHA_RANGE;
             x += TEST_DELTA_ALPHA) {
            // clang-format off
            transform_data[TEST_ALPHA_MODEL_IDX] = x;
            // clang-format on

            Mat transformed_mask;
            Mat transformed_img;

            BoundingBox input_bb = BoundingBox(source);

            BoundingBox output_bb = bounding_box_intersect(
                bounding_box_transform<TransformClass>(input_bb, scale_params),
                BoundingBox(source_small));

            Mat cropped_img = image_crop<PixelType>(source_small, output_bb);
            image_transform<PixelType,
                            1,
                            TransformClass,
                            bilinear_sample<PixelType, 1>>(source,
                                                           transform_params,
                                                           output_bb,
                                                           transformed_img,
                                                           transformed_mask);
            double mi = mutual_information<PixelType, BinningMethod>(
                cropped_img, transformed_img, transformed_mask);

            file_handle << mi << " ";
        }
        file_handle << std::endl;
    }

    /*
    /// Rotation
    const float dtheta = 180.0f / 180.f;
    for (float y = -dtheta; y <= dtheta; y += 1.0f / 180.f) {
        float y_rad = y * static_cast<float>(M_PI) / 180.f;
        // clang-format off
        std::vector<float> recenter_data {
            1.f, 0.f, static_cast<float>(small.cols) / 2.f,
            0.f, 1.f, static_cast<float>(small.rows) / 2.f,
            0.f, 0.f, 1.f
        };
        std::vector<float> rotate_data {
            std::cos(y_rad), -std::sin(y_rad), 0,
            std::sin(y_rad), std::cos(y_rad), 0,
            0.f, 0.f, 1.f
        };
        // clang-format on

        Eigen::Map<const Matrix3fRowMajor> recenter(recenter_data.data());
        Eigen::Map<const Matrix3fRowMajor> rotate(rotate_data.data());
        Matrix3fRowMajor transform = recenter * rotate * recenter.inverse();

        Mat transformed_mask;
        Mat transformed_img;

        BoundingBox input_bb = BoundingBox(small);

        BoundingBox output_bb = bounding_box_intersect(
            bounding_box_transform(input_bb, transform.data()), input_bb);
        Mat cropped_img = image_crop<PixelType>(small, output_bb);
        image_transform<PixelType, 1, jitter_sample<PixelType, 1>>(
            small,
            transform.data(),
            output_bb,
            transformed_img,
            transformed_mask);
        double mi = mutual_information<PixelType>(
            cropped_img, transformed_img, transformed_mask);

        std::cout << mi << " ";
    }
    std::cout << std::endl;
    */

    /// Perspective
    /*
    const float dvalue = 0.00001f;
    for (float y = -dvalue; y <= dvalue; y += 0.000001f / 2.0f) {
        for (float x = -dvalue; x <= dvalue; x += 0.000001f / 2.0f) {
            // clang-format off
            std::vector<float> perspective_data {
                1.f, 0.f, 0.f,
                0.f, 1.f, 0.f,
                x, y, 1.f
            };
            // clang-format on

            Eigen::Map<const Matrix3fRowMajor> perspective(
                perspective_data.data());

            Matrix3fRowMajor transform = perspective * scale_mat;

            Mat transformed_mask;
            Mat transformed_img;

            BoundingBox input_bb = BoundingBox(small_homog);

            BoundingBox output_bb = bounding_box_intersect(
                bounding_box_transform(input_bb, perspective.data()), input_bb);
            Mat cropped_img = image_crop<PixelType>(small_homog, output_bb);

            image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
                source,
                transform.data(),
                output_bb,
                transformed_img,
                transformed_mask);

            double mi = mutual_information<PixelType>(
                cropped_img, transformed_img, transformed_mask);

            std::cout << mi << " ";
        }
        std::cout << std::endl;
    }
    */

    file_handle.close();

    return;
}

void generate_mi_derivative_space(const std::string& file_name,
                                  const Mat& source,
                                  const Mat& destination)
{
    using PixelType           = float;
    using GradPixelType       = float;
    using TransformClass      = HomographyTransform<GradPixelType>;
    using DerivativeMethod    = DerivativeNaive<1>;
    const int gradient_border = DerivativeMethod::border_crop_size();
    using BinningMethod       = BSpline4;

    // Open output file
    std::ofstream file_handle;
    file_handle.open(file_name, std::ios::out);

    // Generate image derivatives
    Mat grad_x;
    Mat grad_y;

    DerivativeMethod::derivative<PixelType, GradPixelType>(
        destination, ImageDerivativeAxis::dX, grad_x);

    DerivativeMethod::derivative<PixelType, GradPixelType>(
        destination, ImageDerivativeAxis::dY, grad_y);

    // Crop image borders to match their sizes with the derivative
    BoundingBox border_bb{
        {{gradient_border, gradient_border}},
        {{static_cast<float>(destination.cols - gradient_border - 1),
          static_cast<float>(destination.rows - gradient_border - 1)}}};

    Mat cropped_destination = image_crop<PixelType>(destination, border_bb);
    Mat cropped_source      = image_crop<PixelType>(source, border_bb);

    ///
    // Generate the scaled-down destination image and its gradients at origin
    const float scale = 1.0f;

    Mat scale_params;
    scale_params.create<float>(1, 8, 1);

    // clang-format off
    scale_params << std::initializer_list<float>{
        scale - 1, 0, 0,
        0, scale - 1, 0,
        0, 0
    };
    // clang-format on

    Mat destination_small;
    Mat grad_x_small;
    Mat grad_y_small;
    {
        Mat tmp_mask;

        // Scale destination
        image_transform<PixelType,
                        1,
                        TransformClass,
                        bilinear_sample<PixelType, 1>>(
            cropped_destination,
            scale_params,
            bounding_box_transform<TransformClass>(
                BoundingBox(cropped_destination), scale_params),
            destination_small,
            tmp_mask);

        // Scale gradient
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            grad_x,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(grad_x),
                                                   scale_params),
            grad_x_small,
            tmp_mask);
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            grad_y,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(grad_y),
                                                   scale_params),
            grad_y_small,
            tmp_mask);
    }

    // Generate steepest descent image
    Mat steepest_destination;
    generate_steepest_gradient<GradPixelType, TransformClass>(
        grad_x_small, grad_y_small, steepest_destination);

    Mat gradient;
    gradient.create<float>(1, TransformClass::number_parameters, 1);

    // clang-format off
    std::vector<float> transform_params_data {
        0.f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 0.f,
    };
    // clang-format on

    Mat transform_params;
    transform_params.create_from_buffer<float>(
        transform_params_data.data(), 1, 8, 1, 8);

    for (float alpha = -TEST_ALPHA_RANGE; alpha <= TEST_ALPHA_RANGE;
         alpha += TEST_DELTA_ALPHA) {
        // clang-format off
        transform_params_data[TEST_ALPHA_MODEL_IDX] = alpha;
        // clang-format on

        BoundingBox destination_bb(cropped_destination);
        BoundingBox interest_bb =
            bounding_box_intersect(bounding_box_transform<TransformClass>(
                                       destination_bb, transform_params),
                                   BoundingBox(destination_small));

        Mat local_destination =
            image_crop<PixelType>(destination_small, interest_bb);
        Mat local_steepest =
            image_crop<float>(steepest_destination, interest_bb);

        Mat local_mask;
        Mat local_source;
        image_transform<PixelType,
                        1,
                        TransformClass,
                        bilinear_sample<PixelType, 1>>(cropped_source,
                                                       transform_params,
                                                       interest_bb,
                                                       local_source,
                                                       local_mask);

        mutual_information_gradient<PixelType,
                                    GradPixelType,
                                    BinningMethod,
                                    HomographyTransform<float>>(
            local_destination,
            local_steepest,
            local_source,
            local_mask,
            gradient);

        Mat::Iterator<float> grad_it(gradient);

        file_handle << alpha << " " << grad_it(0, TEST_ALPHA_MODEL_IDX, 0)
                    << std::endl;
    }

    // Close output file
    file_handle.close();

    return;
}

void generate_mi_hessian_space(const std::string& file_name,
                               const Mat& source,
                               const Mat& destination)
{
    using PixelType        = float;
    using GradPixelType    = float;
    using TransformClass   = HomographyTransform<GradPixelType>;
    using BinningMethod    = BSpline4;
    using DerivativeMethod = DerivativeNaive<1>;

    const int gradient_border = DerivativeMethod::border_crop_size();

    // Open output file
    std::ofstream file_handle;
    file_handle.open(file_name, std::ios::out);

    // Generate image derivatives
    Mat grad_x;
    Mat grad_y;

    Mat grad_xx;
    Mat grad_xy;
    Mat grad_yx;
    Mat grad_yy;

    ///
    // Generate gradients
    DerivativeMethod::derivative<PixelType, GradPixelType>(
        destination, ImageDerivativeAxis::dX, grad_x);

    DerivativeMethod::derivative<PixelType, GradPixelType>(
        destination, ImageDerivativeAxis::dY, grad_y);

    ///
    // Generate second gradients
    DerivativeMethod::derivative<GradPixelType, GradPixelType>(
        grad_x, ImageDerivativeAxis::dX, grad_xx);

    DerivativeMethod::derivative<GradPixelType, GradPixelType>(
        grad_x, ImageDerivativeAxis::dY, grad_xy);

    // Note that grad_xy = grad_yx, so Newton's method implementation can take
    // advantage of that for optimization purposes
    DerivativeMethod::derivative<GradPixelType, GradPixelType>(
        grad_y, ImageDerivativeAxis::dX, grad_yx);

    DerivativeMethod::derivative<GradPixelType, GradPixelType>(
        grad_y, ImageDerivativeAxis::dY, grad_yy);

    ///
    // Crop image borders to match their sizes with the second derivative
    BoundingBox border_bb_1{
        {{gradient_border, gradient_border}},
        {{static_cast<float>(grad_x.cols - gradient_border - 1),
          static_cast<float>(grad_x.rows - gradient_border - 1)}}};
    Mat cropped_grad_x = image_crop<GradPixelType>(grad_x, border_bb_1);
    Mat cropped_grad_y = image_crop<GradPixelType>(grad_y, border_bb_1);

    BoundingBox border_bb_2{
        {{gradient_border * 2, gradient_border * 2}},
        {{static_cast<float>(destination.cols - gradient_border * 2 - 1),
          static_cast<float>(destination.rows - gradient_border * 2 - 1)}}};
    Mat cropped_destination = image_crop<PixelType>(destination, border_bb_2);
    Mat cropped_source      = image_crop<PixelType>(source, border_bb_2);


    ///
    // Generate the scaled-down destination image and its gradients at origin
    const float scale = 1.0f;
    Mat scale_params;
    scale_params.create<float>(1, 8, 1);

    // clang-format off
    scale_params << std::initializer_list<float>{
        scale - 1, 0, 0,
        0, scale - 1, 0,
        0, 0
    };
    // clang-format on

    Mat destination_small;

    // Declare gradient matrices
    Mat grad_x_small;
    Mat grad_y_small;

    Mat grad_xx_small;
    Mat grad_xy_small;
    Mat grad_yx_small;
    Mat grad_yy_small;

    // Declare histogram matrices
    Mat histogram_r;
    Mat histogram_rt;
    Mat histogram_rt_grad;

    {
        Mat tmp_mask;

        // Scale destination
        image_transform<PixelType,
                        1,
                        TransformClass,
                        bilinear_sample<PixelType, 1>>(
            cropped_destination,
            scale_params,
            bounding_box_transform<TransformClass>(
                BoundingBox(cropped_destination), scale_params),
            destination_small,
            tmp_mask);

        // Scale gradient
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            cropped_grad_x,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(cropped_grad_x),
                                                   scale_params),
            grad_x_small,
            tmp_mask);
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            cropped_grad_y,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(cropped_grad_y),
                                                   scale_params),
            grad_y_small,
            tmp_mask);

        // Scale second gradients
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            grad_xx,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(grad_xx),
                                                   scale_params),
            grad_xx_small,
            tmp_mask);
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            grad_xy,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(grad_xy),
                                                   scale_params),
            grad_xy_small,
            tmp_mask);
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            grad_yx,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(grad_yx),
                                                   scale_params),
            grad_yx_small,
            tmp_mask);
        image_transform<GradPixelType,
                        1,
                        TransformClass,
                        bilinear_sample<GradPixelType, 1>>(
            grad_yy,
            scale_params,
            bounding_box_transform<TransformClass>(BoundingBox(grad_yy),
                                                   scale_params),
            grad_yy_small,
            tmp_mask);
    }

    // Generate steepest descent image
    Mat steepest_grad_dst;
    generate_steepest_gradient<GradPixelType, TransformClass>(
        grad_x_small, grad_y_small, steepest_grad_dst);

    // Generate transformed image hessian
    Mat steepest_hess_dst;
    generate_steepest_hessian<GradPixelType, TransformClass>(grad_x_small,
                                                             grad_y_small,
                                                             grad_xx_small,
                                                             grad_xy_small,
                                                             grad_yx_small,
                                                             grad_yy_small,
                                                             steepest_hess_dst);

    Mat hessian;
    hessian.create<float>(TransformClass::number_parameters,
                          TransformClass::number_parameters,
                          1);

    // clang-format off
    std::vector<float> transform_params_data {
        0.f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 0.f,
    };
    // clang-format on

    Mat transform_params;
    transform_params.create_from_buffer<float>(
        transform_params_data.data(), 1, 8, 1, 8);

    for (float alpha = -TEST_ALPHA_RANGE; alpha <= TEST_ALPHA_RANGE;
         alpha += TEST_DELTA_ALPHA) {
        transform_params_data[TEST_ALPHA_MODEL_IDX] = alpha;

        BoundingBox destination_bb(cropped_destination);
        BoundingBox interest_bb =
            bounding_box_intersect(bounding_box_transform<TransformClass>(
                                       destination_bb, transform_params),
                                   BoundingBox(destination_small));

        Mat local_destination =
            image_crop<PixelType>(destination_small, interest_bb);
        Mat local_steepest_grad =
            image_crop<float>(steepest_grad_dst, interest_bb);
        Mat local_steepest_hess =
            image_crop<float>(steepest_hess_dst, interest_bb);

        Mat local_mask;
        Mat local_source;
        image_transform<PixelType,
                        1,
                        TransformClass,
                        bilinear_sample<PixelType, 1>>(cropped_source,
                                                       transform_params,
                                                       interest_bb,
                                                       local_source,
                                                       local_mask);

        // Compute histogram for input images
        double histogram_r_sum  = 0.0;
        double histogram_rt_sum = 0.0;
        joint_hist_gradient<PixelType,
                            GradPixelType,
                            BinningMethod,
                            TransformClass,
                            PositiveMaskIterator,
                            Mat::ConstIterator<MaskPixelType>>(
            local_destination,
            {},
            local_steepest_grad,
            local_source,
            Mat::ConstIterator<MaskPixelType>(local_mask),
            histogram_r_sum,
            histogram_rt_sum,
            histogram_r,
            histogram_rt,
            histogram_rt_grad);

        // Compute MI Hessian
        mutual_information_hessian<PixelType,
                                   GradPixelType,
                                   BinningMethod,
                                   HomographyTransform<float>,
                                   Mat::ConstIterator<MaskPixelType>>(
            local_destination,
            local_steepest_grad,
            local_steepest_hess,
            local_source,
            Mat::ConstIterator<MaskPixelType>(local_mask),
            histogram_r,
            histogram_rt,
            histogram_rt_grad,
            histogram_r_sum,
            histogram_rt_sum,
            hessian);

        Mat::Iterator<float> hess_it(hessian);
        file_handle << alpha << " "
                    << hess_it(TEST_ALPHA_MODEL_IDX, TEST_ALPHA_MODEL_IDX, 0)
                    << std::endl;
    }

    file_handle.close();

    return;
}
}

#endif
