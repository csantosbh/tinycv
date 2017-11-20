#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "bounding_box.h"
#include "derivative.h"
#include "histogram.h"
#include "interpolation.h"
#include "mat.h"
#include "math.h"
#include "mutual_information.h"
#include "registration.h"
#include "sat.h"
#include "transform.h"


const float TEST_ALPHA_RANGE   = 10.f;
const float TEST_DELTA_ALPHA   = 0.1f;
const int TEST_ALPHA_MODEL_IDX = 2;

void visualize_steepest_descent_imgs(const Mat& steepest_img)
{
    Mat steepest0;
    Mat steepest1;
    Mat steepest2;
    Mat steepest3;
    Mat steepest4;
    Mat steepest5;
    Mat steepest6;
    Mat steepest7;

    steepest0.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steepest1.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steepest2.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steepest3.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steepest4.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steepest5.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steepest6.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steepest7.create<float>(steepest_img.rows, steepest_img.cols, 1);

    Mat::Iterator<float> steepest0_it(steepest0);
    Mat::Iterator<float> steepest1_it(steepest1);
    Mat::Iterator<float> steepest2_it(steepest2);
    Mat::Iterator<float> steepest3_it(steepest3);
    Mat::Iterator<float> steepest4_it(steepest4);
    Mat::Iterator<float> steepest5_it(steepest5);
    Mat::Iterator<float> steepest6_it(steepest6);
    Mat::Iterator<float> steepest7_it(steepest7);

    Mat::ConstIterator<float> steepest_it(steepest_img);

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

void visualize_steepest_hessian_imgs(const Mat& steepest_img)
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

    steephess0.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steephess1.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steephess2.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steephess3.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steephess4.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steephess5.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steephess6.create<float>(steepest_img.rows, steepest_img.cols, 1);
    steephess7.create<float>(steepest_img.rows, steepest_img.cols, 1);

    Mat::Iterator<float> steephess0_it(steephess0);
    Mat::Iterator<float> steephess1_it(steephess1);
    Mat::Iterator<float> steephess2_it(steephess2);
    Mat::Iterator<float> steephess3_it(steephess3);
    Mat::Iterator<float> steephess4_it(steephess4);
    Mat::Iterator<float> steephess5_it(steephess5);
    Mat::Iterator<float> steephess6_it(steephess6);
    Mat::Iterator<float> steephess7_it(steephess7);

    Mat::ConstIterator<float> steepest_it(steepest_img);

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


template <typename GradPixelType,
          typename TransformClass,
          typename SteepestPixelType>
void generate_steepest_descent_imgs(const Mat& grad_x,
                                    const Mat& grad_y,
                                    Mat& steepest_img)
{
    using TransformElementType = typename TransformClass::ElementType;
    const int transform_params = TransformClass::number_parameters;

    assert(grad_x.cols == grad_y.cols);
    assert(grad_x.rows == grad_y.rows);
    assert(grad_x.channels() == grad_y.channels());

    if (steepest_img.empty()) {
        steepest_img.create<SteepestPixelType>(
            grad_x.rows, grad_x.cols, transform_params);
    }

    assert(steepest_img.cols == grad_x.cols);
    assert(steepest_img.rows == grad_x.rows);
    assert(steepest_img.channels() == transform_params);

    Mat transform_jacobian;
    transform_jacobian.create<TransformElementType>(2, transform_params, 1);

    Mat::ConstIterator<GradPixelType> grad_x_it(grad_x);
    Mat::ConstIterator<GradPixelType> grad_y_it(grad_y);
    Mat::Iterator<TransformElementType> transform_jacob_it(transform_jacobian);
    Mat::Iterator<SteepestPixelType> steepest_it(steepest_img);

    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            GradPixelType grad_x_pixel = grad_x_it(y, x, 0);
            GradPixelType grad_y_pixel = grad_y_it(y, x, 0);

            TransformClass::jacobian_origin(
                static_cast<TransformElementType>(x),
                static_cast<TransformElementType>(y),
                transform_jacobian);

            for (int param = 0; param < transform_params; ++param) {
                steepest_it(y, x, param) =
                    grad_x_pixel * transform_jacob_it(0, param, 0) +
                    grad_y_pixel * transform_jacob_it(1, param, 0);
            }
        }
    }
}

template <typename GradPixelType,
          typename TransformClass,
          typename HessianPixelType>
void generate_transformed_img_hessian(const Mat& grad_x,
                                      const Mat& grad_y,
                                      const Mat& grad_xx,
                                      const Mat& grad_xy,
                                      const Mat& grad_yx,
                                      const Mat& grad_yy,
                                      Mat& image_hessian)
{
    using TransformElementType = typename TransformClass::ElementType;
    const int transform_params = TransformClass::number_parameters;

    // clang-format off
    using SteepestColType  = Eigen::Matrix<HessianPixelType,
                                           transform_params,
                                           1>;
    using TransformRowType = Eigen::Matrix<TransformElementType,
                                           1,
                                           transform_params>;
    using HessianMatType   = Eigen::Matrix<HessianPixelType,
                                           transform_params,
                                           transform_params,
                                           Eigen::RowMajor>;
    // clang-format on

    ///
    // Allocate output
    if (image_hessian.empty()) {
        image_hessian.create<HessianPixelType>(
            grad_x.rows, grad_x.cols, transform_params * transform_params);
    }

    ///
    // Check for input correctness
    assert(grad_x.cols == grad_y.cols && grad_x.rows == grad_y.rows);
    assert(grad_x.cols == grad_xx.cols && grad_x.rows == grad_xx.rows);
    assert(grad_x.cols == grad_xy.cols && grad_x.rows == grad_xy.rows);
    assert(grad_x.cols == grad_yx.cols && grad_x.rows == grad_yx.rows);
    assert(grad_x.cols == grad_yy.cols && grad_x.rows == grad_yy.rows);
    assert(grad_x.cols == image_hessian.cols &&
           grad_x.rows == image_hessian.rows);

    assert(grad_x.channels() == 1 && grad_y.channels() == 1);
    assert(grad_xx.channels() == 1 && grad_xy.channels() == 1);
    assert(grad_yx.channels() == 1 && grad_yy.channels() == 1);
    assert(image_hessian.channels() == transform_params * transform_params);

    ///
    // Generate second order steepest images
    Mat steepest_grad_x;
    Mat steepest_grad_y;

    generate_steepest_descent_imgs<GradPixelType,
                                   TransformClass,
                                   HessianPixelType>(
        grad_xx, grad_xy, steepest_grad_x);

    generate_steepest_descent_imgs<GradPixelType,
                                   TransformClass,
                                   HessianPixelType>(
        grad_yx, grad_yy, steepest_grad_y);

    ///
    // Allocate buffer for transform jacobian
    Mat transform_jacobian;
    transform_jacobian.create<TransformElementType>(2, transform_params, 1);

    ///
    // Allocate buffer for transform hessians
    Mat transform_hessian_x;
    Mat transform_hessian_y;
    transform_hessian_x.create<TransformElementType>(
        transform_params, transform_params, 1);
    transform_hessian_y.create<TransformElementType>(
        transform_params, transform_params, 1);

    ///
    // Create iterators
    Mat::Iterator<HessianPixelType> steepest_x_it(steepest_grad_x);
    Mat::Iterator<HessianPixelType> steepest_y_it(steepest_grad_y);
    Mat::Iterator<HessianPixelType> img_hessian_it(image_hessian);
    Mat::ConstIterator<GradPixelType> grad_x_it(grad_x);
    Mat::ConstIterator<GradPixelType> grad_y_it(grad_y);
    Mat::Iterator<TransformElementType> transf_jacobian_it(transform_jacobian);

    ///
    // Compute image hessian
    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            ///
            // Generate transform jacobian
            TransformClass::jacobian_origin(
                static_cast<TransformElementType>(x),
                static_cast<TransformElementType>(y),
                transform_jacobian);

            ///
            // Generate transform hessians
            TransformClass::hessian_x_origin(
                static_cast<TransformElementType>(x),
                static_cast<TransformElementType>(y),
                transform_hessian_x);
            TransformClass::hessian_y_origin(
                static_cast<TransformElementType>(x),
                static_cast<TransformElementType>(y),
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
                static_cast<HessianPixelType*>(transform_hessian_x.data));
            Eigen::Map<HessianMatType> transf_hessian_y_mat(
                static_cast<HessianPixelType*>(transform_hessian_y.data));

            pixel_hessian = steepest_x_col * transform_x_row +
                            steepest_y_col * transform_y_row +
                            grad_x_it(y, x, 0) * transf_hessian_x_mat +
                            grad_y_it(y, x, 0) * transf_hessian_y_mat;
        }
    }

    return;
}

void generate_mi_space(const std::string& file_name, const Mat& source)
{
    using PixelType = float;

    // Open output file
    std::ofstream file_handle;
    file_handle.open(file_name, std::ios::out);

    Mat source_small;

    ///
    // Generate the scaled-down destination image
    const float scale = 1.0f;
    // clang-format off
    std::vector<float> scale_data{
        scale, 0.f, 0.f,
        0.f, scale, 0.f,
        0.f, 0.f, 1.f
    };
    Eigen::Map<const Matrix3fRowMajor> scale_mat(scale_data.data());

    {
        Mat tmp_mask;

        image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
            source,
            scale_data.data(),
            bounding_box_transform(BoundingBox(source),
                                   scale_data.data()),
            source_small,
            tmp_mask);
    }

    /// Translation
    for (float y = 0; y <= 0; y += 0.1f) {
        for (float x = -TEST_ALPHA_RANGE; x <= TEST_ALPHA_RANGE; x += TEST_DELTA_ALPHA) {
            // clang-format off
            std::vector<float> translation_data {
                1.f, 0.f, 0.f,
                0.f, 1.f, 0.f,
                0.f, 0.f, 1.f
            };
            translation_data[TEST_ALPHA_MODEL_IDX] = x;
            // clang-format on

            Eigen::Map<const Matrix3fRowMajor> translate(
                translation_data.data());
            Matrix3fRowMajor scale_and_translate = translate * scale_mat;

            Mat transformed_mask;
            Mat transformed_img;

            BoundingBox input_bb = BoundingBox(source);

            BoundingBox output_bb = bounding_box_intersect(
                bounding_box_transform(input_bb, scale_mat.data()),
                BoundingBox(source_small));

            Mat cropped_img = image_crop<PixelType>(source_small, output_bb);
            image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
                source,
                scale_and_translate.data(),
                output_bb,
                transformed_img,
                transformed_mask);
            double mi = mutual_information<PixelType>(
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
    generate_steepest_descent_imgs<GradPixelType,
                                   HomographyTransform<float>,
                                   float>(grad_x, grad_y, steepest_img);

    visualize_steepest_descent_imgs(steepest_img);

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

void generate_mi_derivative_space(const std::string& file_name,
                                  const Mat& source,
                                  const Mat& destination)
{
    using PixelType           = float;
    using GradPixelType       = float;
    using TransformClass      = HomographyTransform<float>;
    using DerivativeMethod    = DerivativeNaive<1>;
    const int gradient_border = DerivativeMethod::border_crop_size();

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
    // clang-format off
    std::vector<float> scale_data{
        scale, 0.f, 0.f,
        0.f, scale, 0.f,
        0.f, 0.f, 1.f
    };
    // clang-format on
    Eigen::Map<const Matrix3fRowMajor> scale_mat(scale_data.data());

    Mat destination_small;
    Mat grad_x_small;
    Mat grad_y_small;
    {
        Mat tmp_mask;

        // Scale destination
        image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
            cropped_destination,
            scale_data.data(),
            bounding_box_transform(BoundingBox(cropped_destination),
                                   scale_data.data()),
            destination_small,
            tmp_mask);

        // Scale gradient
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            grad_x,
            scale_data.data(),
            bounding_box_transform(BoundingBox(grad_x), scale_data.data()),
            grad_x_small,
            tmp_mask);
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            grad_y,
            scale_data.data(),
            bounding_box_transform(BoundingBox(grad_y), scale_data.data()),
            grad_y_small,
            tmp_mask);
    }

    // Generate steepest descent image
    Mat steepest_destination;
    generate_steepest_descent_imgs<GradPixelType, TransformClass, float>(
        grad_x_small, grad_y_small, steepest_destination);

    Mat gradient;
    gradient.create<float>(1, TransformClass::number_parameters, 1);

    for (float alpha = -TEST_ALPHA_RANGE; alpha <= TEST_ALPHA_RANGE;
         alpha += TEST_DELTA_ALPHA) {
        // clang-format off
        std::vector<float> interest_transform {
            1.f, 0.f, 0.f,
            0.f, 1.f, 0.f,
            0.f, 0.f, 1.f
        };
        interest_transform[TEST_ALPHA_MODEL_IDX] = alpha;
        // clang-format on
        Eigen::Map<const Matrix3fRowMajor> interest_transform_mat(
            interest_transform.data());

        Matrix3fRowMajor homography = interest_transform_mat * scale_mat;

        BoundingBox destination_bb(cropped_destination);
        BoundingBox interest_bb = bounding_box_intersect(
            bounding_box_transform(destination_bb, homography.data()),
            BoundingBox(destination_small));

        Mat local_destination =
            image_crop<PixelType>(destination_small, interest_bb);
        Mat local_steepest =
            image_crop<float>(steepest_destination, interest_bb);

        Mat local_mask;
        Mat local_source;
        image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
            cropped_source,
            homography.data(),
            interest_bb,
            local_source,
            local_mask);

        mutual_information_gradient<HomographyTransform<float>>(
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
    using PixelType             = float;
    using GradPixelType         = float;
    using TransformClass        = HomographyTransform<float>;
    using TransformImgDerivType = float;
    using BinningMethod         = BSpline4;
    using DerivativeMethod      = DerivativeNaive<1>;

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
    // clang-format off
    std::vector<float> scale_data{
        scale, 0.f, 0.f,
        0.f, scale, 0.f,
        0.f, 0.f, 1.f
    };
    // clang-format on
    Eigen::Map<const Matrix3fRowMajor> scale_mat(scale_data.data());

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
        image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
            cropped_destination,
            scale_data.data(),
            bounding_box_transform(BoundingBox(cropped_destination),
                                   scale_data.data()),
            destination_small,
            tmp_mask);

        // Scale gradient
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            cropped_grad_x,
            scale_data.data(),
            bounding_box_transform(BoundingBox(cropped_grad_x),
                                   scale_data.data()),
            grad_x_small,
            tmp_mask);
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            cropped_grad_y,
            scale_data.data(),
            bounding_box_transform(BoundingBox(cropped_grad_y),
                                   scale_data.data()),
            grad_y_small,
            tmp_mask);

        // Scale second gradients
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            grad_xx,
            scale_data.data(),
            bounding_box_transform(BoundingBox(grad_xx), scale_data.data()),
            grad_xx_small,
            tmp_mask);
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            grad_xy,
            scale_data.data(),
            bounding_box_transform(BoundingBox(grad_xy), scale_data.data()),
            grad_xy_small,
            tmp_mask);
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            grad_yx,
            scale_data.data(),
            bounding_box_transform(BoundingBox(grad_yx), scale_data.data()),
            grad_yx_small,
            tmp_mask);
        image_transform<GradPixelType, 1, bilinear_sample<GradPixelType, 1>>(
            grad_yy,
            scale_data.data(),
            bounding_box_transform(BoundingBox(grad_yy), scale_data.data()),
            grad_yy_small,
            tmp_mask);
    }

    // Generate steepest descent image
    Mat steepest_grad_dst;
    generate_steepest_descent_imgs<GradPixelType,
                                   TransformClass,
                                   TransformImgDerivType>(
        grad_x_small, grad_y_small, steepest_grad_dst);

    // Generate transformed image hessian
    Mat steepest_hess_dst;
    generate_transformed_img_hessian<GradPixelType,
                                     TransformClass,
                                     TransformImgDerivType>(grad_x_small,
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

    for (float alpha = -TEST_ALPHA_RANGE; alpha <= TEST_ALPHA_RANGE;
         alpha += TEST_DELTA_ALPHA) {
        // clang-format off
        std::vector<float> interest_transform {
            1.f, 0.f, 0.f,
            0.f, 1.f, 0.f,
            0.f, 0.f, 1.f
        };
        interest_transform[TEST_ALPHA_MODEL_IDX] = alpha;
        // clang-format on
        Eigen::Map<const Matrix3fRowMajor> interest_transform_mat(
            interest_transform.data());

        Matrix3fRowMajor homography = interest_transform_mat * scale_mat;

        BoundingBox destination_bb(cropped_destination);
        BoundingBox interest_bb = bounding_box_intersect(
            bounding_box_transform(destination_bb, homography.data()),
            BoundingBox(destination_small));

        Mat local_destination =
            image_crop<PixelType>(destination_small, interest_bb);
        Mat local_steepest_grad =
            image_crop<float>(steepest_grad_dst, interest_bb);
        Mat local_steepest_hess =
            image_crop<float>(steepest_hess_dst, interest_bb);

        Mat local_mask;
        Mat local_source;
        image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
            cropped_source,
            homography.data(),
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
                            Mat::ConstIterator<MaskType>>(local_destination,
                                                          {},
                                                          local_steepest_grad,
                                                          local_source,
                                                          local_mask,
                                                          histogram_r_sum,
                                                          histogram_rt_sum,
                                                          histogram_r,
                                                          histogram_rt,
                                                          histogram_rt_grad);

        // Compute MI Hessian
        mutual_information_hessian<HomographyTransform<float>>(
            local_destination,
            local_steepest_grad,
            local_steepest_hess,
            local_source,
            local_mask,
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

bool register_translation(const Mat& source,
                          const Mat& destination,
                          float* translation)
{

    /*
    Mat src_sat;
    generate_sat<1>(source, src_sat);
    */

    /*
    Mat src_sat;
    generate_sat<1>(source, src_sat);
    float scale = 0.99f;

    for (int i = 0; i < 20; ++i) {
        Mat small;
        scale_from_sat<uint8_t, 3>(src_sat, scale, small);
        scale *= 0.97f;
    }
    */

    /*
    Mat src_sat;
    generate_sat<1>(source, src_sat);
    Mat small;
    scale_from_sat<uint8_t, 1>(src_sat, 0.1f, small);
    */

    /*
    using ProbClass = BSpline4;
    {
        std::vector<float> histogram;
        image_histogram<uint8_t, ProbClass>(source, histogram);
        for (const auto& i : histogram)
            std::cout << i << " ";
        std::cout << std::endl;
    }

    std::vector<float> histogram;
    joint_image_histogram<uint8_t, ProbClass>(source, source, histogram);
    float* hist_data = histogram.data();
    for (int y = 0;
         y < NUM_HISTOGRAM_CENTRAL_BINS + ProbClass::INFLUENCE_MARGIN * 2;
         ++y) {
        for (int x = 0;
             x < NUM_HISTOGRAM_CENTRAL_BINS + ProbClass::INFLUENCE_MARGIN * 2;
             ++x) {
            std::cout << *hist_data++ << " ";
        }
        std::cout << std::endl;
    }
    */

    /*
    // clang-format off
    std::vector<float> homography {
        1.f, 0.f, 10.f,
        0.f, 1.f, -10.f,
        0.0001f, 0.0001f, 1.f
    };
    // clang-format on
    Mat transformed_source;
    BoundingBox interest_bb(source);
    bounding_box_transform(homography.data(), interest_bb);
    interest_bb = bounding_box_intersect(interest_bb, BoundingBox(source));
    Mat cropped_source = image_crop<uint8_t>(source, interest_bb);
    Mat transformed_mask;
    image_transform<uint8_t, 1>(source,
                                homography.data(),
                                interest_bb,
                                transformed_source,
                                transformed_mask);

    generate_mi_space(source);
    */

    // test_bspline_4();

    // test_steepest_descent_imgs(source);

    const float scale = 0.4f;
    // clang-format off
    std::vector<float> scale_data{
        scale, 0.f, 0.f,
        0.f, scale, 0.f,
        0.f, 0.f, 1.f
    };
    // clang-format on
    Eigen::Map<const Matrix3fRowMajor> scale_mat(scale_data.data());
    Mat small_homog;
    {
        Mat tmp_mask;
        image_transform<uint8_t, 1, bilinear_sample<uint8_t, 1>>(
            source,
            scale_mat.data(),
            bounding_box_transform(BoundingBox(source), scale_mat.data()),
            small_homog,
            tmp_mask);
    }

    Mat source_blurred;
    // gaussian_blur<uint8_t, uint8_t, 1>(source, 5, 2.f, source_blurred);

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

    gaussian_blur<uint8_t, uint8_t, 1>(small_homog, 5, 2.f, source_blurred);

    // Preprocess image
    Mat blurred_normalized =
        image_remap_histogram<uint8_t, float, PositiveMaskIterator>(
            source_blurred, {});

    // test_image_derivative(source);

    generate_mi_space("mi.txt", blurred_normalized);
    generate_mi_derivative_space(
        "dmi.txt", blurred_normalized, blurred_normalized);
    generate_mi_hessian_space(
        "d2mi.txt", blurred_normalized, blurred_normalized);

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

    return true;
}
