#include <iostream>
#include <limits>
#include <vector>

#include "bounding_box.h"
#include "histogram.h"
#include "interpolation.h"
#include "mat.h"
#include "math.h"
#include "mutual_information.h"
#include "registration.h"
#include "sat.h"
#include "transform.h"

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
    } else {
        assert(steepest_img.cols == grad_x.cols);
        assert(steepest_img.rows == grad_x.rows);
        assert(steepest_img.channels() == transform_params);
    }

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

void generate_mi_space(const Mat& source)
{
    using PixelType = float;

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
    const float dt = 0.0001f;
    for (float y = 0; y <= 0; y += 0.1f) {
        for (float x = -dt; x <= dt; x += 0.000001f) {
            const int alpha_pos = 6;

            // clang-format off
            std::vector<float> translation_data {
                1.f, 0.f, 0.f,
                0.f, 1.f, 0.f,
                0.f, 0.f, 1.f
            };
            translation_data[alpha_pos] = x;
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

            std::cout << mi << " ";
        }
        std::cout << std::endl;
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

    return;
}

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

void test_steepest_descent_imgs(const Mat& source)
{
    using GradPixelType = float;

    Mat grad_x;
    Mat grad_y;
    derivative_holoborodko<uint8_t, GradPixelType, 1>(
        source, ImageDerivativeAxis::dX, FilterOrder::Fifth, grad_x);

    derivative_holoborodko<uint8_t, GradPixelType, 1>(
        source, ImageDerivativeAxis::dY, FilterOrder::Fifth, grad_y);

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

    derivative_holoborodko<uint8_t, float, 1>(
        source, ImageDerivativeAxis::dX, FilterOrder::Fifth, dx);

    derivative_holoborodko<uint8_t, float, 1>(
        source, ImageDerivativeAxis::dY, FilterOrder::Fifth, dy);

    derivative_holoborodko<float, float, 1>(
        dx, ImageDerivativeAxis::dY, FilterOrder::Fifth, dxy);

    Mat cropped = image_crop<uint8_t>(
        source,
        BoundingBox({{{2, 2}},
                     {{static_cast<float>(source.cols) - 3,
                       static_cast<float>(source.rows - 3)}}}));

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

void generate_mi_derivative_space(const Mat& source, const Mat& destination)
{
    using PixelType      = float;
    using GradPixelType  = float;
    using TransformClass = HomographyTransform<float>;

    // Generate image derivatives
    Mat grad_x;
    Mat grad_y;

    const int gradient_border = 2;
    derivative_holoborodko<PixelType, GradPixelType, 1>(
        destination, ImageDerivativeAxis::dX, FilterOrder::Fifth, grad_x);

    derivative_holoborodko<PixelType, GradPixelType, 1>(
        destination, ImageDerivativeAxis::dY, FilterOrder::Fifth, grad_y);

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

    const float dt = 0.0001f;
    for (float alpha = dt; alpha <= dt; alpha += 0.000001f) {
        const int alpha_pos = 6;
        // clang-format off
        std::vector<float> interest_transform {
            1.f, 0.f, 0.f,
            0.f, 1.f, 0.f,
            0.f, 0.f, 1.f
        };
        interest_transform[alpha_pos] = alpha;
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

        // visualize_steepest_descent_imgs(steepest_destination);
        // visualize_steepest_descent_imgs(local_steepest);

        mutual_information_gradient<HomographyTransform<float>>(
            local_destination,
            local_steepest,
            local_source,
            local_mask,
            gradient);

        Mat::Iterator<float> grad_it(gradient);
        std::cout << alpha << " " << grad_it(0, alpha_pos, 0) << std::endl;

        int breakpoint = 0;
        breakpoint++;
    }

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

    // test_image_derivative(source);

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
    gaussian_blur<uint8_t, uint8_t, 1>(small_homog, 5, 2.f, source_blurred);

    /*
    small_homog_blurred.for_each<Mat::Iterator<uint8_t>>(
        [](Mat::Iterator<uint8_t>& it, int y, int x, int c) {
            int qx = (int)(x < it.m.cols / 2 ? 0 : 255);
            int qy = (int)(y < it.m.rows / 2 ? 0 : 255);
            if (x > it.m.cols * 0.25 && x < it.m.cols * 0.75 &&
                y > it.m.rows * 0.25 && y < it.m.rows * 0.75) {
                it(y, x, c) = (uint8_t)(qx + qy > 255 ? 0 : qx + qy);
            }
        });
        */
    // Preprocess image
    Mat blurred_normalized = image_scale_histogram<uint8_t, float>(source_blurred);

    generate_mi_derivative_space(blurred_normalized, blurred_normalized);
    std::cout << std::endl;
    generate_mi_space(blurred_normalized);

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
