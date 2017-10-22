#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Eigen>

#include "bounding_box.h"
#include "histogram.h"
#include "interpolation.h"
#include "mat.h"
#include "math.h"
#include "mutual_information.h"
#include "registration.h"
#include "sat.h"

using Matrix3fRowMajor = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

template <typename PixelType>
Mat image_crop(const Mat& image, const BoundingBox& crop_bb)
{
    assert(image.type() == Mat::get_type_enum<PixelType>());

    Mat output;

    assert(crop_bb.left_top[0] >= 0.f);
    assert(crop_bb.left_top[1] >= 0.f);

    assert(crop_bb.left_top[0] <= crop_bb.right_bottom[0]);
    assert(crop_bb.left_top[1] <= crop_bb.right_bottom[1]);

    assert(crop_bb.left_top[0] <= crop_bb.right_bottom[0]);
    assert(crop_bb.left_top[1] <= crop_bb.right_bottom[1]);

    output.create_from_buffer<PixelType>(
        static_cast<PixelType*>(image.data) +
            static_cast<int>(crop_bb.left_top[1]) * image.row_stride() +
            static_cast<int>(crop_bb.left_top[0]) * image.channels(),
        crop_bb.flooring_height(),
        crop_bb.flooring_width(),
        image.channels(),
        image.row_stride());
    output.data_mgr_ = image.data_mgr_;

    return output;
}

template <typename PixelType>
using InterpolationFunctor =
    void (*)(const Mat::ConstIterator<PixelType>& it_img,
             const float* coordinates,
             PixelType* output);

template <typename PixelType,
          int channels,
          InterpolationFunctor<PixelType> interpolation_method>
void image_transform(const Mat& image,
                     const float* homography_ptr,
                     const BoundingBox& output_bb,
                     Mat& output_image,
                     Mat& output_mask)
{
    using Eigen::Vector2f;
    using Eigen::Vector3f;
    using std::vector;

    assert(image.channels() == channels);

    // Compute bounding box of the transformed image by transforming its corners
    Eigen::Map<const Matrix3fRowMajor> homography(homography_ptr);

    int output_width  = output_bb.flooring_width();
    int output_height = output_bb.flooring_height();

    output_image.create<PixelType>(output_height, output_width, channels);
    output_mask.create<uint8_t>(output_height, output_width, 1);
    memset(output_image.data, 0, output_width * output_height * channels);

    Matrix3fRowMajor homography_inv = homography.inverse();

    // Converts from output space to transformed bounding box space
    Matrix3fRowMajor transf_bb_pivot;
    // clang-format off
    transf_bb_pivot << 1.f, 0.f, std::floor(output_bb.left_top[0]),
                       0.f, 1.f, std::floor(output_bb.left_top[1]),
                       0.f, 0.f, 1.f;
    // clang-format on
    homography_inv = homography_inv * transf_bb_pivot;

    Mat::ConstIterator<PixelType> it_img(image);
    Mat::Iterator<PixelType> it_transf_img(output_image);
    Mat::Iterator<uint8_t> it_mask(output_mask);

    float last_input_col = static_cast<float>(image.cols) - 1.f;
    float last_input_row = static_cast<float>(image.rows) - 1.f;

    for (int y_buff = 0; y_buff < output_image.rows; ++y_buff) {
        for (int x_buff = 0; x_buff < output_image.cols; ++x_buff) {
            Vector3f transformed_coord =
                homography_inv * Vector3f(static_cast<float>(x_buff),
                                          static_cast<float>(y_buff),
                                          1.f);
            // Normalize homogeneous coordinates
            transformed_coord /= transformed_coord[2];

            if (transformed_coord[0] >= 0.f &&
                transformed_coord[0] <= last_input_col &&
                transformed_coord[1] >= 0.f &&
                transformed_coord[1] <= last_input_row) {

                interpolation_method(it_img,
                                     transformed_coord.data(),
                                     &it_transf_img(y_buff, x_buff, 0));

                it_mask(y_buff, x_buff, 0) = 255;
            } else {
                it_mask(y_buff, x_buff, 0) = 0;
            }
        }
    }

    return;
}

/**
 * Separable kernel convolution
 */
template <typename InputPixelType, typename OutputPixelType, int channels>
void image_convolve(const Mat& image,
                    const Mat& kernel_v,
                    const Mat& kernel_h,
                    const float kernel_norm_factor,
                    Mat& output_image)
{
    assert(kernel_v.cols % 2 == 1);
    assert(kernel_v.rows == 1);
    assert(kernel_h.cols % 2 == 1);
    assert(kernel_h.rows == 1);

    assert(kernel_v.type() == Mat::Type::FLOAT32);
    assert(kernel_h.type() == Mat::Type::FLOAT32);

    int border_v = (kernel_v.cols - 1) / 2;
    int border_h = (kernel_h.cols - 1) / 2;

    Mat::ConstIterator<float> kernel_v_it(kernel_v);
    Mat::ConstIterator<float> kernel_h_it(kernel_h);

    // Create output image for first pass
    Mat output_vertical_pass;
    output_vertical_pass.create<OutputPixelType>(
        image.rows - 2 * border_v, image.cols, channels);

    // Create output image for second pass
    output_image.create<OutputPixelType>(
        image.rows - 2 * border_v, image.cols - 2 * border_h, channels);

    // Convolution core
    enum class ConvolutionDirection { Horizontal, Vertical };
    const auto convolve = [](const ConvolutionDirection direction,
                             const Mat::ConstIterator<float>& kernel_it,
                             const float norm_factor,
                             const auto& input_it,
                             Mat::Iterator<OutputPixelType>& output_it) {
        for (int y = 0; y < output_it.m.rows; ++y) {
            for (int x = 0; x < output_it.m.cols; ++x) {
                for (int c = 0; c < output_it.m.channels(); ++c) {
                    float conv_sum = 0.f;

                    for (int n = 0; n < kernel_it.m.cols; ++n) {
                        auto input_pix =
                            (direction == ConvolutionDirection::Vertical)
                                ? input_it(y + n, x, c)
                                : input_it(y, x + n, c);

                        conv_sum += kernel_it(0, n, 0) * input_pix;
                    }

                    float normalized_pix = conv_sum * norm_factor;
                    if (std::is_integral<OutputPixelType>::value) {
                        normalized_pix = std::round(normalized_pix);
                    }

                    output_it(y, x, c) =
                        static_cast<OutputPixelType>(normalized_pix);
                }
            }
        }
    };

    // First pass: Vertical convolution
    Mat::Iterator<OutputPixelType> first_pass_it(output_vertical_pass);
    convolve(ConvolutionDirection::Vertical,
             kernel_v_it,
             1.f,
             Mat::ConstIterator<InputPixelType>(image),
             first_pass_it);

    // Second pass: Horizontal convolution
    Mat::Iterator<OutputPixelType> second_pass_it(output_image);
    convolve(ConvolutionDirection::Horizontal,
             kernel_h_it,
             kernel_norm_factor,
             Mat::ConstIterator<OutputPixelType>(output_vertical_pass),
             second_pass_it);

    return;
}

enum class ImageDerivativeAxis { dX, dY };
template <typename InputPixelType, typename OutputPixelType, int channels>
void derivative_holoborodko_impl(
    const Mat& image,
    ImageDerivativeAxis axis,
    const std::initializer_list<float>& high_pass_component,
    const std::initializer_list<float>& low_pass_component,
    const float norm_factor,
    Mat& output_image)
{
    assert(std::is_signed<OutputPixelType>::value);
    assert(high_pass_component.size() == low_pass_component.size());

    Mat horizontal_kernel;
    Mat vertical_kernel;

    const int kernel_length = static_cast<int>(high_pass_component.size());

    if (axis == ImageDerivativeAxis::dX) {
        horizontal_kernel.create<float>(
            1, kernel_length, 1, high_pass_component);

        vertical_kernel.create<float>(1, kernel_length, 1, low_pass_component);
    } else {
        assert(axis == ImageDerivativeAxis::dY);

        horizontal_kernel.create<float>(
            1, kernel_length, 1, low_pass_component);

        vertical_kernel.create<float>(1, kernel_length, 1, high_pass_component);
    }

    image_convolve<InputPixelType, OutputPixelType, channels>(
        image, vertical_kernel, horizontal_kernel, norm_factor, output_image);
}

enum class FilterOrder { Fifth, Seventh };
template <typename InputPixelType, typename OutputPixelType, int channels>
void derivative_holoborodko(const Mat& image,
                            ImageDerivativeAxis axis,
                            FilterOrder filter_order,
                            Mat& output_image)
{
    // Sample kernels:
    // [1 2 1]' * [-1 -2 0 2 1], border: 1 row, 2 cols
    // [1 4 6 4 1]' * [-1 -4 -5 0 5 4 1], border: 2 rows, 3 cols

    if (filter_order == FilterOrder::Fifth) {
        derivative_holoborodko_impl<InputPixelType, OutputPixelType, channels>(
            image,
            axis,
            {-1, -2, 0, 2, 1},
            {1, 1, 2, 1, 1},
            1.f / 24.f,
            output_image);
    } else {
        assert(filter_order == FilterOrder::Seventh);

        derivative_holoborodko_impl<InputPixelType, OutputPixelType, channels>(
            image,
            axis,
            {-1, -4, -5, 0, 5, 4, 1},
            {1, 1, 4, 6, 4, 1, 1},
            1.f / 200.f,
            output_image);
    }
}

void generate_mi_space(const Mat& source)
{
    using PixelType = uint8_t;

    Mat src_sat;
    const float scale = 0.3f;
    generate_sat<1>(source, src_sat);
    Mat small;
    scale_from_sat<PixelType, 1>(src_sat, scale, small);

    // clang-format off
    std::vector<float> scale_data {
        scale, 0.f, 0.f,
        0.f, scale, 0.f,
        0.f, 0.f, 1.f
    };
    // clang-format on
    Eigen::Map<const Matrix3fRowMajor> scale_mat(scale_data.data());
    Mat small_homog;
    {
        Mat tmp_mask;
        image_transform<PixelType, 1, bilinear_sample<PixelType, 1>>(
            source,
            scale_mat.data(),
            bounding_box_transform(BoundingBox(source), scale_mat.data()),
            small_homog,
            tmp_mask);
    }

    /// Translation
    const float dt = 10.f;
    for (float y = -dt; y <= dt; y += 0.1f) {
        for (float x = -dt; x <= dt; x += 0.1f) {

            // clang-format off
            std::vector<float> translation_data {
                1.f, 0.f, x,
                0.f, 1.f, y,
                0.f, 0.f, 1.f
            };
            // clang-format on
            Eigen::Map<const Matrix3fRowMajor> translate(
                translation_data.data());
            Matrix3fRowMajor scale_and_translate = translate * scale_mat;

            Mat transformed_mask;
            Mat transformed_img;

            BoundingBox input_bb = BoundingBox(small_homog);

            BoundingBox output_bb = bounding_box_intersect(
                bounding_box_transform(input_bb, translate.data()), input_bb);
            Mat cropped_img = image_crop<PixelType>(small_homog, output_bb);
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
    test_image_derivative(source);

    return true;
}
