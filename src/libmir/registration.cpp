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
    void (*)(const Mat::ConstIterator<PixelType>& img_it,
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

    Mat::ConstIterator<PixelType> img_it(image);
    Mat::Iterator<PixelType> transf_img_it(output_image);
    Mat::Iterator<uint8_t> mask_it(output_mask);

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

                interpolation_method(img_it,
                                     transformed_coord.data(),
                                     &transf_img_it(y_buff, x_buff, 0));

                mask_it(y_buff, x_buff, 0) = 255;
            } else {
                mask_it(y_buff, x_buff, 0) = 0;
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
    using IntermediatePixelType = decltype(InputPixelType() * InputPixelType());

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
    output_vertical_pass.create<IntermediatePixelType>(
        image.rows - 2 * border_v, image.cols, channels);

    // Create output image for second pass
    output_image.create<OutputPixelType>(
        image.rows - 2 * border_v, image.cols - 2 * border_h, channels);

    // Convolution core
    enum class ConvolutionDirection { Horizontal, Vertical };
    const auto convolve = [](const ConvolutionDirection direction,
                             const Mat::ConstIterator<float>& kernel_it,
                             const float norm_factor,
                             const bool clamp_output,
                             const auto& input_it,
                             auto& output_it) {
        for (int y = 0; y < output_it.m.rows; ++y) {
            for (int x = 0; x < output_it.m.cols; ++x) {
                for (int c = 0; c < output_it.m.channels(); ++c) {
                    float conv_sum = 0.f;

                    for (int n = 0; n < kernel_it.m.cols; ++n) {
                        auto input_pix =
                            (direction == ConvolutionDirection::Vertical)
                                ? input_it(y + n, x, c)
                                : input_it(y, x + n, c);

                        conv_sum +=
                            kernel_it(0, n, 0) * static_cast<float>(input_pix);
                    }

                    float normalized_pix = conv_sum * norm_factor;
                    if (std::is_integral<OutputPixelType>::value) {
                        normalized_pix = std::round(normalized_pix);
                    }

                    if (clamp_output) {
                        if (std::is_same<OutputPixelType, uint8_t>::value) {
                            normalized_pix =
                                std::min(255.f, std::max(0.f, normalized_pix));
                        }
                    }

                    output_it(y, x, c) =
                        static_cast<typename std::remove_reference<decltype(
                            output_it(y, x, c))>::type>(normalized_pix);
                }
            }
        }
    };

    // First pass: Vertical convolution
    Mat::Iterator<IntermediatePixelType> first_pass_it(output_vertical_pass);
    convolve(ConvolutionDirection::Vertical,
             kernel_v_it,
             1.f,
             false,
             Mat::ConstIterator<InputPixelType>(image),
             first_pass_it);

    // Second pass: Horizontal convolution
    Mat::Iterator<OutputPixelType> second_pass_it(output_image);
    convolve(ConvolutionDirection::Horizontal,
             kernel_h_it,
             kernel_norm_factor,
             true,
             Mat::ConstIterator<IntermediatePixelType>(output_vertical_pass),
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

    horizontal_kernel.create<float>(1, kernel_length, 1);
    vertical_kernel.create<float>(1, kernel_length, 1);

    if (axis == ImageDerivativeAxis::dX) {
        horizontal_kernel << high_pass_component;
        vertical_kernel << low_pass_component;
    } else {
        assert(axis == ImageDerivativeAxis::dY);

        horizontal_kernel << low_pass_component;
        vertical_kernel << high_pass_component;
    }

    image_convolve<InputPixelType, OutputPixelType, channels>(
        image, vertical_kernel, horizontal_kernel, norm_factor, output_image);
}

template <typename InputPixelType, typename OutputPixelType, int channels>
void gaussian_blur(const Mat& image,
                   int kernel_border_size,
                   float standard_deviation,
                   Mat& output_image)
{
    Mat kernel;
    kernel.create<float>(1, 2 * kernel_border_size + 1, 1);
    Mat::Iterator<float> kernel_it(kernel);

    float kernel_summation = 0.f;
    for (int i = 0; i < kernel.cols; ++i) {
        float x =
            static_cast<float>(i - kernel_border_size) / standard_deviation;
        float current_value = std::exp(-(x * x) * 0.5f);
        kernel_it(0, i, 0)  = current_value;
        kernel_summation += current_value;
    }

    float norm_factor = 1.f / (kernel_summation * kernel_summation);
    image_convolve<InputPixelType, OutputPixelType, channels>(
        image, kernel, kernel, norm_factor, output_image);

    return;
}

enum class FilterOrder { Fifth, Seventh };
/**
 * Compute the image derivative along the specified axis.
 *
 * Note that the output image borders WILL BE CROPPED by an amount proportional
 * to the chosen filter order.
 */
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
            1.f / 32.f,
            output_image);
    } else {
        assert(filter_order == FilterOrder::Seventh);

        derivative_holoborodko_impl<InputPixelType, OutputPixelType, channels>(
            image,
            axis,
            {-1, -4, -5, 0, 5, 4, 1},
            {1, 1, 4, 6, 4, 1, 1},
            1.f / 512.f,
            output_image);
    }
}

void generate_mi_space(const Mat& source)
{
    using PixelType = uint8_t;

    /// Translation
    const float dt = 20.f;
    for (float y = 0; y <= 0; y += 0.1f) {
        for (float x = -dt; x <= dt; x += 0.05f) {

            // clang-format off
            std::vector<float> translation_data {
                1.f, 0.f, x,
                0.f, 1.f, y,
                0.f, 0.f, 1.f
            };
            // clang-format on
            Eigen::Map<const Matrix3fRowMajor> translate(
                translation_data.data());
            Matrix3fRowMajor scale_and_translate = translate;

            Mat transformed_mask;
            Mat transformed_img;

            BoundingBox input_bb = BoundingBox(source);

            BoundingBox output_bb = bounding_box_intersect(
                bounding_box_transform(input_bb, translate.data()), input_bb);
            Mat cropped_img = image_crop<PixelType>(source, output_bb);
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

/**
 * Jacobian of the Affine transform w(coordinate, Dp) evaluated at Dp=0
 */
template <typename T>
void affine_jacobian_origin(const T* coordinate, Mat& output)
{
    if (output.empty()) {
        output.create<T>(2, 6, 1);
    }

    const T& x = coordinate[0];
    const T& y = coordinate[1];

    // clang-format off
    output << std::initializer_list<T>{
        x, y, static_cast<T>(1.0), 0, 0, 0,
        0, 0, 0, x, y, static_cast<T>(1.0)
    };
    // clang-format on
}

template <typename TransformElementType>
struct HomographyTransform
{
    using ElementType                  = TransformElementType;
    static const int number_parameters = 8;

    /**
     * Jacobian of the Homography transform w(coordinate, Dp) evaluated at Dp=0
     */
    static void
    homography_jacobian_origin(ElementType x, ElementType y, Mat& output)
    {
        assert(!output.empty());

        const ElementType xy = x * y;

        // clang-format off
        output << std::initializer_list<ElementType>{
            x, y, static_cast<ElementType>(1.0), 0, 0, 0, -x * x, -xy,
            0, 0, 0, x, y, static_cast<ElementType>(1.0), -xy, -y * y
        };
        // clang-format on
    }

    void homography_hessian_x_origin(ElementType x, ElementType y, Mat& output)
    {
        assert(!output.empty());

        const ElementType xx = x * x;
        const ElementType yy = y * y;
        const ElementType xy = x * y;

        // clang-format off
        output << std::initializer_list<ElementType>{
            0,     0,  0, 0, 0, 0,        -xx,        -xy,
            0,     0,  0, 0, 0, 0,        -xy,        -yy,
            0,     0,  0, 0, 0, 0,         -x,         -y,
            0,     0,  0, 0, 0, 0,          0,          0,
            0,     0,  0, 0, 0, 0,          0,          0,
            0,     0,  0, 0, 0, 0,          0,          0,
            -xx, -xy, -x, 0, 0, 0, 2 * xx * x, 2 * xy * x,
            -xy, -yy, -y, 0, 0, 0, 2 * xx * y, 2 * xy * y,
        };
        // clang-format on
    }

    void homography_hessian_y_origin(ElementType x, ElementType y, Mat& output)
    {
        assert(!output.empty());

        const ElementType xx = x * x;
        const ElementType yy = y * y;
        const ElementType xy = x * y;

        // clang-format off
        output << std::initializer_list<ElementType>{
            0, 0, 0,   0,   0,  0,          0,          0,
            0, 0, 0,   0,   0,  0,          0,          0,
            0, 0, 0,   0,   0,  0,          0,          0,
            0, 0, 0,   0,   0,  0,        -xx,        -xy,
            0, 0, 0,   0,   0,  0,        -xy,        -yy,
            0, 0, 0,   0,   0,  0,         -x,         -y,
            0, 0, 0, -xx, -xy, -x, 2 * xy * x, 2 * yy * x,
            0, 0, 0, -xy, -yy, -y, 2 * xy * y, 2 * yy * y,
        };
        // clang-format on
    }
};

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

            TransformClass::homography_jacobian_origin(
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

void mutual_information_gradient(const Mat& reference,
                                 const Mat& steepest_img,
                                 const Mat& tracked,
                                 const Mat& tracked_mask,
                                 Mat& gradient)
{
    using std::vector;
    using BinningMethod  = BSpline4;
    using PixelType      = uint8_t;
    using TransformClass = HomographyTransform<float>;
    using GradientType   = float;

    assert(reference.rows == tracked.rows);
    assert(reference.cols == tracked.cols);

    if (gradient.empty()) {
        gradient.create<GradientType>(1, TransformClass::number_parameters, 1);
    } else {
        assert(gradient.rows == 1);
        assert(gradient.cols == TransformClass::number_parameters);
        assert(gradient.channels() == 1);
    }

    gradient.fill<GradientType>(0);

    Mat histogram_r;
    Mat histogram_rt;
    Mat histogram_rt_grad;

    joint_hist_gradient<PixelType,
                        float,
                        BinningMethod,
                        PositiveMaskIterator,
                        Mat::ConstIterator<uint8_t>>(reference,
                                                     {},
                                                     steepest_img,
                                                     tracked,
                                                     tracked_mask,
                                                     histogram_r,
                                                     histogram_rt,
                                                     histogram_rt_grad);

    const int number_practical_bins = static_cast<int>(histogram_r.cols);
    const int number_parameters     = histogram_rt_grad.channels();

    Mat::ConstIterator<float> hist_r_it(histogram_r);
    Mat::ConstIterator<float> hist_rt_it(histogram_rt);
    Mat::ConstIterator<float> hist_rt_grad_it(histogram_rt_grad);

    Mat::Iterator<float> gradient_it(gradient);

    for (int i = 0; i < number_practical_bins; ++i) {
        for (int j = 0; j < number_practical_bins; ++j) {
            for (int param = 0; param < number_parameters; ++param) {
                float grad_at_ij = hist_rt_grad_it(i, j, param);
                float hist_at_ij = hist_rt_it(i, j, 0);
                float hist_at_j  = hist_r_it(0, j, 0);

                assert(hist_at_ij <= hist_at_j);

                if (hist_at_ij > 0.f) {
                    gradient_it(0, param, 0) +=
                        grad_at_ij * std::log(hist_at_ij / hist_at_j);
                } else {
                    assert(hist_at_ij == 0.0f);
                }
            }
        }
    }

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
    using PixelType      = uint8_t;
    using GradPixelType  = int16_t;
    using TransformClass = HomographyTransform<float>;

    Mat grad_x;
    Mat grad_y;

    derivative_holoborodko<PixelType, GradPixelType, 1>(
        destination, ImageDerivativeAxis::dX, FilterOrder::Fifth, grad_x);

    derivative_holoborodko<PixelType, GradPixelType, 1>(
        destination, ImageDerivativeAxis::dY, FilterOrder::Fifth, grad_y);

    Mat steepest_destination;
    generate_steepest_descent_imgs<GradPixelType, TransformClass, float>(
        grad_x, grad_y, steepest_destination);

    BoundingBox border_bb{{{2, 2}},
                          {{static_cast<float>(destination.cols - 3),
                            static_cast<float>(destination.rows - 3)}}};

    Mat cropped_destination = image_crop<PixelType>(destination, border_bb);
    Mat cropped_source      = image_crop<PixelType>(source, border_bb);

    Mat gradient;
    gradient.create<float>(1, 8, 1);

    for (float alpha = -20.f; alpha <= 20.f; alpha += 0.05f) {
        const int alpha_pos = 2;
        // clang-format off
        std::vector<float> homography {
            1.f, 0.f, 0.f,
            0.f, 1.f, 0.f,
            0.f, 0.f, 1.f
        };
        homography[alpha_pos] = alpha;
        // clang-format on

        BoundingBox destination_bb(cropped_destination);
        BoundingBox interest_bb = bounding_box_intersect(
            bounding_box_transform(destination_bb, homography.data()),
            destination_bb);

        Mat local_destination =
            image_crop<PixelType>(cropped_destination, interest_bb);
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

        //visualize_steepest_descent_imgs(steepest_destination);
        visualize_steepest_descent_imgs(local_steepest);

        mutual_information_gradient(local_destination,
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

    float scale = 0.4f;
    std::vector<float> scale_data{
        scale, 0.f, 0.f, 0.f, scale, 0.f, 0.f, 0.f, 1.f};
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

    Mat small_homog_blurred;
    gaussian_blur<uint8_t, uint8_t, 1>(
        small_homog, 5, 2.f, small_homog_blurred);

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

    generate_mi_derivative_space(small_homog_blurred, small_homog_blurred);
    std::cout << std::endl;
    generate_mi_space(small_homog_blurred);

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
