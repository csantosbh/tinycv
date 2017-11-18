#ifndef _LIBMIR_TRANSFORM_H_
#define _LIBMIR_TRANSFORM_H_

#include <utility>

#include <Eigen/Eigen>

#include "mat.h"

using Matrix3fRowMajor = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

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

    int output_width  = output_bb.ceiling_width();
    int output_height = output_bb.ceiling_height();

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

    float in_image_cols_f = static_cast<float>(image.cols);
    float in_image_rows_f = static_cast<float>(image.rows);

    float last_input_col = in_image_cols_f - 1.f;
    float last_input_row = in_image_rows_f - 1.f;

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
            } else if (transformed_coord[0] > -1.f &&
                       transformed_coord[0] < in_image_cols_f &&
                       transformed_coord[1] > -1.f &&
                       transformed_coord[1] < in_image_rows_f) {
                ///
                // Handle image borders by smoothly decreasing the value of the
                // mask at those regions
                float clamped_coords[] = {
                    clamp(transformed_coord[0], 0.f, last_input_col),
                    clamp(transformed_coord[1], 0.f, last_input_row)};
                interpolation_method(
                    img_it, clamped_coords, &transf_img_it(y_buff, x_buff, 0));

                float mask_alpha_x = 1.f;
                float mask_alpha_y = 1.f;

                if (transformed_coord[0] < 0.f) {
                    mask_alpha_x = 1.f + transformed_coord[0];
                } else if (transformed_coord[0] > last_input_col) {
                    mask_alpha_x = in_image_cols_f - transformed_coord[0];
                }

                if (transformed_coord[1] < 0.f) {
                    mask_alpha_y = 1.f + transformed_coord[1];
                } else if (transformed_coord[1] > last_input_row) {
                    mask_alpha_y = in_image_rows_f - transformed_coord[1];
                }

                mask_it(y_buff, x_buff, 0) =
                    fast_positive_round<float, uint8_t>(255.f * mask_alpha_x *
                                                        mask_alpha_y);
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

    if(axis==ImageDerivativeAxis::dX) {
        Mat::ConstIterator<InputPixelType> inp_it(image);
        output_image.for_each<Mat::Iterator<OutputPixelType>>(
         [&inp_it]
         (Mat::Iterator<OutputPixelType>& it, int y, int x, int c) {
            int xNext = 2+x+1;
            int xPrev = 2+x-1;
            it(y, x, c) = (OutputPixelType)(inp_it(y, xNext, c) - inp_it(y, xPrev, c))/(OutputPixelType)2;
        });
    } else {
        Mat::ConstIterator<InputPixelType> inp_it(image);
        output_image.for_each<Mat::Iterator<OutputPixelType>>(
         [&inp_it]
         (Mat::Iterator<OutputPixelType>& it, int y, int x, int c) {
            int yNext = 2+y+1;
            int yPrev = 2+y-1;
            it(y, x, c) = (OutputPixelType)(inp_it(yNext, x, c) - inp_it(yPrev, x, c))/(OutputPixelType)2;
        });
    }
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

template <typename TransformElementType>
struct AffineTransform
{
    using ElementType = TransformElementType;

    void jacobian_origin(const ElementType x, ElementType y, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(2, 6, 1);
        }

        // clang-format off
        output << std::initializer_list<ElementType>{
            x, y, static_cast<ElementType>(1.0), 0, 0, 0,
            0, 0, 0, x, y, static_cast<ElementType>(1.0)
        };
        // clang-format on
    }
};

template <typename TransformElementType>
struct HomographyTransform
{
    using ElementType                  = TransformElementType;
    static const int number_parameters = 8;

    /**
     * Jacobian of the Homography transform w(coordinate, Dp) evaluated at Dp=0
     */
    static void jacobian_origin(ElementType x, ElementType y, Mat& output)
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

    static void hessian_x_origin(ElementType x, ElementType y, Mat& output)
    {
        assert(!output.empty());

        assert(output.cols == number_parameters);
        assert(output.rows == number_parameters);
        assert(output.channels() == 1);

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

    static void hessian_y_origin(ElementType x, ElementType y, Mat& output)
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

#endif
