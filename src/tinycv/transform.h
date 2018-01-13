#ifndef _TINYCV_TRANSFORM_H_
#define _TINYCV_TRANSFORM_H_

#include <utility>

#include "third_party/eigen/Eigen"

#include "bounding_box.h"
#include "mat.h"
#include "math.h"

using Matrix3fRowMajor = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

template <typename PixelType>
using InterpolationFunctor =
    void (*)(const Mat::ConstIterator<PixelType>& img_it,
             const float* coordinates,
             PixelType* output);

template <typename PixelType,
          int channels,
          typename TransformClass,
          InterpolationFunctor<PixelType> interpolation_method>
void image_transform(const Mat& image,
                     const Mat& transform_parameters,
                     const BoundingBox& output_bb,
                     Mat& output_image,
                     Mat& output_mask)
{
    using Eigen::Vector2f;
    using Eigen::Vector3f;
    using std::vector;

    assert(image.channels() == channels);

    // Compute bounding box of the transformed image by transforming its corners
    int output_width  = output_bb.ceiling_width();
    int output_height = output_bb.ceiling_height();

    // TODO use MaskType instead of uint8_t
    // TODO only create if it doesn't exist
    // Create and initialize output image
    output_image.create<PixelType>(output_height, output_width, channels);
    output_mask.create<uint8_t>(output_height, output_width, 1);
#ifndef NDEBUG
    output_image.fill<PixelType>(0);
#endif

    Mat transform_params_inv;
    TransformClass::inverse(transform_parameters, transform_params_inv);

    // Transform from destination coordinates to transformed world coordinates,
    // then back to source coordinates

    Mat composed_transform;
    // clang-format off
    TransformClass::compose(
        transform_params_inv, Point<float>{
            std::floor(output_bb.left_top[0]),
            std::floor(output_bb.left_top[1])
    }, composed_transform);
    // clang-format on

    // Create image iterators
    Mat::ConstIterator<PixelType> img_it(image);
    Mat::Iterator<PixelType> transf_img_it(output_image);
    Mat::Iterator<uint8_t> mask_it(output_mask);

    // Image limits
    const float in_image_cols_f = static_cast<float>(image.cols);
    const float in_image_rows_f = static_cast<float>(image.rows);

    const float last_input_col = in_image_cols_f - 1.f;
    const float last_input_row = in_image_rows_f - 1.f;

    for (int y_buff = 0; y_buff < output_image.rows; ++y_buff) {
        for (int x_buff = 0; x_buff < output_image.cols; ++x_buff) {
            // TODO use asserts
            // Transform coordinates to find source interpolation coords
            Point<float> transformed_coord = TransformClass::transform(
                {static_cast<float>(x_buff), static_cast<float>(y_buff)},
                composed_transform);

            if (transformed_coord.x >= 0.f &&
                transformed_coord.x <= last_input_col &&
                transformed_coord.y >= 0.f &&
                transformed_coord.y <= last_input_row) {

                interpolation_method(img_it,
                                     transformed_coord.ptr(),
                                     &transf_img_it(y_buff, x_buff, 0));

                mask_it(y_buff, x_buff, 0) = 255;
            } else if (transformed_coord.x > -1.f &&
                       transformed_coord.x < in_image_cols_f &&
                       transformed_coord.y > -1.f &&
                       transformed_coord.y < in_image_rows_f) {
                ///
                // Handle image borders by smoothly decreasing the value of the
                // mask at those regions
                float clamped_coords[] = {
                    clamp(transformed_coord.x, 0.f, last_input_col),
                    clamp(transformed_coord.y, 0.f, last_input_row)};
                interpolation_method(
                    img_it, clamped_coords, &transf_img_it(y_buff, x_buff, 0));

                float mask_alpha_x = 1.f;
                float mask_alpha_y = 1.f;

                if (transformed_coord.x < 0.f) {
                    mask_alpha_x = 1.f + transformed_coord.x;
                } else if (transformed_coord.x > last_input_col) {
                    mask_alpha_x = in_image_cols_f - transformed_coord.x;
                }

                if (transformed_coord.y < 0.f) {
                    mask_alpha_y = 1.f + transformed_coord.y;
                } else if (transformed_coord.y > last_input_row) {
                    mask_alpha_y = in_image_rows_f - transformed_coord.y;
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

template <typename TransformElementType>
struct AffineTransform
{
    using ElementType                  = TransformElementType;
    static const int number_parameters = 6;

    static Point<ElementType> transform(const Point<ElementType>& x,
                                        const Mat& parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());

        Mat::ConstIterator<ElementType> p_it(parameters);

        // clang-format off
        return {
            (1 + p_it(0, 0, 0)) * x.x + p_it(0, 1, 0) * x.y + p_it(0, 2, 0),
            p_it(0, 3, 0) * x.x + (1 + p_it(0, 4, 0)) * x.y + p_it(0, 5, 0)
        };
        // clang-format on
    }

    static void inverse(const Mat& parameters, Mat& inverted_parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());

        if (inverted_parameters.empty()) {
            inverted_parameters.create<ElementType>(1, number_parameters, 1);
        } else {
            assert(!inverted_parameters.empty());
            assert(inverted_parameters.rows == 1);
            assert(inverted_parameters.cols == number_parameters);
            assert(inverted_parameters.channels() == 1);
            assert(inverted_parameters.type() ==
                   Mat::get_type_enum<ElementType>());
        }

        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        Mat::ConstIterator<ElementType> params_it(parameters);
        Matrix3RowMajor params_mat;
        // clang-format off
        params_mat <<
          1 + params_it(0, 0, 0),     params_it(0, 1, 0), params_it(0, 2, 0),
              params_it(0, 3, 0), 1 + params_it(0, 4, 0), params_it(0, 5, 0),
                               0,                      0,                  1;
        // clang-format on

        Matrix3RowMajor inv_mat = params_mat.inverse();

        // clang-format off
        inverted_parameters << std::initializer_list<ElementType>{
          -1 + inv_mat(0, 0),      inv_mat(0, 1), inv_mat(0, 2),
               inv_mat(1, 0), -1 + inv_mat(1, 1), inv_mat(1, 2)
        };
        // clang-format on
    }

    static void compose(const Mat& outer_params,
                        const Point<ElementType>& inner_translation_params,
                        Mat& composed_params)
    {
        // TODO refactor with similar methods
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        assert(!outer_params.empty());
        assert(outer_params.rows == 1);
        assert(outer_params.cols == number_parameters);
        assert(outer_params.channels() == 1);
        assert(outer_params.type() == Mat::get_type_enum<ElementType>());

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        } else {
            assert(!composed_params.empty());
            assert(composed_params.rows == 1);
            assert(composed_params.cols == number_parameters);
            assert(composed_params.channels() == 1);
            assert(composed_params.type() == Mat::get_type_enum<ElementType>());
        }

        Matrix3RowMajor inner_mat;
        // clang-format off
        inner_mat << 1, 0, inner_translation_params.x,
                     0, 1, inner_translation_params.y,
                     0, 0, 1;
        // clang-format on

        // Create outer eigen matrix
        Matrix3RowMajor outer_mat;
        Mat::ConstIterator<ElementType> outer_it(outer_params);
        // clang-format off
        outer_mat <<
          1 + outer_it(0, 0, 0),     outer_it(0, 1, 0), outer_it(0, 2, 0),
              outer_it(0, 3, 0), 1 + outer_it(0, 4, 0), outer_it(0, 5, 0),
                              0,                     0,                 1;
        // clang-format on

        Matrix3RowMajor composed_mat = outer_mat * inner_mat;

        // Fill output parameter matrix
        // clang-format off
        composed_params << std::initializer_list<ElementType>{
          -1 + composed_mat(0, 0),      composed_mat(0, 1), composed_mat(0, 2),
               composed_mat(1, 0), -1 + composed_mat(1, 1), composed_mat(1, 2)
        };
        // clang-format on
    }

    static void compose(const Mat& outer_params,
                        const Mat& inner_params,
                        Mat& composed_params)
    {
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        assert(!outer_params.empty());
        assert(outer_params.rows == 1);
        assert(outer_params.cols == number_parameters);
        assert(outer_params.channels() == 1);
        assert(outer_params.type() == Mat::get_type_enum<ElementType>());

        assert(!inner_params.empty());
        assert(inner_params.rows == 1);
        assert(inner_params.cols == number_parameters);
        assert(inner_params.channels() == 1);
        assert(inner_params.type() == Mat::get_type_enum<ElementType>());

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        } else {
            assert(!composed_params.empty());
            assert(composed_params.rows == 1);
            assert(composed_params.cols == number_parameters);
            assert(composed_params.channels() == 1);
            assert(composed_params.type() == Mat::get_type_enum<ElementType>());
        }

        // Create inner eigen matrix
        Matrix3RowMajor inner_mat;
        Mat::ConstIterator<ElementType> inner_it(inner_params);
        // clang-format off
        inner_mat <<
          1 + inner_it(0, 0, 0),     inner_it(0, 1, 0), inner_it(0, 2, 0),
              inner_it(0, 3, 0), 1 + inner_it(0, 4, 0), inner_it(0, 5, 0),
                              0,                     0,                 1;
        // clang-format on

        // Create outer eigen matrix
        Matrix3RowMajor outer_mat;
        Mat::ConstIterator<ElementType> outer_it(outer_params);
        // clang-format off
        outer_mat <<
          1 + outer_it(0, 0, 0),     outer_it(0, 1, 0), outer_it(0, 2, 0),
              outer_it(0, 3, 0), 1 + outer_it(0, 4, 0), outer_it(0, 5, 0),
                              0,                     0,                 1;
        // clang-format on

        Matrix3RowMajor composed_mat = outer_mat * inner_mat;

        // Fill output parameter matrix
        // clang-format off
        composed_params << std::initializer_list<ElementType>{
          -1 + composed_mat(0, 0),      composed_mat(0, 1), composed_mat(0, 2),
               composed_mat(1, 0), -1 + composed_mat(1, 1), composed_mat(1, 2),
        };
        // clang-format on
    }

    static void jacobian_origin(const ElementType x, ElementType y, Mat& output)
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

    static void hessian_x_origin(ElementType x, ElementType y, Mat& output)
    {
        assert(!output.empty());
        assert(output.cols == number_parameters);
        assert(output.cols == number_parameters);
        assert(output.channels() == 1);
        assert(output.type() == Mat::get_type_enum<ElementType>());

        // clang-format off
        output << std::initializer_list<ElementType>{
            0,     0,  0, 0, 0, 0,
            0,     0,  0, 0, 0, 0,
            0,     0,  0, 0, 0, 0,
            0,     0,  0, 0, 0, 0,
            0,     0,  0, 0, 0, 0,
            0,     0,  0, 0, 0, 0,
        };
        // clang-format on
    }

    static void hessian_y_origin(ElementType x, ElementType y, Mat& output)
    {
        assert(!output.empty());
        assert(output.cols == number_parameters);
        assert(output.cols == number_parameters);
        assert(output.channels() == 1);
        assert(output.type() == Mat::get_type_enum<ElementType>());

        // clang-format off
        output << std::initializer_list<ElementType>{
            0, 0, 0,   0,   0,  0,
            0, 0, 0,   0,   0,  0,
            0, 0, 0,   0,   0,  0,
            0, 0, 0,   0,   0,  0,
            0, 0, 0,   0,   0,  0,
            0, 0, 0,   0,   0,  0,
        };
        // clang-format on
    }
};

template <typename TransformElementType>
struct HomographyTransform
{
    using ElementType                  = TransformElementType;
    static const int number_parameters = 8;

    static Point<ElementType> transform(const Point<ElementType>& x,
                                        const Mat& parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());

        Mat::ConstIterator<ElementType> p_it(parameters);

        ElementType dividend = p_it(0, 6, 0) * x.x + p_it(0, 7, 0) * x.y + 1;

        // clang-format off
        return {
            ((1 + p_it(0, 0, 0)) * x.x + p_it(0, 1, 0) * x.y + p_it(0, 2, 0)) /
             dividend,
            (p_it(0, 3, 0) * x.x + (1 + p_it(0, 4, 0)) * x.y + p_it(0, 5, 0)) /
             dividend
        };
        // clang-format on
    }

    static void scale(const float factor_x, const float factor_y, Mat& parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());

        Mat::Iterator<ElementType> p_it(parameters);
        p_it(0, 1, 0) *= factor_y / factor_x;
        p_it(0, 2, 0) *= factor_x;

        p_it(0, 3, 0) *= factor_x / factor_y;
        p_it(0, 5, 0) *= factor_y;

        p_it(0, 6, 0) *= 1.0f / factor_x;
        p_it(0, 7, 0) *= 1.0f / factor_y;
    }

    static void inverse(const Mat& parameters, Mat& inverted_parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());

        if (inverted_parameters.empty()) {
            inverted_parameters.create<ElementType>(1, number_parameters, 1);
        } else {
            assert(!inverted_parameters.empty());
            assert(inverted_parameters.rows == 1);
            assert(inverted_parameters.cols == number_parameters);
            assert(inverted_parameters.channels() == 1);
            assert(inverted_parameters.type() ==
                   Mat::get_type_enum<ElementType>());
        }

        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        Mat::ConstIterator<ElementType> params_it(parameters);
        Matrix3RowMajor params_mat;
        // clang-format off
        params_mat <<
          1 + params_it(0, 0, 0),     params_it(0, 1, 0), params_it(0, 2, 0),
              params_it(0, 3, 0), 1 + params_it(0, 4, 0), params_it(0, 5, 0),
              params_it(0, 6, 0),     params_it(0, 7, 0),                  1;
        // clang-format on

        Matrix3RowMajor inv_mat = params_mat.inverse();

        // clang-format off
        inverted_parameters << std::initializer_list<ElementType>{
          -1 + inv_mat(0, 0),      inv_mat(0, 1), inv_mat(0, 2),
               inv_mat(1, 0), -1 + inv_mat(1, 1), inv_mat(1, 2),
               inv_mat(2, 0),      inv_mat(2, 1)
        };
        // clang-format on
    }

    /**
     * Compose the transformation defined by outer_params with the translational
     * transformation given by the inner_translation_params.
     *
     * Equivalent to:
     *  w(x, composed_params) <- w(w(x, inner_params), outer_params)
     */
    static void compose(const Mat& outer_params,
                        const Point<ElementType>& inner_translation_params,
                        Mat& composed_params)
    {
        // TODO refactor with similar methods
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        assert(!outer_params.empty());
        assert(outer_params.rows == 1);
        assert(outer_params.cols == number_parameters);
        assert(outer_params.channels() == 1);
        assert(outer_params.type() == Mat::get_type_enum<ElementType>());

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        } else {
            assert(!composed_params.empty());
            assert(composed_params.rows == 1);
            assert(composed_params.cols == number_parameters);
            assert(composed_params.channels() == 1);
            assert(composed_params.type() == Mat::get_type_enum<ElementType>());
        }

        Matrix3RowMajor inner_mat;
        // clang-format off
        inner_mat << 1, 0, inner_translation_params.x,
                     0, 1, inner_translation_params.y,
                     0, 0, 1;
        // clang-format on

        // Create outer eigen matrix
        Matrix3RowMajor outer_mat;
        Mat::ConstIterator<ElementType> outer_it(outer_params);
        // clang-format off
        outer_mat <<
          1 + outer_it(0, 0, 0),     outer_it(0, 1, 0), outer_it(0, 2, 0),
              outer_it(0, 3, 0), 1 + outer_it(0, 4, 0), outer_it(0, 5, 0),
              outer_it(0, 6, 0),     outer_it(0, 7, 0), 1;
        // clang-format on

        Matrix3RowMajor composed_mat = outer_mat * inner_mat;

        // Fill output parameter matrix
        // clang-format off
        composed_params << std::initializer_list<ElementType>{
          -1 + composed_mat(0, 0),      composed_mat(0, 1), composed_mat(0, 2),
               composed_mat(1, 0), -1 + composed_mat(1, 1), composed_mat(1, 2),
               composed_mat(2, 0),      composed_mat(2, 1)
        };
        // clang-format on
    }

    /**
     * Compose the transformation defined by outer_params with the
     * transformation given by the inner_params.
     *
     * Equivalent to:
     *  w(x, composed_params) <- w(w(x, inner_params), outer_params)
     */
    static void compose(const Mat& outer_params,
                        const Mat& inner_params,
                        Mat& composed_params)
    {
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        assert(!outer_params.empty());
        assert(outer_params.rows == 1);
        assert(outer_params.cols == number_parameters);
        assert(outer_params.channels() == 1);
        assert(outer_params.type() == Mat::get_type_enum<ElementType>());

        assert(!inner_params.empty());
        assert(inner_params.rows == 1);
        assert(inner_params.cols == number_parameters);
        assert(inner_params.channels() == 1);
        assert(inner_params.type() == Mat::get_type_enum<ElementType>());

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        } else {
            assert(!composed_params.empty());
            assert(composed_params.rows == 1);
            assert(composed_params.cols == number_parameters);
            assert(composed_params.channels() == 1);
            assert(composed_params.type() == Mat::get_type_enum<ElementType>());
        }

        // Create inner eigen matrix
        Matrix3RowMajor inner_mat;
        Mat::ConstIterator<ElementType> inner_it(inner_params);
        // clang-format off
        inner_mat <<
          1 + inner_it(0, 0, 0),     inner_it(0, 1, 0), inner_it(0, 2, 0),
              inner_it(0, 3, 0), 1 + inner_it(0, 4, 0), inner_it(0, 5, 0),
              inner_it(0, 6, 0),     inner_it(0, 7, 0), 1;
        // clang-format on

        // Create outer eigen matrix
        Matrix3RowMajor outer_mat;
        Mat::ConstIterator<ElementType> outer_it(outer_params);
        // clang-format off
        outer_mat <<
          1 + outer_it(0, 0, 0),     outer_it(0, 1, 0), outer_it(0, 2, 0),
              outer_it(0, 3, 0), 1 + outer_it(0, 4, 0), outer_it(0, 5, 0),
              outer_it(0, 6, 0),     outer_it(0, 7, 0), 1;
        // clang-format on

        Matrix3RowMajor composed_mat = outer_mat * inner_mat;

        // Fill output parameter matrix
        // clang-format off
        composed_params << std::initializer_list<ElementType>{
          -1 + composed_mat(0, 0),      composed_mat(0, 1), composed_mat(0, 2),
               composed_mat(1, 0), -1 + composed_mat(1, 1), composed_mat(1, 2),
               composed_mat(2, 0),      composed_mat(2, 1)
        };
        // clang-format on
    }

    /**
     * Jacobian of the Homography transform w(coordinate, Dp) evaluated at Dp=0
     */
    static void jacobian_origin(ElementType x, ElementType y, Mat& output)
    {
        assert(!output.empty());
        assert(output.rows == 2);
        assert(output.cols == number_parameters);
        assert(output.channels() == 1);
        assert(output.type() == Mat::get_type_enum<ElementType>());

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
        assert(output.cols == number_parameters);
        assert(output.channels() == 1);
        assert(output.type() == Mat::get_type_enum<ElementType>());

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
        assert(output.cols == number_parameters);
        assert(output.cols == number_parameters);
        assert(output.channels() == 1);
        assert(output.type() == Mat::get_type_enum<ElementType>());

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
