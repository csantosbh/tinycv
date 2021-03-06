#ifndef _TINYCV_TRANSFORM_H_
#define _TINYCV_TRANSFORM_H_

#include <utility>

#include "third_party/eigen/Eigen"

#include "bounding_box.h"
#include "mat.h"
#include "math.h"


namespace tinycv
{

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
            std::floor(output_bb.left_top.x),
            std::floor(output_bb.left_top.y)
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

template <typename InputPixelType, typename OutputPixelType>
void rgb_to_gray(const Mat& rgb_image, Mat& gray_image)
{
    if (gray_image.empty()) {
        gray_image.create<OutputPixelType>(rgb_image.rows, rgb_image.cols, 1);
    }

    assert(!rgb_image.empty());
    assert(!gray_image.empty());

    assert(rgb_image.type() == Mat::get_type_enum<InputPixelType>());
    assert(gray_image.type() == Mat::get_type_enum<OutputPixelType>());

    assert(rgb_image.channels() == 3);
    assert(gray_image.channels() == 1);

    assert(rgb_image.cols == gray_image.cols);
    assert(rgb_image.rows == gray_image.rows);

    Mat::ConstIterator<InputPixelType> rgb_it(rgb_image);
    Mat::Iterator<InputPixelType> gray_it(gray_image);

    for (int row = 0; row < rgb_image.rows; ++row) {
        for (int col = 0; col < rgb_image.cols; ++col) {
            gray_it(row, col, 0) = 0.299 * rgb_it(row, col, 0) +
                                   0.587 * rgb_it(row, col, 1) +
                                   0.114 * rgb_it(row, col, 2);
        }
    }

    return;
}

/**
 * Separable kernel convolution
 */
// Convolution core
enum class ConvolutionDirection { Horizontal, Vertical };
enum class BorderTreatment { Crop, Reflect };

template <typename PixelType,
          ConvolutionDirection direction,
          BorderTreatment border_treatment>
PixelType sample_pixel(const Mat::ConstIterator<PixelType> &input_it,
                       const int border_size,
                       const int n,
                       int y,
                       int x,
                       int c)
{
    if (border_treatment == BorderTreatment::Crop) {
        return (direction == ConvolutionDirection::Vertical)
                   ? input_it(y + n, x, c)
                   : input_it(y, x + n, c);
    } else {
        const auto reflect = []
                             (int value,
                              int higher_limit) {
            return std::abs(higher_limit - std::abs(value - higher_limit));
        };
        const int last_row = input_it.m.rows - 1;
        const int last_col = input_it.m.cols - 1;

        return (direction == ConvolutionDirection::Vertical)
                   ? input_it(reflect(y - border_size + n, last_row), x, c)
                   : input_it(y, reflect(x - border_size + n, last_col), c);
    }
}

template <typename InputPixelType,
          typename OutputPixelType,
          ConvolutionDirection direction,
          BorderTreatment border_treatment>
void convolve(const Mat::ConstIterator<float>& kernel_it,
              const float norm_factor,
              const bool clamp_output,
              const Mat::ConstIterator<InputPixelType>& input_it,
              Mat::Iterator<OutputPixelType>& output_it)
{
    assert(input_it.m.type() == Mat::get_type_enum<InputPixelType>());
    assert(output_it.m.type() == Mat::get_type_enum<OutputPixelType>());
    const int border_size = (kernel_it.m.cols - 1) / 2;

    for (int y = 0; y < output_it.m.rows; ++y) {
        for (int x = 0; x < output_it.m.cols; ++x) {
            for (int c = 0; c < output_it.m.channels(); ++c) {
                float conv_sum = 0.f;

                for (int n = 0; n < kernel_it.m.cols; ++n) {
                    auto input_pix = sample_pixel<InputPixelType,
                                                  direction,
                                                  border_treatment>(
                        input_it, border_size, n, y, x, c);

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
}

template <typename InputPixelType,
          typename OutputPixelType,
          int channels,
          BorderTreatment border_treatment = BorderTreatment::Crop>
void image_convolve(const Mat &image,
                    const Mat &kernel_v,
                    const Mat &kernel_h,
                    const float kernel_norm_factor,
                    Mat &output_image)
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

    Mat output_vertical_pass;

    if (border_treatment == BorderTreatment::Crop) {
        // Create output image for first pass
        output_vertical_pass.create<IntermediatePixelType>(
            image.rows - 2 * border_v, image.cols, channels);

        // Create output image for second pass
        output_image.create<OutputPixelType>(
            image.rows - 2 * border_v, image.cols - 2 * border_h, channels);
    } else {
        assert(border_treatment == BorderTreatment::Reflect);

        // Create output image for first pass
        output_vertical_pass.create<IntermediatePixelType>(
            image.rows, image.cols, channels);

        // Create output image for second pass
        output_image.create<OutputPixelType>(
            image.rows, image.cols, channels);
    }

    // First pass: Vertical convolution
    Mat::Iterator<IntermediatePixelType> first_pass_it(output_vertical_pass);
    bool clamp_output = false;
    convolve<InputPixelType,
             IntermediatePixelType,
             ConvolutionDirection::Vertical,
             border_treatment>(kernel_v_it,
                               1.f,
                               clamp_output,
                               Mat::ConstIterator<InputPixelType>(image),
                               first_pass_it);

    // Second pass: Horizontal convolution
    clamp_output = true;
    Mat::Iterator<OutputPixelType> second_pass_it(output_image);
    convolve<IntermediatePixelType,
             OutputPixelType,
             ConvolutionDirection::Horizontal,
             border_treatment>(
        kernel_h_it,
        kernel_norm_factor,
        clamp_output,
        Mat::ConstIterator<IntermediatePixelType>(output_vertical_pass),
        second_pass_it);

    return;
}

template <typename InputPixelType,
          typename OutputPixelType,
          int channels,
          BorderTreatment border_treatment>
void gaussian_blur(const Mat &image,
                   int kernel_border_size,
                   float standard_deviation,
                   Mat &output_image)
{
    assert(std::abs(standard_deviation) > 1e-6f);
    assert(image.channels() == channels);
    assert(image.type() == Mat::get_type_enum<InputPixelType>());

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

    // TODO image might contain black border, but convolution is ignoring the
    // mask
    float norm_factor = 1.f / (kernel_summation * kernel_summation);
    image_convolve<InputPixelType, OutputPixelType, channels, border_treatment>(
        image, kernel, kernel, norm_factor, output_image);

    return;
}

template <typename TransformElementType>
struct HomographyTransform
{
    using ElementType                  = TransformElementType;
    static const int number_parameters = 8;

    static bool assert_validity(const Mat& parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());
    }

    static void identity(Mat& parameters)
    {
        if (parameters.empty()) {
            parameters.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);

        // clang-format off
        parameters << std::initializer_list<float> {
            0, 0, 0, 0,
            0, 0, 0, 0
        };
        // clang-format on
    }

    /**
     * Given at least 4 correspondent points <reference> and <tracked>,
     * find the homography <parameters> such that
     *
     *   reference = w(tracked, parameters).
     */
    static void from_matches(const std::vector<Point<ElementType>>& reference,
                             const std::vector<Point<ElementType>>& tracked,
                             Mat& parameters)
    {
        assert(reference.size() >= 4);
        assert(tracked.size() >= 4);

        assert(reference.size() == tracked.size());

        if (parameters.empty()) {
            parameters.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);

        // Normalize coordinates to avoid numerical issues
        float norm_factor = 1.f;
        for (const auto& container : {reference, tracked}) {
            for (const auto& p : container) {
                norm_factor = std::max(norm_factor,
                                       std::max(std::abs(p.x), std::abs(p.y)));
            }
        }
        norm_factor = 1.f / norm_factor;

        // Generate coefficient matrix
        Eigen::MatrixXf coeffs(2 * reference.size(), 9);

        for (size_t i = 0; i < reference.size(); ++i) {
            Point<ElementType> norm_r = {reference[i].x * norm_factor,
                                         reference[i].y * norm_factor};

            Point<ElementType> norm_t = {tracked[i].x * norm_factor,
                                         tracked[i].y * norm_factor};

            // clang-format off
            coeffs.row(2 * i) <<
                            norm_t.x,             norm_t.y,         1,
                                   0,                    0,         0,
                -norm_t.x * norm_r.x, -norm_t.y * norm_r.x, -norm_r.x;

            coeffs.row(2 * i + 1) <<
                                   0,                    0,         0,
                            norm_t.x,             norm_t.y,         1,
                -norm_t.x * norm_r.y, -norm_t.y * norm_r.y, -norm_r.y;
            // clang-format on
        }

        // Compute eigenvalues
        Eigen::MatrixXf cTc = coeffs.transpose() * coeffs;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigen_solver(cTc);

        // Obtain the eigen vector with lowest eigen value
        Eigen::VectorXf parameters_eig = eigen_solver.eigenvectors().col(0);

        // Normalize homography by last element
        parameters_eig *= 1.f / parameters_eig[8];

        // Convert homography to transform parameters
        // clang-format off
        parameters << std::initializer_list<ElementType> {
            parameters_eig[0] - 1.f, parameters_eig[1], parameters_eig[2],
            parameters_eig[3], parameters_eig[4] - 1.f, parameters_eig[5],
            parameters_eig[6], parameters_eig[7]
        };
        // clang-format on

        // Update transform scale to original resolution of input points
        change_scale(
            {1.f / norm_factor, 1.f / norm_factor}, parameters, parameters);

        return;
    }

    static Point<ElementType> transform(const Point<ElementType>& x,
                                        const Mat& parameters)
    {
        assert_validity(parameters);

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

    /**
     * The homography with <parameters> maps p to p':
     *    w(p, parameters) = p'
     *
     * If we want to change the scale of the space where p and p' lie (e.g. when
     * adapting the homography to work on a different image resolution) such
     * that
     *
     *     P = s * p = w(p, scale),
     *     P' = s * p' = w(p', scale),
     *
     * then
     *
     *     p = w^-1(P, scale),
     *
     * and
     *
     *     p' = w(p, parameters) = w(w^-1(P, scale), parameters).
     *
     * Therefore,
     *
     *     P' = w(w(w^-1(P, scale), parameters), scale).
     *
     * In matrix form, this is equivalent of transforming the homography H of
     * <parameters> by using the scale-only homography S as follows:
     *     H <- S * H * S^-1
     */
    static void change_scale(const Point<float>& scale,
                             const Mat& parameters,
                             Mat& adjusted_parameters)
    {
        assert_validity(parameters);

        Mat::ConstIterator<ElementType> params_it(parameters);

        adjusted_parameters << std::initializer_list<float>{
            params_it(0, 0, 0),
            params_it(0, 1, 0) * scale.x / scale.y,
            params_it(0, 2, 0) * scale.x,

            params_it(0, 3, 0) * scale.y / scale.x,
            params_it(0, 4, 0),
            params_it(0, 5, 0) * scale.y,

            params_it(0, 6, 0) * 1.0f / scale.x,
            params_it(0, 7, 0) * 1.0f / scale.y,
        };
    }

    /**
     * The homography with <parameters> maps p to p':
     *    w(p, parameters) = p'
     *
     * If we want to change the position of the space where p and p' lie (e.g.
     * if the homography was estimated for a cropped image and we want to obtain
     * the homography that works for the original image) such that
     *
     *     P = t + p = w(p, translation),
     *     P' = t + p' = w(p', translation),
     *
     * then
     *
     *     p = w^-1(P, translation),
     *
     * and
     *
     *     p' = w(p, parameters) = w(w^-1(P, translation), parameters).
     *
     * Therefore,
     *
     *     P' = w(w(w^-1(P, translation), parameters), translation).
     *        = w(w(w(P, inv_translation), parameters), translation).
     *
     */
    static void change_position(const Point<ElementType>& translation,
                                const Mat& parameters,
                                Mat& adjusted_parameters)
    {
        compose(parameters, -translation, adjusted_parameters);
        compose(translation, adjusted_parameters, adjusted_parameters);
    }

    static void inverse(const Mat& parameters, Mat& inverted_parameters)
    {
        if (inverted_parameters.empty()) {
            inverted_parameters.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);
        assert_validity(inverted_parameters);

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
        inv_mat *= 1.f / inv_mat(2, 2);

        // clang-format off
        inverted_parameters << std::initializer_list<ElementType>{
          -1 + inv_mat(0, 0),      inv_mat(0, 1), inv_mat(0, 2),
               inv_mat(1, 0), -1 + inv_mat(1, 1), inv_mat(1, 2),
               inv_mat(2, 0),      inv_mat(2, 1)
        };
        // clang-format on
    }

    /**
     * Compose the translational transformation given by the
     * inner_translation_params with the transformation defined by outer_params.
     *
     * Equivalent to:
     *  w(x, composed_params) <- w(w(x, outer_params), inner_translation_params)
     */
    static void compose(const Point<ElementType>& outer_translation_params,
                        const Mat& inner_params,
                        Mat& composed_params)
    {
        assert_validity(inner_params);

        Mat outer_params;
        outer_params.create<ElementType>(1, number_parameters, 1);

        // clang-format off
        outer_params << std::initializer_list<ElementType>{
            0, 0, outer_translation_params.x,
            0, 0, outer_translation_params.y,
            0, 0
        };
        // clang-format on

        compose(outer_params, inner_params, composed_params);
    }

    /**
     * Compose the transformation defined by outer_params with the translational
     * transformation given by the inner_translation_params.
     *
     * Equivalent to:
     *  w(x, composed_params) <- w(w(x, inner_translation_params), outer_params)
     */
    static void compose(const Mat& outer_params,
                        const Point<ElementType>& inner_translation_params,
                        Mat& composed_params)
    {
        assert_validity(outer_params);

        Mat inner_params;
        inner_params.create<ElementType>(1, number_parameters, 1);

        // clang-format off
        inner_params << std::initializer_list<ElementType>{
            0, 0, inner_translation_params.x,
            0, 0, inner_translation_params.y,
            0, 0
        };
        // clang-format on

        compose(outer_params, inner_params, composed_params);
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

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(outer_params);
        assert_validity(inner_params);
        assert_validity(composed_params);

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
        composed_mat *= 1.f / composed_mat(2, 2);

        // Fill output parameter matrix
        // clang-format off
        composed_params << std::initializer_list<ElementType>{
          -1 + composed_mat(0, 0),      composed_mat(0, 1), composed_mat(0, 2),
               composed_mat(1, 0), -1 + composed_mat(1, 1), composed_mat(1, 2),
               composed_mat(2, 0),      composed_mat(2, 1)
        };
        // clang-format on

        return;
    }

    static void to_homography(const Mat& parameters, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);
        assert_validity(output);

        output = Mat(parameters, Mat::CopyMode::Deep);
    }

    static void from_homography(const Mat& parameters, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);
        assert_validity(output);

        output = Mat(parameters, Mat::CopyMode::Deep);
    }

    /**
     * Jacobian of the Homography transform w(coordinate, Dp) evaluated at Dp=0
     */
    static void jacobian_origin(ElementType x, ElementType y, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(2, number_parameters, 1);
        }

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

template <typename TransformElementType>
struct AffineTransform
{
    using ElementType                  = TransformElementType;
    static const int number_parameters = 6;

    static void assert_validity(const Mat& parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());
    }

    static void identity(Mat& parameters)
    {
        if (parameters.empty()) {
            parameters.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);

        // clang-format off
        parameters << std::initializer_list<float> {
            0, 0, 0,
            0, 0, 0,
        };
        // clang-format on
    }

    static Point<ElementType> transform(const Point<ElementType>& x,
                                        const Mat& parameters)
    {
        assert_validity(parameters);

        Mat::ConstIterator<ElementType> p_it(parameters);

        // clang-format off
        return {
            (1 + p_it(0, 0, 0)) * x.x + p_it(0, 1, 0) * x.y + p_it(0, 2, 0),
            p_it(0, 3, 0) * x.x + (1 + p_it(0, 4, 0)) * x.y + p_it(0, 5, 0)
        };
        // clang-format on
    }

    /**
     * See docs for HomographyTransform::change_scale
     */
    static void change_scale(const Point<float>& scale,
                             const Mat& parameters,
                             Mat& adjusted_parameters)
    {
        assert_validity(parameters);

        Mat::ConstIterator<ElementType> params_it(parameters);

        adjusted_parameters << std::initializer_list<float>{
            params_it(0, 0, 0),
            params_it(0, 1, 0) * scale.x / scale.y,
            params_it(0, 2, 0) * scale.x,

            params_it(0, 3, 0) * scale.y / scale.x,
            params_it(0, 4, 0),
            params_it(0, 5, 0) * scale.y,
        };
    }

    /**
     * See docs for HomographyTransform::change_position
     */
    static void change_position(const Point<ElementType>& translation,
                                const Mat& parameters,
                                Mat& adjusted_parameters)
    {
        compose(parameters, -translation, adjusted_parameters);
        compose(translation, adjusted_parameters, adjusted_parameters);
    }

    static void inverse(const Mat& parameters, Mat& inverted_parameters)
    {
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        if (inverted_parameters.empty()) {
            inverted_parameters.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);
        assert_validity(inverted_parameters);

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

    static void compose(const Point<ElementType>& outer_translation_params,
                        const Mat& inner_params,
                        Mat& composed_params)
    {
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(inner_params);
        assert_validity(composed_params);

        Mat outer_params;
        outer_params.create<ElementType>(1, number_parameters, 1);

        // clang-format off
        outer_params << std::initializer_list<ElementType>{
            0, 0, outer_translation_params.x,
            0, 0, outer_translation_params.y,
        };
        // clang-format on

        compose(outer_params, inner_params, composed_params);
    }

    static void compose(const Mat& outer_params,
                        const Point<ElementType>& inner_translation_params,
                        Mat& composed_params)
    {
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(outer_params);
        assert_validity(composed_params);

        Mat inner_params;
        inner_params.create<ElementType>(1, number_parameters, 1);

        // clang-format off
        inner_params << std::initializer_list<ElementType>{
            0, 0, inner_translation_params.x,
            0, 0, inner_translation_params.y,
        };
        // clang-format on

        compose(outer_params, inner_params, composed_params);
    }

    static void compose(const Mat& outer_params,
                        const Mat& inner_params,
                        Mat& composed_params)
    {
        using Matrix3RowMajor =
            Eigen::Matrix<ElementType, 3, 3, Eigen::RowMajor>;

        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(outer_params);
        assert_validity(inner_params);
        assert_validity(composed_params);

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

    static void to_homography(const Mat& parameters, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(
                1, HomographyTransform<ElementType>::number_parameters, 1);
        }

        assert_validity(parameters);
        HomographyTransform<ElementType>::assert_validity(output);

        Mat::ConstIterator<ElementType> params_it(parameters);

        // clang-format off
        output << std::initializer_list<ElementType> {
            params_it(0, 0, 0), params_it(0, 1, 0), params_it(0, 2, 0),
            params_it(0, 3, 0), params_it(0, 4, 0), params_it(0, 5, 0),
            0, 0
        };
        // clang-format on
    }

    static void from_homography(const Mat& parameters, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(1, number_parameters, 1);
        }

        HomographyTransform<ElementType>::assert_validity(parameters);
        assert_validity(output);

        Mat::ConstIterator<ElementType> params_it(parameters);

        // The homography must contain an affine transform
        assert(params_it(0, 6, 0) == 0.0f);
        assert(params_it(0, 7, 0) == 0.0f);

        // clang-format off
        output << std::initializer_list<ElementType> {
            params_it(0, 0, 0), params_it(0, 1, 0), params_it(0, 2, 0),
            params_it(0, 3, 0), params_it(0, 4, 0), params_it(0, 5, 0),
        };
        // clang-format on
    }

    static void jacobian_origin(const ElementType x, ElementType y, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(2, number_parameters, 1);
        }

        assert(!output.empty());
        assert(output.rows == 2);
        assert(output.cols == number_parameters);
        assert(output.channels() == 1);
        assert(output.type() == Mat::get_type_enum<ElementType>());

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
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
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
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        };
        // clang-format on
    }
};

template <typename TransformElementType>
struct TranslationTransform
{
    using ElementType                  = TransformElementType;
    static const int number_parameters = 2;

    static void assert_validity(const Mat& parameters)
    {
        assert(!parameters.empty());
        assert(parameters.rows == 1);
        assert(parameters.cols == number_parameters);
        assert(parameters.channels() == 1);
        assert(parameters.type() == Mat::get_type_enum<ElementType>());
    }

    static void identity(Mat& parameters)
    {
        if (parameters.empty()) {
            parameters.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);

        // clang-format off
        parameters << std::initializer_list<float> {
            0, 0
        };
        // clang-format on
    }

    static Point<ElementType> transform(const Point<ElementType>& x,
                                        const Mat& parameters)
    {
        assert_validity(parameters);

        Mat::ConstIterator<ElementType> p_it(parameters);

        // clang-format off
        return {
            x.x + p_it(0, 0, 0),
            x.y + p_it(0, 1, 0)
        };
        // clang-format on
    }

    /**
     * See docs for HomographyTransform::change_scale
     */
    static void change_scale(const Point<float>& scale,
                             const Mat& parameters,
                             Mat& adjusted_parameters)
    {
        assert_validity(parameters);

        Mat::ConstIterator<ElementType> params_it(parameters);

        adjusted_parameters << std::initializer_list<float>{
            params_it(0, 0, 0) * scale.x,
            params_it(0, 1, 0) * scale.y,
        };
    }

    /**
     * See docs for HomographyTransform::change_position
     */
    static void change_position(const Point<ElementType>& translation,
                                const Mat& parameters,
                                Mat& adjusted_parameters)
    {
        compose(parameters, -translation, adjusted_parameters);
        compose(translation, adjusted_parameters, adjusted_parameters);
    }

    static void inverse(const Mat& parameters, Mat& inverted_parameters)
    {
        if (inverted_parameters.empty()) {
            inverted_parameters.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(parameters);
        assert_validity(inverted_parameters);

        Mat::ConstIterator<ElementType> params_it(parameters);

        // clang-format off
        inverted_parameters << std::initializer_list<ElementType>{
            -params_it(0, 0, 0), -params_it(0, 1, 0)
        };
        // clang-format on
    }

    static void compose(const Point<ElementType>& outer_translation_params,
                        const Mat& inner_params,
                        Mat& composed_params)
    {
        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(inner_params);
        assert_validity(composed_params);

        Mat outer_params;
        outer_params.create<ElementType>(1, number_parameters, 1);

        // clang-format off
        outer_params << std::initializer_list<ElementType>{
            outer_translation_params.x, outer_translation_params.y,
        };
        // clang-format on

        compose(outer_params, inner_params, composed_params);
    }

    static void compose(const Mat& outer_params,
                        const Point<ElementType>& inner_translation_params,
                        Mat& composed_params)
    {
        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(outer_params);
        assert_validity(composed_params);

        Mat inner_params;
        inner_params.create<ElementType>(1, number_parameters, 1);

        // clang-format off
        inner_params << std::initializer_list<ElementType>{
            inner_translation_params.x, inner_translation_params.y,
        };
        // clang-format on

        compose(outer_params, inner_params, composed_params);
    }

    static void compose(const Mat& outer_params,
                        const Mat& inner_params,
                        Mat& composed_params)
    {
        if (composed_params.empty()) {
            composed_params.create<ElementType>(1, number_parameters, 1);
        }

        assert_validity(outer_params);
        assert_validity(inner_params);
        assert_validity(composed_params);

        // Create parameter iterators
        Mat::ConstIterator<ElementType> inner_it(inner_params);
        Mat::ConstIterator<ElementType> outer_it(outer_params);

        // Fill output parameter matrix
        // clang-format off
        composed_params << std::initializer_list<ElementType>{
          outer_it(0, 0, 0) + inner_it(0, 0, 0),
          outer_it(0, 1, 0) + inner_it(0, 1, 0),
        };
        // clang-format on
    }

    static void to_homography(const Mat& parameters, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(
                1, HomographyTransform<ElementType>::number_parameters, 1);
        }

        assert_validity(parameters);
        HomographyTransform<ElementType>::assert_validity(output);

        Mat::ConstIterator<ElementType> params_it(parameters);

        // clang-format off
        output << std::initializer_list<ElementType> {
            0.f, 0.f, params_it(0, 0, 0),
            0.f, 0.f, params_it(0, 1, 0),
            0, 0
        };
        // clang-format on
    }

    static void from_homography(const Mat& parameters, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(1, number_parameters, 1);
        }

        HomographyTransform<ElementType>::assert_validity(parameters);
        assert_validity(output);

        Mat::ConstIterator<ElementType> params_it(parameters);

        // The homography must contain a translational transform
        assert(params_it(0, 0, 0) == 0.0f);
        assert(params_it(0, 1, 0) == 0.0f);
        assert(params_it(0, 3, 0) == 0.0f);
        assert(params_it(0, 4, 0) == 0.0f);
        assert(params_it(0, 6, 0) == 0.0f);
        assert(params_it(0, 7, 0) == 0.0f);

        // clang-format off
        output << std::initializer_list<ElementType> {
            params_it(0, 2, 0),
            params_it(0, 5, 0),
        };
        // clang-format on
    }

    static void jacobian_origin(const ElementType x, ElementType y, Mat& output)
    {
        if (output.empty()) {
            output.create<ElementType>(2, number_parameters, 1);
        }

        assert(!output.empty());
        assert(output.rows == 2);
        assert(output.cols == number_parameters);
        assert(output.channels() == 1);
        assert(output.type() == Mat::get_type_enum<ElementType>());

        // clang-format off
        output << std::initializer_list<ElementType>{
            1, 0,
            0, 1,
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
            0, 0,
            0, 0,
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
            0, 0,
            0, 0,
        };
        // clang-format on
    }
};
}

#endif
