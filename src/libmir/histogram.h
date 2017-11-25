#ifndef _LIBMIR_HISTOGRAM_H_
#define _LIBMIR_HISTOGRAM_H_

#include <cmath>

#include <Eigen/Eigen>

#include "mat.h"

class HistogramConfig
{
  public:
    static constexpr int num_central_bins()
    {
        return 8;
    }

    static constexpr float remap_epsilon()
    {
        return 1e-4f;
    }

    template <typename BinningMethod>
    static constexpr int num_total_bins()
    {
        return 2 * BinningMethod::influence_margin() + num_central_bins();
    }
};

/**
 * Kronecker's binary function
 */
class KroneckerFunction
{
  public:
    static constexpr int influence_margin()
    {
        return 0;
    }

    static float histogram_bin_function(const float i)
    {
        return (i >= -0.5f && i <= 0.5f) ? 1 : 0;
    }
};

/**
 * 4th order B-splines
 */
class BSpline4
{
  public:
    static constexpr int influence_margin()
    {
        return 2;
    }

    static float histogram_bin_function(const float i)
    {
        float result;

        if (i < -2.f) {
            // i < -2.0
            result = 0.f;
        } else if (i < -1.f) {
            // -2.0 <= i < -1.0
            float k = (2.f + i);

            result = k * k * k / 6.f;
        } else if (i < 0.f) {
            // -1.0 <= i < 0.0
            float k = 1.f + i;

            result = (1.f + 3.f * k + 3 * k * k - 3.f * k * k * k) / 6.f;
        } else if (i < 1.f) {
            // 0.0 <= i < 1.0
            float k = 1.f - i;

            result = (1.f + 3.f * k + 3 * k * k - 3.f * k * k * k) / 6.f;
        } else if (i < 2.f) {
            // 1.0 <= i < 2.0
            float k = (2.f - i);

            result = k * k * k / 6.f;
        } else {
            // 2.0 <= i
            assert(2.f <= i);

            result = 0.f;
        }

        return result;
    }

    /*
     * Derivative of the Histogram Bin Function at point i
     */
    static float hbf_derivative(const float i)
    {
        const auto b_spline_3 = [](const float i) -> float {
            float result;

            if (i < -1.5) {
                // i < -1.5
                result = 0.f;
            } else if (i < -0.5) {
                // -1.5 <= i < -0.5
                float k = (1.5f + i);

                result = k * k / 2.f;
            } else if (i < 0.f) {
                // -0.5 <= i < 0.0
                float k = 0.5f + i;

                result = 1 + i - k * k;
            } else if (i < 0.5f) {
                // 0.0 <= i < 0.5
                float k = 0.5f - i;

                result = 1 - i - k * k;
            } else if (i < 1.5) {
                // 0.5 <= i < 1.5
                float k = 1.5f - i;

                result = k * k / 2.f;
            } else {
                // 1.5 <= i
                assert(1.5 <= i);

                result = 0.f;
            }

            return result;
        };

        return b_spline_3(i + 0.5f) - b_spline_3(i - 0.5f);
    }

    /*
     * Second order derivative of the Histogram Bin Function at point i
     */
    static float hbf_second_derivative(const float i)
    {
        const auto b_spline_2 = [](const float i) {
            float result;

            if (i < -1.f) {
                // i < -1
                result = 0.f;
            } else if (i < 0.f) {
                // -1 <= i < 0
                result = 1.f + i;
            } else if (i < 1.f) {
                // 0 <= i < 1
                result = 1.f - i;
            } else {
                // 1 <= i
                result = 0.f;
            }

            return result;
        };

        return b_spline_2(i + 1.0f) - 2.f * b_spline_2(i) +
               b_spline_2(i - 1.0f);
    }
};

/**
 * Support structure for storing bin weight contributions when computing image
 * histograms
 */
template <typename BinningMethod>
using WeightArray =
    std::array<float, 2 * BinningMethod::influence_margin() + 1>;

/**
 * Creates an image by mapping the ranges [0, 255] to [-0.5,
 * HistogramConfig::num_central_bins()-0.5).
 *
 * For instance, if HistogramConfig::num_central_bins() is 8, then [0, 255] is
 * mapped to [-0.5, 7.5). Note the open right end on this set, which is
 * intentional.
 */
template <typename InputPixelType,
          typename OutputPixelType,
          typename MaskIterator>
Mat image_remap_histogram(const Mat& input_img,
                          const MaskIterator& mask_iterator)
{
    // The input image must have been initialized
    assert(!input_img.empty());

    // The mask iterator must be correct
    assert(mask_iterator.is_mask_of(input_img));

    // Initialize and allocate output mat
    Mat output_img;
    output_img.create<OutputPixelType>(
        input_img.rows, input_img.cols, input_img.channels());

    // Compute coefficients of linear bin mapping function
    const float epsilon   = HistogramConfig::remap_epsilon();
    const float color_max = 255.f + epsilon;
    const float a_bin_map =
        static_cast<float>(HistogramConfig::num_central_bins()) / color_max;
    const float b_bin_map = -0.5;

    Mat::ConstIterator<InputPixelType> input_it(input_img);
    output_img.for_each<Mat::Iterator<OutputPixelType>>(
        [&input_it, &mask_iterator, a_bin_map, b_bin_map](
            Mat::Iterator<OutputPixelType>& output_it,
            const int y,
            const int x,
            const int c) {
            if (mask_iterator(y, x, c) != 0) {
                output_it(y, x, c) = a_bin_map * input_it(y, x, c) + b_bin_map;
            }
        });

    return output_img;
}

double sum_histogram(const Mat::ConstIterator<float>& histogram)
{
    assert(!histogram.m.empty());

    double histogram_summation = 0.0;

    for (int y = 0; y < histogram.m.rows; ++y) {
        for (int x = 0; x < histogram.m.cols; ++x) {
            for (int c = 0; c < histogram.m.channels(); ++c) {
                histogram_summation += histogram(y, x, c);
            }
        }
    }

    return histogram_summation;
}

void normalize_histogram(const double histogram_summation,
                         Mat::Iterator<float>& histogram)
{
    assert(!histogram.m.empty());

    const float histogram_normalization =
        static_cast<float>(1.0 / histogram_summation);

    for (int y = 0; y < histogram.m.rows; ++y) {
        for (int x = 0; x < histogram.m.cols; ++x) {
            for (int c = 0; c < histogram.m.channels(); ++c) {
                histogram(y, x, c) *= histogram_normalization;
            }
        }
    }
}

/**
 * Helper function for computing which histogram bins are affected by a pixel,
 * and how much
 */
template <typename PixelType, typename BinningMethod>
void compute_bin_contribution_range(const int y,
                                    const int x,
                                    const Mat::ConstIterator<PixelType>& img_it,
                                    float& pixel_bin,
                                    int& lower_bin,
                                    int& upper_bin,
                                    WeightArray<BinningMethod>& bin_weights)
{
    const PixelType& pixel_val = img_it(y, x, 0);

    // Compute pixel bin coordinate
    const int pixel_bin_rounded = static_cast<int>(pixel_val + 0.5);
    pixel_bin                   = pixel_val + BinningMethod::influence_margin();

    assert(std::abs(static_cast<float>(pixel_bin_rounded) - pixel_val) <= 0.5f);

    // Compute bin index range for the weight vector
    lower_bin = pixel_bin_rounded;
    upper_bin = pixel_bin_rounded + 2 * BinningMethod::influence_margin();

    assert(0 <= lower_bin);
    assert(lower_bin <= upper_bin);
    assert(upper_bin <= HistogramConfig::num_total_bins<BinningMethod>() - 1);

    // Update weight storage for given pixel
    int bin_weights_idx = 0;
    for (int neighbor = lower_bin; neighbor <= upper_bin; ++neighbor) {
        const float distance_to_neighbor =
            static_cast<float>(neighbor) - pixel_bin;

        assert(std::abs(distance_to_neighbor) <
               (BinningMethod::influence_margin() + 1));

        bin_weights[bin_weights_idx] =
            BinningMethod::histogram_bin_function(distance_to_neighbor);
        bin_weights_idx++;
    }
}

/**
 * Build histograms for two images, as well as their joint histogram
 *
 * The output is rowmajor, with rows representing the bins for
 * image_r, and the columns representing the bins for image_t.
 *
 * Binning layout sample for 8 bins:
 *
 * Color intensity: 0    31    63    95   127   159   191   223   255
 * Bin index:       |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
 *                  |_____|_____|_____|_____|_____|_____|_____|_____|
 *
 * Given color intensity i, its bin k can be computed with a linear equation as:
 *
 * k = a * i + b,
 *
 * where a = NUM_BINS / (COLOR_MAX + epsilon) and
 *       b = -0.5.
 *
 * In the example above, a = 8 / (255 + epsilon) and epsilon = 1e-4.
 */
template <typename PixelType,
          typename BinningMethod,
          typename MaskIteratorR,
          typename MaskIteratorT>
void joint_image_histogram(const Mat& img_reference,
                           const MaskIteratorR& mask_reference_it,
                           const Mat& img_tracked,
                           const MaskIteratorT& mask_tracked_it,
                           Mat& histogram_r,
                           Mat& histogram_t,
                           Mat& histogram_rt)
{
    const int number_hist_bins =
        HistogramConfig::num_total_bins<BinningMethod>();

    // This function works for single channel images only
    assert(img_reference.channels() == 1);
    assert(img_tracked.channels() == 1);

    // The input images must have the same dimensions
    assert(img_reference.rows == img_tracked.rows);
    assert(img_reference.cols == img_tracked.cols);

    // The input masks must have been correctly initialized
    assert(mask_reference_it.is_mask_of(img_reference));
    assert(mask_tracked_it.is_mask_of(img_tracked));

    // Create iterators for input images
    Mat::ConstIterator<PixelType> img_r_it(img_reference);
    Mat::ConstIterator<PixelType> img_t_it(img_tracked);

    // Resize histogram_r
    if (histogram_r.empty()) {
        histogram_r.create<float>(1, number_hist_bins, 1);
    } else {
        assert(histogram_r.type() == Mat::Type::FLOAT32);
        assert(histogram_r.rows == 1);
        assert(histogram_r.cols == number_hist_bins);
        assert(histogram_r.channels() == 1);
    }

    // Resize histogram_t
    if (histogram_t.empty()) {
        histogram_t.create<float>(1, number_hist_bins, 1);
    } else {
        assert(histogram_t.type() == Mat::Type::FLOAT32);
        assert(histogram_t.rows == 1);
        assert(histogram_t.cols == number_hist_bins);
        assert(histogram_t.channels() == 1);
    }

    // Resize joint histogram
    if (histogram_rt.empty()) {
        histogram_rt.create<float>(number_hist_bins, number_hist_bins, 1);
    } else {
        assert(histogram_rt.type() == Mat::Type::FLOAT32);
        assert(histogram_rt.rows == number_hist_bins);
        assert(histogram_rt.cols == number_hist_bins);
        assert(histogram_rt.channels() == 1);
    }

    // Create iterators for histograms
    Mat::Iterator<float> hist_r_it(histogram_r);
    Mat::Iterator<float> hist_t_it(histogram_t);
    Mat::Iterator<float> hist_rt_it(histogram_rt);

    // Initialize histograms
    histogram_r.fill<float>(0.f);
    histogram_t.fill<float>(0.f);
    histogram_rt.fill<float>(0.f);

    // Store how much each pixel contribute to all histogram bins that are
    // influenced by its value
    WeightArray<BinningMethod> bin_weights_r{};
    WeightArray<BinningMethod> bin_weights_t{};

    for (int y = 0; y < img_reference.rows; ++y) {
        for (int x = 0; x < img_reference.cols; ++x) {
            int lower_bin_r;
            int upper_bin_r;
            float pixel_bin_r;

            int lower_bin_t;
            int upper_bin_t;
            float pixel_bin_t;

            // Compute pixel weight due to masking at borders
            float mask_weight_r     = mask_reference_it(y, x, 0) / 255.f;
            float mask_weight_t     = mask_tracked_it(y, x, 0) / 255.f;
            float joint_mask_weight = mask_weight_r * mask_weight_t;

            const bool pixel_mask = (mask_reference_it(y, x, 0) != 0) &&
                                    (mask_tracked_it(y, x, 0) != 0);

            if (pixel_mask) {
                // Update histogram_r
                compute_bin_contribution_range<PixelType, BinningMethod>(
                    y,
                    x,
                    img_r_it,
                    pixel_bin_r,
                    lower_bin_r,
                    upper_bin_r,
                    bin_weights_r);

                int bin_weights_idx = 0;
                for (int neighbor = lower_bin_r; neighbor <= upper_bin_r;
                     ++neighbor) {
                    hist_r_it(0, neighbor, 0) +=
                        mask_weight_r * bin_weights_r[bin_weights_idx];

                    bin_weights_idx++;
                }

                // Update histogram_t
                compute_bin_contribution_range<PixelType, BinningMethod>(
                    y,
                    x,
                    img_t_it,
                    pixel_bin_t,
                    lower_bin_t,
                    upper_bin_t,
                    bin_weights_t);

                bin_weights_idx = 0;
                for (int neighbor = lower_bin_t; neighbor <= upper_bin_t;
                     ++neighbor) {
                    hist_t_it(0, neighbor, 0) +=
                        mask_weight_t * bin_weights_t[bin_weights_idx];

                    bin_weights_idx++;
                }

                // Update histogram_rt
                for (int neighbor_t = lower_bin_t; neighbor_t <= upper_bin_t;
                     ++neighbor_t) {
                    const int bin_weight_t_idx = neighbor_t - lower_bin_t;

                    for (int neighbor_r = lower_bin_r;
                         neighbor_r <= upper_bin_r;
                         ++neighbor_r) {

                        const int bin_weight_r_idx = neighbor_r - lower_bin_r;

                        hist_rt_it(neighbor_t, neighbor_r, 0) +=
                            joint_mask_weight *
                            bin_weights_r[bin_weight_r_idx] *
                            bin_weights_t[bin_weight_t_idx];
                    }
                }
            }
        }
    }

    // Normalize histograms
    normalize_histogram(sum_histogram(hist_r_it), hist_r_it);
    normalize_histogram(sum_histogram(hist_t_it), hist_t_it);
    normalize_histogram(sum_histogram(hist_rt_it), hist_rt_it);

    return;
}

template <typename PixelType,
          typename GradPixelType,
          typename BinningMethod,
          typename TransformClass,
          typename MaskIteratorR,
          typename MaskIteratorT>
void joint_hist_gradient(const Mat& img_reference,
                         const MaskIteratorR& mask_reference_it,
                         const Mat& steepest_grad_r,
                         const Mat& img_tracked,
                         const MaskIteratorT& mask_tracked_it,
                         double& histogram_r_sum,
                         double& histogram_rt_sum,
                         Mat& histogram_r,
                         Mat& histogram_rt,
                         Mat& histogram_rt_grad)
{
    const int number_transform_params = TransformClass::number_parameters;
    const int number_hist_bins =
        HistogramConfig::num_total_bins<BinningMethod>();

    // This function works for single channel images only
    assert(img_reference.channels() == 1);
    assert(img_tracked.channels() == 1);

    // The input images must have the same dimensions
    assert(img_reference.rows == img_tracked.rows);
    assert(img_reference.cols == img_tracked.cols);

    // The steepest image must be of same width and height as the input images
    assert(img_reference.rows == steepest_grad_r.rows);
    assert(img_reference.cols == steepest_grad_r.cols);

    // The input masks must have been correctly initialized
    assert(mask_reference_it.is_mask_of(img_reference));
    assert(mask_tracked_it.is_mask_of(img_tracked));

    // The provided steepest image must have been generated with the same
    // TransformClass
    assert(steepest_grad_r.channels() == number_transform_params);

    // Create iterators for input images
    Mat::ConstIterator<PixelType> img_r_it(img_reference);
    Mat::ConstIterator<PixelType> img_t_it(img_tracked);
    Mat::ConstIterator<GradPixelType> steepest_grad_it(steepest_grad_r);

    ///
    // Allocate and initialize histograms

    // Resize histogram of reference image
    if (histogram_r.empty()) {
        histogram_r.create<float>(1, number_hist_bins, 1);
    } else {
        assert(histogram_r.type() == Mat::Type::FLOAT32);
        assert(histogram_r.rows == 1);
        assert(histogram_r.cols == number_hist_bins);
        assert(histogram_r.channels() == 1);
    }

    // Resize joint histogram
    if (histogram_rt.empty()) {
        histogram_rt.create<float>(number_hist_bins, number_hist_bins, 1);
    } else {
        assert(histogram_rt.type() == Mat::Type::FLOAT32);
        assert(histogram_rt.rows == number_hist_bins);
        assert(histogram_rt.cols == number_hist_bins);
        assert(histogram_rt.channels() == 1);
    }

    // Resize joint histogram gradient
    if (histogram_rt_grad.empty()) {
        histogram_rt_grad.create<float>(
            number_hist_bins, number_hist_bins, number_transform_params);
    } else {
        assert(histogram_rt_grad.type() == Mat::Type::FLOAT32);
        assert(histogram_rt_grad.rows == number_hist_bins);
        assert(histogram_rt_grad.cols == number_hist_bins);
        assert(histogram_rt_grad.channels() == number_transform_params);
    }

    // Create histogram iterators
    Mat::Iterator<float> hist_r_it(histogram_r);
    Mat::Iterator<float> hist_rt_it(histogram_rt);
    Mat::Iterator<float> hist_rt_grad_it(histogram_rt_grad);

    // Initialize histograms
    histogram_r.fill<float>(0.f);
    histogram_rt.fill<float>(0.f);
    histogram_rt_grad.fill<float>(0.f);

    // Storage for how much each pixel contribute to all histogram bins that are
    // influenced by its value
    WeightArray<BinningMethod> bin_weights_r{};
    WeightArray<BinningMethod> bin_weights_r_derivative{};
    WeightArray<BinningMethod> bin_weights_t{};

    for (int y = 0; y < img_reference.rows; ++y) {
        for (int x = 0; x < img_reference.cols; ++x) {
            int lower_bin_r;
            int upper_bin_r;
            float pixel_bin_r;

            int lower_bin_t;
            int upper_bin_t;
            float pixel_bin_t;

            // Compute pixel weight due to masking at borders
            const float mask_weight_r     = mask_reference_it(y, x, 0) / 255.f;
            const float mask_weight_t     = mask_tracked_it(y, x, 0) / 255.f;
            const float joint_mask_weight = mask_weight_r * mask_weight_t;

            const bool pixel_mask = (mask_reference_it(y, x, 0) != 0) &&
                                    (mask_tracked_it(y, x, 0) != 0);

            assert(0.f <= mask_weight_r && mask_weight_r <= 1.f);
            assert(0.f <= mask_weight_t && mask_weight_t <= 1.f);

            if (pixel_mask) {
                // Get contribution data for reference image
                compute_bin_contribution_range<PixelType, BinningMethod>(
                    y,
                    x,
                    img_r_it,
                    pixel_bin_r,
                    lower_bin_r,
                    upper_bin_r,
                    bin_weights_r);

                // Update reference image histogram and its derivative
                int bin_weights_idx = 0;
                for (int neighbor = lower_bin_r; neighbor <= upper_bin_r;
                     ++neighbor) {
                    const float distance_to_neighbor =
                        static_cast<float>(neighbor) - pixel_bin_r;

                    assert(std::abs(distance_to_neighbor) <
                           (BinningMethod::influence_margin() + 1));

                    bin_weights_r_derivative[bin_weights_idx] =
                        BinningMethod::hbf_derivative(distance_to_neighbor);

                    hist_r_it(0, neighbor, 0) +=
                        mask_weight_r * bin_weights_r[bin_weights_idx];

                    bin_weights_idx++;
                }

                // Get contribution data for tracked image
                compute_bin_contribution_range<PixelType, BinningMethod>(
                    y,
                    x,
                    img_t_it,
                    pixel_bin_t,
                    lower_bin_t,
                    upper_bin_t,
                    bin_weights_t);

                ///
                // Update joint histogram and its gradient
                const GradPixelType* steepest_grad_ptr =
                    &steepest_grad_it(y, x, 0);

                for (int neighbor_t = lower_bin_t; neighbor_t <= upper_bin_t;
                     ++neighbor_t) {
                    const int bin_weight_t_idx = neighbor_t - lower_bin_t;

                    for (int neighbor_r = lower_bin_r;
                         neighbor_r <= upper_bin_r;
                         ++neighbor_r) {

                        const int bin_weight_r_idx = neighbor_r - lower_bin_r;

                        // Update joint histogram
                        hist_rt_it(neighbor_t, neighbor_r, 0) +=
                            joint_mask_weight *
                            bin_weights_r[bin_weight_r_idx] *
                            bin_weights_t[bin_weight_t_idx];

                        // Update joint histogram gradient
                        const float grad_weights =
                            joint_mask_weight *
                            -bin_weights_t[bin_weight_t_idx] *
                            bin_weights_r_derivative[bin_weight_r_idx];

                        float* grad_ptr =
                            &hist_rt_grad_it(neighbor_t, neighbor_r, 0);

                        for (int param = 0; param < number_transform_params;
                             ++param) {
                            grad_ptr[param] +=
                                grad_weights * steepest_grad_ptr[param];
                        }
                    }
                }
            }
        }
    }

    // Get histogram summations
    histogram_r_sum  = sum_histogram(hist_r_it);
    histogram_rt_sum = sum_histogram(hist_rt_it);

    // Normalize histograms
    normalize_histogram(histogram_r_sum, hist_r_it);
    normalize_histogram(histogram_rt_sum, hist_rt_it);
    normalize_histogram(histogram_rt_sum, hist_rt_grad_it);

    return;
}

template <typename PixelType,
          typename GradPixelType,
          typename BinningMethod,
          typename TransformClass,
          typename MaskIteratorR,
          typename MaskIteratorT>
void joint_hist_hessian(const Mat& img_reference,
                        const MaskIteratorR& mask_reference_it,
                        const Mat& steepest_grad_r,
                        const Mat& steepest_hess_r,
                        const Mat& img_tracked,
                        const MaskIteratorT& mask_tracked_it,
                        const double histogram_r_sum,
                        const double histogram_rt_sum,
                        Mat& histogram_r_grad,
                        Mat& histogram_rt_hess)
{
    const int number_transform_params = TransformClass::number_parameters;
    const int number_hist_bins =
        HistogramConfig::num_total_bins<BinningMethod>();

    // clang-format off
    using SteepestColType  = Eigen::Matrix<float,
                                           number_transform_params,
                                           1>;
    using SteepestRowType = Eigen::Matrix<float,
                                           1,
                                           number_transform_params>;
    using HessianMatType   = Eigen::Matrix<float,
                                           number_transform_params,
                                           number_transform_params,
                                           Eigen::RowMajor>;
    // clang-format on

    // This function works for single channel images only
    assert(img_reference.channels() == 1);
    assert(img_tracked.channels() == 1);

    // The input images must have the same dimensions
    assert(img_reference.rows == img_tracked.rows);
    assert(img_reference.cols == img_tracked.cols);

    // The steepest images must be of same width and height as the input images
    assert(img_reference.rows == steepest_grad_r.rows &&
           img_reference.cols == steepest_grad_r.cols);
    assert(img_reference.rows == steepest_hess_r.rows &&
           img_reference.cols == steepest_hess_r.cols);

    // The input masks must have been correctly initialized
    assert(mask_reference_it.is_mask_of(img_reference));
    assert(mask_tracked_it.is_mask_of(img_tracked));

    // The joint histogram sum must be valid
    assert(histogram_rt_sum > 0.f);

    // The images derived from the gradient must be of the same type
    assert(steepest_grad_r.type() == Mat::get_type_enum<GradPixelType>());
    assert(steepest_grad_r.type() == steepest_hess_r.type());

    // The provided steepest images must have been generated with the same
    // TransformClass
    assert(steepest_grad_r.channels() == number_transform_params);
    assert(steepest_hess_r.channels() ==
           number_transform_params * number_transform_params);

    // The histogram sums must not be zero
    assert(histogram_r_sum > 0.0);
    assert(histogram_rt_sum > 0.0);

    // Create iterators for input images
    Mat::ConstIterator<PixelType> img_r_it(img_reference);
    Mat::ConstIterator<PixelType> img_t_it(img_tracked);
    Mat::ConstIterator<GradPixelType> steepest_grad_it(steepest_grad_r);
    Mat::ConstIterator<GradPixelType> steepest_hess_it(steepest_hess_r);

    // Storage for how much each pixel contribute to all histogram bins that are
    // influenced by its value
    WeightArray<BinningMethod> bin_weights_r{};
    WeightArray<BinningMethod> bin_weights_r_derivative{};
    WeightArray<BinningMethod> bin_weights_r_second_derivative{};
    WeightArray<BinningMethod> bin_weights_t{};

    // Allocate gradient output
    if (histogram_r_grad.empty()) {
        histogram_r_grad.create<float>(
            1, number_hist_bins, number_transform_params);
    } else {
        assert(histogram_r_grad.type() == Mat::Type::FLOAT32);
        assert(histogram_r_grad.rows == 1);
        assert(histogram_r_grad.cols == number_hist_bins);
        assert(histogram_r_grad.channels() == number_transform_params);
    }

    // Allocate hessian output
    if (histogram_rt_hess.empty()) {
        histogram_rt_hess.create<float>(number_hist_bins,
                                        number_hist_bins,
                                        number_transform_params *
                                            number_transform_params);
    } else {
        assert(histogram_rt_hess.type() == Mat::Type::FLOAT32);
        assert(histogram_rt_hess.rows == number_hist_bins);
        assert(histogram_rt_hess.cols == number_hist_bins);
        assert(histogram_rt_hess.channels() ==
               number_transform_params * number_transform_params);
    }

    // Initialize histograms
    histogram_r_grad.fill<float>(0.f);
    histogram_rt_hess.fill<float>(0.f);

    // Create iterator for output histograms
    Mat::Iterator<float> hist_r_grad_it(histogram_r_grad);
    Mat::Iterator<float> hist_rt_hess_it(histogram_rt_hess);

    // Iterate over all intersection pixels
    for (int y = 0; y < img_reference.rows; ++y) {
        for (int x = 0; x < img_reference.cols; ++x) {
            int lower_bin_r;
            int upper_bin_r;
            float pixel_bin_r;

            int lower_bin_t;
            int upper_bin_t;
            float pixel_bin_t;

            // Compute pixel weight due to masking at borders
            const float mask_weight_r     = mask_reference_it(y, x, 0) / 255.f;
            const float mask_weight_t     = mask_tracked_it(y, x, 0) / 255.f;
            const float joint_mask_weight = mask_weight_r * mask_weight_t;

            assert(0.f <= mask_weight_r && mask_weight_r <= 1.f);
            assert(0.f <= mask_weight_t && mask_weight_t <= 1.f);

            const bool pixel_mask = (mask_reference_it(y, x, 0) != 0) &&
                                    (mask_tracked_it(y, x, 0) != 0);

            if (pixel_mask) {
                // Get contribution data for reference image
                compute_bin_contribution_range<PixelType, BinningMethod>(
                    y,
                    x,
                    img_r_it,
                    pixel_bin_r,
                    lower_bin_r,
                    upper_bin_r,
                    bin_weights_r);

                // Update reference image histogram and its derivative
                int bin_weights_idx = 0;
                for (int neighbor = lower_bin_r; neighbor <= upper_bin_r;
                     ++neighbor) {
                    const float distance_to_neighbor =
                        static_cast<float>(neighbor) - pixel_bin_r;

                    assert(std::abs(distance_to_neighbor) <
                           (BinningMethod::influence_margin() + 1));

                    bin_weights_r_derivative[bin_weights_idx] =
                        BinningMethod::hbf_derivative(distance_to_neighbor);

                    bin_weights_r_second_derivative[bin_weights_idx] =
                        BinningMethod::hbf_second_derivative(
                            distance_to_neighbor);

                    bin_weights_idx++;
                }

                // Get contribution data for tracked image
                compute_bin_contribution_range<PixelType, BinningMethod>(
                    y,
                    x,
                    img_t_it,
                    pixel_bin_t,
                    lower_bin_t,
                    upper_bin_t,
                    bin_weights_t);

                // Create Eigen support structures that only depend on the x,y
                // coords
                Eigen::Map<const HessianMatType> steepest_hess_mat(
                    &steepest_hess_it(y, x, 0));
                Eigen::Map<const SteepestColType> steepest_grad_col(
                    &steepest_grad_it(y, x, 0));
                Eigen::Map<const SteepestRowType> steepest_grad_row(
                    &steepest_grad_it(y, x, 0));

                for (int neighbor_r = lower_bin_r; neighbor_r <= upper_bin_r;
                     ++neighbor_r) {

                    const int bin_weight_r_idx = neighbor_r - lower_bin_r;

                    assert(bin_weight_r_idx >= 0 &&
                           bin_weight_r_idx < number_hist_bins);

                    Eigen::Map<SteepestRowType> grad_row(
                        &hist_r_grad_it(0, neighbor_r, 0));

                    ///
                    // Evaluate gradient expression
                    grad_row += mask_weight_r *
                                -bin_weights_r_derivative[bin_weight_r_idx] *
                                steepest_grad_row;

                    for (int neighbor_t = lower_bin_t;
                         neighbor_t <= upper_bin_t;
                         ++neighbor_t) {
                        const int bin_weight_t_idx = neighbor_t - lower_bin_t;

                        assert(bin_weight_t_idx >= 0 &&
                               bin_weight_t_idx < number_hist_bins);

                        // Create Eigen support structures that depend on the
                        // current bin
                        Eigen::Map<HessianMatType> hess_mat(
                            &hist_rt_hess_it(neighbor_t, neighbor_r, 0));

                        ///
                        // Evaluate Hessian expression
                        HessianMatType hessian_expression =
                            bin_weights_r_derivative[bin_weight_r_idx] *
                                steepest_hess_mat -
                            bin_weights_r_second_derivative[bin_weight_r_idx] *
                                steepest_grad_col * steepest_grad_row;

                        // Update Histogram Hessian
                        hess_mat += joint_mask_weight *
                                    -bin_weights_t[bin_weight_t_idx] *
                                    hessian_expression;
                    }
                }
            }
        }
    }

    // Normalize output histograms
    normalize_histogram(histogram_r_sum, hist_r_grad_it);
    normalize_histogram(histogram_rt_sum, hist_rt_hess_it);

    return;
}

#endif
