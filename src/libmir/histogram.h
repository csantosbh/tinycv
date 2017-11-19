#ifndef _LIBMIR_HISTOGRAM_H_
#define _LIBMIR_HISTOGRAM_H_

#include <cmath>

#include <Eigen/Eigen>

#include "mat.h"

static const int NUM_HISTOGRAM_CENTRAL_BINS = 8;

/**
 * Kronecker's binary function
 */
class KroneckerFunction
{
  public:
    static const int INFLUENCE_MARGIN = 0;

    static float histogram_bin_function(float i)
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
    static const int INFLUENCE_MARGIN = 2;

    static float histogram_bin_function(float i)
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
    static float hbf_derivative(float i)
    {
        const auto b_spline_3 = [](float i) -> float {
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
    static float hbf_second_derivative(float i)
    {
        const auto b_spline_2 = [](float i) {
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
 * Creates an image by mapping the ranges [0, 255] to [-0.5,
 * NUM_HISTOGRAM_CENTRAL_BINS-0.5]. When the output image has a floating point
 * type, this is equivalent to normalizing the image.
 */
template <typename InputPixelType, typename OutputPixelType>
Mat image_scale_histogram(const Mat& input_img)
{
    Mat output_img;
    output_img.create<OutputPixelType>(
        input_img.rows, input_img.cols, input_img.channels());

    // Compute coefficients of linear bin mapping function
    const InputPixelType color_max = static_cast<InputPixelType>(255);
    const float a_bin_map = static_cast<float>(NUM_HISTOGRAM_CENTRAL_BINS) /
                            static_cast<float>(color_max);
    const float b_bin_map = -0.5;

    Mat::ConstIterator<InputPixelType> input_it(input_img);
    output_img.for_each<Mat::Iterator<OutputPixelType>>(
        [&input_it, a_bin_map, b_bin_map](
            Mat::Iterator<OutputPixelType>& output_it, int y, int x, int c) {
            output_it(y, x, c) = static_cast<OutputPixelType>(
                a_bin_map * input_it(y, x, c) + b_bin_map);
        });

    return output_img;
}

double sum_histogram(const Mat::ConstIterator<float>& histogram)
{
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

void normalize_histogram(double histogram_summation,
                         Mat::Iterator<float>& histogram)
{
    float histogram_normalization =
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
 * Build histograms for two images, as well as their joint histogram
 *
 * The output is rowmajor, with rows representing the bins for
 * image_a, and the columns representing the bins for image_b.
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
 * where a = NUM_BINS / COLOR_MAX and
 *       b = -0.5.
 *
 * In the example above, a = 8 / 255.
 */
template <typename PixelType,
          typename BinningMethod,
          typename MaskIteratorA,
          typename MaskIteratorB>
void joint_image_histogram(const Mat& image_a,
                           const MaskIteratorA& mask_a_it,
                           const Mat& image_b,
                           const MaskIteratorB& mask_b_it,
                           Mat& histogram_a,
                           Mat& histogram_b,
                           Mat& histogram_ab)
{
    // This function works for single channel images only
    assert(image_a.channels() == 1);
    assert(image_b.channels() == 1);

    // The input images must have the same dimensions
    assert(image_a.rows == image_b.rows);
    assert(image_a.cols == image_b.cols);

    // The input masks must have been correctly initialized
    assert(mask_a_it.is_mask_of(image_a));
    assert(mask_b_it.is_mask_of(image_b));

    // Create iterators for input images
    Mat::ConstIterator<PixelType> img_a_it(image_a);
    Mat::ConstIterator<PixelType> img_b_it(image_b);

    const int hist_length =
        NUM_HISTOGRAM_CENTRAL_BINS + 2 * BinningMethod::INFLUENCE_MARGIN;

    // Resize histogram a
    if (histogram_a.empty()) {
        histogram_a.create<float>(1, hist_length, 1);
    } else {
        assert(histogram_a.type() == Mat::Type::FLOAT32);
        assert(histogram_a.rows == 1);
        assert(histogram_a.cols == hist_length);
        assert(histogram_a.channels() == 1);
    }

    // Resize histogram b
    if (histogram_b.empty()) {
        histogram_b.create<float>(1, hist_length, 1);
    } else {
        assert(histogram_b.type() == Mat::Type::FLOAT32);
        assert(histogram_b.rows == 1);
        assert(histogram_b.cols == hist_length);
        assert(histogram_b.channels() == 1);
    }

    // Resize histogram ab
    if (histogram_ab.empty()) {
        histogram_ab.create<float>(hist_length, hist_length, 1);
    } else {
        assert(histogram_ab.type() == Mat::Type::FLOAT32);
        assert(histogram_ab.rows == hist_length);
        assert(histogram_ab.cols == hist_length);
        assert(histogram_ab.channels() == 1);
    }

    // Create iterators for histograms
    Mat::Iterator<float> hist_a_it(histogram_a);
    Mat::Iterator<float> hist_b_it(histogram_b);
    Mat::Iterator<float> hist_ab_it(histogram_ab);

    // Initialize histograms
    histogram_a.fill<float>(0.f);
    histogram_b.fill<float>(0.f);
    histogram_ab.fill<float>(0.f);

    // Store how much each pixel contribute to all histogram bins that are
    // influenced by its value
    using WeightArray =
        std::array<float, 2 * BinningMethod::INFLUENCE_MARGIN + 1>;
    WeightArray bin_weights_a{};
    WeightArray bin_weights_b{};

    // Helper function for computing which histogram bins are affected by a
    // pixel, and how much
    const auto updateImageHistogram =
        [](int y,
           int x,
           const Mat::ConstIterator<PixelType>& img_it,
           float pixel_mask_weight,
           float& pixel_bin,
           int& lower_bin,
           int& upper_bin,
           WeightArray& bin_weights,
           Mat::Iterator<float>& histogram_it) {
            const PixelType& pixel_val = img_it(y, x, 0);

            // TODO move to binning module
            // Compute pixel bin coordinate
            int pixel_bin_rounded = static_cast<int>(pixel_val + 0.5);
            pixel_bin             = pixel_val + BinningMethod::INFLUENCE_MARGIN;

            assert(std::abs(static_cast<float>(pixel_bin_rounded) -
                            pixel_val) <= 0.5f);

            // Compute bin index range for the weight vector
            lower_bin = pixel_bin_rounded;
            upper_bin = pixel_bin_rounded + 2 * BinningMethod::INFLUENCE_MARGIN;

            assert(0 <= lower_bin);
            assert(lower_bin <= upper_bin);
            assert(upper_bin <= NUM_HISTOGRAM_CENTRAL_BINS +
                                    2 * BinningMethod::INFLUENCE_MARGIN - 1);

            // Update weight storage for given pixel
            int bin_weights_idx = 0;
            for (int neighbor = lower_bin; neighbor <= upper_bin; ++neighbor) {
                const float distance_to_neighbor =
                    static_cast<float>(neighbor) - pixel_bin;

                assert(std::abs(distance_to_neighbor) <
                       (BinningMethod::INFLUENCE_MARGIN + 1));

                bin_weights[bin_weights_idx] =
                    BinningMethod::histogram_bin_function(distance_to_neighbor);
                histogram_it(0, neighbor, 0) +=
                    pixel_mask_weight * bin_weights[bin_weights_idx];
                bin_weights_idx++;
            }
        };

    for (int y = 0; y < image_a.rows; ++y) {
        for (int x = 0; x < image_a.cols; ++x) {
            int lower_bin_a;
            int upper_bin_a;
            float pixel_bin_a;

            int lower_bin_b;
            int upper_bin_b;
            float pixel_bin_b;

            // Compute pixel weight due to masking at borders
            float mask_weight_a     = mask_a_it(y, x, 0) / 255.f;
            float mask_weight_b     = mask_b_it(y, x, 0) / 255.f;
            float joint_mask_weight = mask_weight_a * mask_weight_b;

            const bool pixel_mask =
                (mask_a_it(y, x, 0) != 0) && (mask_b_it(y, x, 0) != 0);

            if (pixel_mask) {
                // Update histogram_a
                updateImageHistogram(y,
                                     x,
                                     img_a_it,
                                     mask_weight_a,
                                     pixel_bin_a,
                                     lower_bin_a,
                                     upper_bin_a,
                                     bin_weights_a,
                                     hist_a_it);

                // Update histogram_b
                updateImageHistogram(y,
                                     x,
                                     img_b_it,
                                     mask_weight_b,
                                     pixel_bin_b,
                                     lower_bin_b,
                                     upper_bin_b,
                                     bin_weights_b,
                                     hist_b_it);

                // Update histogram_ab
                for (int neighbor_b = lower_bin_b; neighbor_b <= upper_bin_b;
                     ++neighbor_b) {
                    const int bin_weight_b_idx = neighbor_b - lower_bin_b;

                    for (int neighbor_a = lower_bin_a;
                         neighbor_a <= upper_bin_a;
                         ++neighbor_a) {

                        const int bin_weight_a_idx = neighbor_a - lower_bin_a;

                        hist_ab_it(neighbor_b, neighbor_a, 0) +=
                            joint_mask_weight *
                            bin_weights_a[bin_weight_a_idx] *
                            bin_weights_b[bin_weight_b_idx];
                    }
                }
            }
        }
    }

    // Normalize histograms
    normalize_histogram(sum_histogram(hist_a_it), hist_a_it);
    normalize_histogram(sum_histogram(hist_b_it), hist_b_it);
    normalize_histogram(sum_histogram(hist_ab_it), hist_ab_it);

    return;
}

template <typename BinningMethod>
using WeightArray = std::array<float, 2 * BinningMethod::INFLUENCE_MARGIN + 1>;

/**
 * Helper function for computing which histogram bins are affected by a pixel,
 * and how much
 */
template <typename PixelType, typename BinningMethod>
void computeBinContributionRange(int y,
                                 int x,
                                 const Mat::ConstIterator<PixelType>& img_it,
                                 float& pixel_bin,
                                 int& lower_bin,
                                 int& upper_bin,
                                 WeightArray<BinningMethod>& bin_weights)
{
    const PixelType& pixel_val = img_it(y, x, 0);

    // Compute pixel bin coordinate
    int pixel_bin_rounded = static_cast<int>(pixel_val + 0.5);
    pixel_bin             = pixel_val + BinningMethod::INFLUENCE_MARGIN;

    assert(std::abs(static_cast<float>(pixel_bin_rounded) - pixel_val) <= 0.5f);

    // Compute bin index range for the weight vector
    lower_bin = pixel_bin_rounded;
    upper_bin = pixel_bin_rounded + 2 * BinningMethod::INFLUENCE_MARGIN;

    // TODO the pixel val can go up to 7.5 (when pix=255). I have to deal with
    // it.
    assert(0 <= lower_bin);
    assert(lower_bin <= upper_bin);
    assert(upper_bin <= NUM_HISTOGRAM_CENTRAL_BINS +
                            2 * BinningMethod::INFLUENCE_MARGIN - 1);

    // Update weight storage for given pixel
    int bin_weights_idx = 0;
    for (int neighbor = lower_bin; neighbor <= upper_bin; ++neighbor) {
        const float distance_to_neighbor =
            static_cast<float>(neighbor) - pixel_bin;

        assert(std::abs(distance_to_neighbor) <
               (BinningMethod::INFLUENCE_MARGIN + 1));

        bin_weights[bin_weights_idx] =
            BinningMethod::histogram_bin_function(distance_to_neighbor);
        bin_weights_idx++;
    }
}

template <typename PixelType,
          typename SteepestType,
          typename BinningMethod,
          typename MaskIteratorA,
          typename MaskIteratorB>
void joint_hist_gradient(const Mat& reference,
                         const MaskIteratorA& mask_reference_it,
                         const Mat& steepest_ref_img,
                         const Mat& tracked,
                         const MaskIteratorB& mask_tracked_it,
                         double& histogram_r_sum,
                         double& histogram_rt_sum,
                         Mat& histogram_r,
                         Mat& histogram_rt,
                         Mat& histogram_rt_grad)
{
    // This function works for single channel images only
    assert(reference.channels() == 1);
    assert(tracked.channels() == 1);

    // The input images must have the same dimensions
    assert(reference.rows == tracked.rows);
    assert(reference.cols == tracked.cols);

    // The steepest image must be of same width and height as the input images
    assert(reference.rows == steepest_ref_img.rows);
    assert(reference.cols == steepest_ref_img.cols);

    // The input masks must have been correctly initialized
    assert(mask_reference_it.is_mask_of(reference));
    assert(mask_tracked_it.is_mask_of(tracked));

    // Create iterators for input images
    Mat::ConstIterator<PixelType> img_r_it(reference);
    Mat::ConstIterator<PixelType> img_t_it(tracked);
    Mat::ConstIterator<SteepestType> steepest_it(steepest_ref_img);

    ///
    // Allocate and initialize histograms
    // TODO receive TransformClass instead; will also be able to run some
    // asserts
    const int hist_length =
        NUM_HISTOGRAM_CENTRAL_BINS + 2 * BinningMethod::INFLUENCE_MARGIN;
    const int model_num_params = steepest_ref_img.channels();

    // Resize histogram of reference image
    if (histogram_r.empty()) {
        histogram_r.create<float>(1, hist_length, 1);
    } else {
        assert(histogram_r.type() == Mat::Type::FLOAT32);
        assert(histogram_r.rows == 1);
        assert(histogram_r.cols == hist_length);
        assert(histogram_r.channels() == 1);
    }

    // Resize joint histogram
    if (histogram_rt.empty()) {
        histogram_rt.create<float>(hist_length, hist_length, 1);
    } else {
        assert(histogram_rt.type() == Mat::Type::FLOAT32);
        assert(histogram_rt.rows == hist_length);
        assert(histogram_rt.cols == hist_length);
        assert(histogram_rt.channels() == 1);
    }

    // Resize joint histogram gradient
    if (histogram_rt_grad.empty()) {
        histogram_rt_grad.create<float>(
            hist_length, hist_length, model_num_params);
    } else {
        assert(histogram_rt_grad.type() == Mat::Type::FLOAT32);
        assert(histogram_rt_grad.rows == hist_length);
        assert(histogram_rt_grad.cols == hist_length);
        assert(histogram_rt_grad.channels() == model_num_params);
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

    for (int y = 0; y < reference.rows; ++y) {
        for (int x = 0; x < reference.cols; ++x) {
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

            assert(0.f <= mask_weight_r && mask_weight_r <= 1.f);
            assert(0.f <= mask_weight_t && mask_weight_t <= 1.f);

            if (pixel_mask) {
                // Get contribution data for reference image
                computeBinContributionRange<PixelType, BinningMethod>(
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
                           (BinningMethod::INFLUENCE_MARGIN + 1));

                    bin_weights_r_derivative[bin_weights_idx] =
                        BinningMethod::hbf_derivative(distance_to_neighbor);

                    hist_r_it(0, neighbor, 0) +=
                        mask_weight_r * bin_weights_r[bin_weights_idx];

                    bin_weights_idx++;
                }

                // Get contribution data for tracked image
                computeBinContributionRange<PixelType, BinningMethod>(
                    y,
                    x,
                    img_t_it,
                    pixel_bin_t,
                    lower_bin_t,
                    upper_bin_t,
                    bin_weights_t);

                ///
                // Update joint histogram and its gradient
                const SteepestType* steepest_ptr = &steepest_it(y, x, 0);

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
                        float grad_weights =
                            joint_mask_weight *
                            -bin_weights_t[bin_weight_t_idx] *
                            bin_weights_r_derivative[bin_weight_r_idx];

                        float* grad_ptr =
                            &hist_rt_grad_it(neighbor_t, neighbor_r, 0);

                        for (int param = 0; param < model_num_params; ++param) {
                            grad_ptr[param] +=
                                grad_weights * steepest_ptr[param];
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
          typename SteepestType,
          typename BinningMethod,
          typename TransformClass,
          typename MaskIteratorA,
          typename MaskIteratorB>
void joint_hist_hessian(const Mat& reference,
                        const MaskIteratorA& mask_reference_it,
                        const Mat& steepest_ref_grad,
                        const Mat& steepest_ref_hess,
                        const Mat& tracked,
                        const MaskIteratorB& tracked_mask_it,
                        const double histogram_r_sum,
                        const double histogram_rt_sum,
                        Mat& histogram_r_grad,
                        Mat& histogram_rt_hess)
{
    const int transform_params = TransformClass::number_parameters;
    const int hist_length =
        NUM_HISTOGRAM_CENTRAL_BINS + 2 * BinningMethod::INFLUENCE_MARGIN;

    // clang-format off
    using SteepestColType  = Eigen::Matrix<float,
                                           transform_params,
                                           1>;
    using SteepestRowType = Eigen::Matrix<float,
                                           1,
                                           transform_params>;
    using HessianMatType   = Eigen::Matrix<float,
                                           transform_params,
                                           transform_params,
                                           Eigen::RowMajor>;
    // clang-format on

    // This function works for single channel images only
    assert(reference.channels() == 1);
    assert(tracked.channels() == 1);

    // The input images must have the same dimensions
    assert(reference.rows == tracked.rows);
    assert(reference.cols == tracked.cols);

    // The steepest images must be of same width and height as the input images
    assert(reference.rows == steepest_ref_grad.rows &&
           reference.cols == steepest_ref_grad.cols);
    assert(reference.rows == steepest_ref_hess.rows &&
           reference.cols == steepest_ref_hess.cols);

    // The input masks must have been correctly initialized
    assert(mask_reference_it.is_mask_of(reference));
    assert(tracked_mask_it.is_mask_of(tracked));

    // The joint histogram sum must be valid
    assert(histogram_rt_sum > 0.f);

    // This was only tested with float steepest hessian and grad
    assert(steepest_ref_grad.type() == Mat::Type::FLOAT32);
    assert(steepest_ref_hess.type() == Mat::Type::FLOAT32);

    // The provided steepest images must have been generated with the same
    // TransformClass
    assert(steepest_ref_grad.channels() == transform_params);
    assert(steepest_ref_hess.channels() == transform_params * transform_params);

    // The histogram sums must not be zero
    assert(histogram_r_sum > 0.0);
    assert(histogram_rt_sum > 0.0);

    // Create iterators for input images
    Mat::ConstIterator<PixelType> img_r_it(reference);
    Mat::ConstIterator<PixelType> img_t_it(tracked);
    Mat::ConstIterator<SteepestType> steepest_grad_it(steepest_ref_grad);
    Mat::ConstIterator<SteepestType> steepest_hess_it(steepest_ref_hess);

    // Storage for how much each pixel contribute to all histogram bins that are
    // influenced by its value
    WeightArray<BinningMethod> bin_weights_r{};
    WeightArray<BinningMethod> bin_weights_r_derivative{};
    WeightArray<BinningMethod> bin_weights_r_second_derivative{};
    WeightArray<BinningMethod> bin_weights_t{};

    // Allocate gradient output
    if (histogram_r_grad.empty()) {
        histogram_r_grad.create<float>(1, hist_length, transform_params);
    } else {
        assert(histogram_r_grad.type() == Mat::Type::FLOAT32);
        assert(histogram_r_grad.rows == 1);
        assert(histogram_r_grad.cols == hist_length);
        assert(histogram_r_grad.channels() == transform_params);
    }

    // Allocate hessian output
    if (histogram_rt_hess.empty()) {
        histogram_rt_hess.create<float>(
            hist_length, hist_length, transform_params * transform_params);
    } else {
        assert(histogram_rt_hess.type() == Mat::Type::FLOAT32);
        assert(histogram_rt_hess.rows == hist_length);
        assert(histogram_rt_hess.cols == hist_length);
        assert(histogram_rt_hess.channels() ==
               transform_params * transform_params);
    }

    // Initialize histograms
    histogram_r_grad.fill<float>(0.f);
    histogram_rt_hess.fill<float>(0.f);

    // Create iterator for output histograms
    Mat::Iterator<float> hist_r_grad_it(histogram_r_grad);
    Mat::Iterator<float> hist_rt_hess_it(histogram_rt_hess);

    // Iterate over all intersection pixels
    for (int y = 0; y < reference.rows; ++y) {
        for (int x = 0; x < reference.cols; ++x) {
            int lower_bin_r;
            int upper_bin_r;
            float pixel_bin_r;

            int lower_bin_t;
            int upper_bin_t;
            float pixel_bin_t;

            // Compute pixel weight due to masking at borders
            float mask_weight_r     = mask_reference_it(y, x, 0) / 255.f;
            float mask_weight_t     = tracked_mask_it(y, x, 0) / 255.f;
            float joint_mask_weight = mask_weight_r * mask_weight_t;

            assert(0.f <= mask_weight_r && mask_weight_r <= 1.f);
            assert(0.f <= mask_weight_t && mask_weight_t <= 1.f);

            const bool pixel_mask = (mask_reference_it(y, x, 0) != 0) &&
                                    (tracked_mask_it(y, x, 0) != 0);

            if (pixel_mask) {
                // Get contribution data for reference image
                computeBinContributionRange<PixelType, BinningMethod>(
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
                           (BinningMethod::INFLUENCE_MARGIN + 1));

                    bin_weights_r_derivative[bin_weights_idx] =
                        BinningMethod::hbf_derivative(distance_to_neighbor);

                    bin_weights_r_second_derivative[bin_weights_idx] =
                        BinningMethod::hbf_second_derivative(
                            distance_to_neighbor);

                    bin_weights_idx++;
                }

                // Get contribution data for tracked image
                computeBinContributionRange<PixelType, BinningMethod>(
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
                           bin_weight_r_idx < hist_length);

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
                               bin_weight_t_idx < hist_length);

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
