#ifndef _LIBMIR_HISTOGRAM_H_
#define _LIBMIR_HISTOGRAM_H_

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
 * The output is TODO(rowmajor ou colmajor), with rows representing the bins for
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

    Mat::ConstIterator<PixelType> img_a_it(image_a);
    Mat::ConstIterator<PixelType> img_b_it(image_b);

    const int hist_row_size =
        NUM_HISTOGRAM_CENTRAL_BINS + 2 * BinningMethod::INFLUENCE_MARGIN;

    // Resize histogram a
    histogram_a.create<float>(1, hist_row_size, 1).fill<float>(0.f);

    // Resize histogram b
    histogram_b.create<float>(1, hist_row_size, 1).fill<float>(0.f);

    // Resize histogram ab
    histogram_ab.create<float>(hist_row_size, hist_row_size, 1)
        .fill<float>(0.f);

    const PixelType color_max = static_cast<PixelType>(255);

    const int bin_width =
        (static_cast<int>(color_max) + 1) / NUM_HISTOGRAM_CENTRAL_BINS;

    const float a_bin_map = static_cast<float>(NUM_HISTOGRAM_CENTRAL_BINS) /
                            static_cast<float>(color_max);
    const float b_bin_map = -0.5;

    // Iterators for histograms
    Mat::Iterator<float> hist_a_it(histogram_a);
    Mat::Iterator<float> hist_b_it(histogram_b);
    Mat::Iterator<float> hist_ab_it(histogram_ab);

    // Store how much each pixel contribute to all histogram bins that are
    // influenced by its value
    using WeightArray =
        std::array<float, 2 * BinningMethod::INFLUENCE_MARGIN + 1>;
    WeightArray bin_weights_a{};
    WeightArray bin_weights_b{};

    const auto updateImageHistogram =
        [a_bin_map, b_bin_map, bin_width](
            int y,
            int x,
            const Mat::ConstIterator<PixelType>& img_it,
            float& pixel_bin,
            int& lower_bin,
            int& upper_bin,
            WeightArray& bin_weights,
            Mat::Iterator<float>& histogram_it) {
            const PixelType& pixel_val = img_it(y, x, 0);

            int pixel_bin_rounded = static_cast<int>(pixel_val) / bin_width;
            pixel_bin             = a_bin_map * pixel_val + b_bin_map;

            lower_bin =
                std::max(-BinningMethod::INFLUENCE_MARGIN,
                         pixel_bin_rounded - BinningMethod::INFLUENCE_MARGIN);

            upper_bin =
                std::min(pixel_bin_rounded + BinningMethod::INFLUENCE_MARGIN,
                         NUM_HISTOGRAM_CENTRAL_BINS - 1 +
                             BinningMethod::INFLUENCE_MARGIN);

            int bin_weights_idx = 0;
            for (int neighbor = lower_bin; neighbor <= upper_bin; ++neighbor) {
                const float distance_to_neighbor =
                    pixel_bin - static_cast<float>(neighbor);

                bin_weights[bin_weights_idx] =
                    BinningMethod::histogram_bin_function(distance_to_neighbor);
                histogram_it(0,
                             neighbor + BinningMethod::INFLUENCE_MARGIN,
                             0) += bin_weights[bin_weights_idx];
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

            const bool pixel_mask =
                (mask_a_it(y, x, 0) != 0) && (mask_b_it(y, x, 0) != 0);

            if (pixel_mask) {
                // Update histogram_a
                updateImageHistogram(y,
                                     x,
                                     img_a_it,
                                     pixel_bin_a,
                                     lower_bin_a,
                                     upper_bin_a,
                                     bin_weights_a,
                                     hist_a_it);

                // Update histogram_b
                updateImageHistogram(y,
                                     x,
                                     img_b_it,
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

                        const int hist_row =
                            neighbor_b + BinningMethod::INFLUENCE_MARGIN;
                        const int hist_col =
                            neighbor_a + BinningMethod::INFLUENCE_MARGIN;

                        const int bin_weight_a_idx = neighbor_a - lower_bin_a;

                        hist_ab_it(hist_row, hist_col, 0) +=
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

    // Create iterators for input images
    Mat::ConstIterator<PixelType> img_r_it(reference);
    Mat::ConstIterator<PixelType> img_t_it(tracked);
    Mat::ConstIterator<SteepestType> steepest_it(steepest_ref_img);

    const int hist_length =
        NUM_HISTOGRAM_CENTRAL_BINS + 2 * BinningMethod::INFLUENCE_MARGIN;
    // Resize histogram of reference image
    histogram_r.create<float>(1, hist_length, 1).fill<float>(0.f);

    // Resize joint histogram
    histogram_rt.create<float>(hist_length, hist_length, 1).fill<float>(0.f);

    // Resize joint histogram gradient
    const int model_num_params = steepest_ref_img.channels();
    histogram_rt_grad.create<float>(hist_length, hist_length, model_num_params)
        .fill<float>(0.f);

    // Create histogram iterators
    Mat::Iterator<float> hist_r_it(histogram_r);
    Mat::Iterator<float> hist_rt_it(histogram_rt);
    Mat::Iterator<float> hist_rt_grad_it(histogram_rt_grad);

    // Compute coefficients of linear bin mapping function
    const PixelType color_max = static_cast<PixelType>(255);
    const int bin_width =
        (static_cast<int>(color_max) + 1) / NUM_HISTOGRAM_CENTRAL_BINS;
    const float a_bin_map = static_cast<float>(NUM_HISTOGRAM_CENTRAL_BINS) /
                            static_cast<float>(color_max);
    const float b_bin_map = -0.5;

    // Storage for how much each pixel contribute to all histogram bins that are
    // influenced by its value
    using WeightArray =
        std::array<float, 2 * BinningMethod::INFLUENCE_MARGIN + 1>;
    WeightArray bin_weights_r{};
    WeightArray bin_weights_r_derivative{};
    WeightArray bin_weights_t{};

    // Helper function for computing which histogram bins are affected by a
    // pixel, and how much
    const auto computeBinContributionRange =
        [a_bin_map, b_bin_map, bin_width](
            int y,
            int x,
            const Mat::ConstIterator<PixelType>& img_it,
            float& pixel_bin,
            int& lower_bin,
            int& upper_bin,
            WeightArray& bin_weights) {
            const PixelType& pixel_val = img_it(y, x, 0);

            // Compute pixel bin coordinate
            int pixel_bin_rounded = static_cast<int>(pixel_val) / bin_width;
            pixel_bin             = a_bin_map * pixel_val + b_bin_map;

            // Compute bin index range for the weight vector
            lower_bin = pixel_bin_rounded - BinningMethod::INFLUENCE_MARGIN;
            upper_bin = pixel_bin_rounded + BinningMethod::INFLUENCE_MARGIN;

            assert(-BinningMethod::INFLUENCE_MARGIN <= lower_bin);
            assert(lower_bin <= upper_bin);
            assert(upper_bin <= NUM_HISTOGRAM_CENTRAL_BINS - 1 +
                                    BinningMethod::INFLUENCE_MARGIN);

            // Update weight storage for given pixel
            int bin_weights_idx = 0;
            for (int neighbor = lower_bin; neighbor <= upper_bin; ++neighbor) {
                const float distance_to_neighbor =
                    pixel_bin - static_cast<float>(neighbor);

                bin_weights[bin_weights_idx] =
                    BinningMethod::histogram_bin_function(distance_to_neighbor);
                bin_weights_idx++;
            }
        };

    for (int y = 0; y < reference.rows; ++y) {
        for (int x = 0; x < reference.cols; ++x) {
            int lower_bin_r;
            int upper_bin_r;
            float pixel_bin_r;

            int lower_bin_t;
            int upper_bin_t;
            float pixel_bin_t;

            const bool pixel_mask = (mask_reference_it(y, x, 0) != 0) &&
                                    (mask_tracked_it(y, x, 0) != 0);

            if (pixel_mask) {
                // Get contribution data for reference image
                computeBinContributionRange(y,
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
                        pixel_bin_r - static_cast<float>(neighbor);

                    bin_weights_r_derivative[bin_weights_idx] =
                        BinningMethod::hbf_derivative(distance_to_neighbor);

                    hist_r_it(0,
                              neighbor + BinningMethod::INFLUENCE_MARGIN,
                              0) += bin_weights_r[bin_weights_idx];

                    bin_weights_idx++;
                }

                // Get contribution data for tracked image
                computeBinContributionRange(y,
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

                        const int hist_row =
                            neighbor_t + BinningMethod::INFLUENCE_MARGIN;
                        const int hist_col =
                            neighbor_r + BinningMethod::INFLUENCE_MARGIN;

                        const int bin_weight_r_idx = neighbor_r - lower_bin_r;

                        // Update joint histogram
                        hist_rt_it(hist_row, hist_col, 0) +=
                            bin_weights_r[bin_weight_r_idx] *
                            bin_weights_t[bin_weight_t_idx];

                        // Update joint histogram gradient
                        float grad_weights =
                            -bin_weights_t[bin_weight_t_idx] *
                            bin_weights_r_derivative[bin_weight_r_idx];

                        float* grad_ptr =
                            &hist_rt_grad_it(hist_row, hist_col, 0);

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
    double hist_r_sum  = sum_histogram(hist_r_it);
    double hist_rt_sum = sum_histogram(hist_rt_it);

    // Normalize histograms
    normalize_histogram(hist_r_sum, hist_r_it);
    normalize_histogram(hist_rt_sum, hist_rt_it);
    normalize_histogram(hist_rt_sum, hist_rt_grad_it);

    return;
}


#endif
