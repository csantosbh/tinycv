#ifndef _LIBMIR_HISTOGRAM_H_
#define _LIBMIR_HISTOGRAM_H_

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
                           const MaskIteratorA& it_mask_a,
                           const Mat& image_b,
                           const MaskIteratorB& it_mask_b,
                           std::vector<float>& histogram_a,
                           std::vector<float>& histogram_b,
                           std::vector<float>& histogram_ab)
{
    // This function works for single channel images only
    assert(image_a.channels() == 1);
    assert(image_b.channels() == 1);
    // The input images must have the same dimensions
    assert(image_a.rows == image_b.rows);
    assert(image_a.cols == image_b.cols);

    Mat::ConstIterator<PixelType> it_img_a(image_a);
    Mat::ConstIterator<PixelType> it_img_b(image_b);

    // Resize histogram a
    histogram_a.resize(NUM_HISTOGRAM_CENTRAL_BINS +
                       2 * BinningMethod::INFLUENCE_MARGIN);
    std::fill(histogram_a.begin(), histogram_a.end(), 0.f);

    // Resize histogram b
    histogram_b.resize(NUM_HISTOGRAM_CENTRAL_BINS +
                       2 * BinningMethod::INFLUENCE_MARGIN);
    std::fill(histogram_b.begin(), histogram_b.end(), 0.f);

    // Resize histogram ab
    const int hist_ab_row_size =
        NUM_HISTOGRAM_CENTRAL_BINS + 2 * BinningMethod::INFLUENCE_MARGIN;
    const int hist_ab_size = hist_ab_row_size * hist_ab_row_size;
    histogram_ab.resize(hist_ab_size);
    std::fill(histogram_ab.begin(), histogram_ab.end(), 0.f);

    const PixelType color_max = static_cast<PixelType>(255);

    const int bin_width =
        (static_cast<int>(color_max) + 1) / NUM_HISTOGRAM_CENTRAL_BINS;

    const float a_bin_map = static_cast<float>(NUM_HISTOGRAM_CENTRAL_BINS) /
                            static_cast<float>(color_max);
    const float b_bin_map = -0.5;

    // Helper function for accessing arbitrary 2d positions in histogram ab
    const auto hist_ab_pos = [hist_ab_row_size](int x, int y) {
        return y * hist_ab_row_size + x;
    };

    // Store how much each pixel contribute to all histogram bins that are
    // influenced by its value
    using WeightArray =
        std::array<float, 2 * BinningMethod::INFLUENCE_MARGIN + 1>;
    WeightArray bin_weights_a{};
    WeightArray bin_weights_b{};

    const auto updateImageHistogram = [a_bin_map, b_bin_map, bin_width](
                                          int y,
                                          int x,
                                          bool pixel_mask,
                                          const Mat::ConstIterator<PixelType>&
                                              it_img,
                                          float& pixel_bin,
                                          int& lower_bin,
                                          int& upper_bin,
                                          WeightArray& bin_weights,
                                          std::vector<float>& histogram) {
        if (pixel_mask) {
            const PixelType& pixel_val = it_img(y, x, 0);

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
                histogram[neighbor + BinningMethod::INFLUENCE_MARGIN] +=
                    bin_weights[bin_weights_idx];
                bin_weights_idx++;
            }
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
                (it_mask_a(y, x, 0) != 0) && (it_mask_b(y, x, 0) != 0);

            updateImageHistogram(y,
                                 x,
                                 pixel_mask,
                                 it_img_a,
                                 pixel_bin_a,
                                 lower_bin_a,
                                 upper_bin_a,
                                 bin_weights_a,
                                 histogram_a);

            updateImageHistogram(y,
                                 x,
                                 pixel_mask,
                                 it_img_b,
                                 pixel_bin_b,
                                 lower_bin_b,
                                 upper_bin_b,
                                 bin_weights_b,
                                 histogram_b);

            if (pixel_mask) {
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

                        const int joint_hist_pos =
                            hist_ab_pos(hist_row, hist_col);
                        histogram_ab[joint_hist_pos] +=
                            bin_weights_a[bin_weight_a_idx] *
                            bin_weights_b[bin_weight_b_idx];
                    }
                }
            }
        }
    }

    const auto normalizeHistogram = [](std::vector<float>& histogram) {
        double histogram_summation = 0.0;
        for (const auto& h_value : histogram) {
            histogram_summation += h_value;
        }

        float histogram_normalization =
            static_cast<float>(1.0 / histogram_summation);
        for (auto& h_value : histogram) {
            h_value *= histogram_normalization;
        }
    };

    // Normalize histograms
    normalizeHistogram(histogram_a);
    normalizeHistogram(histogram_b);
    normalizeHistogram(histogram_ab);

    return;
}


#endif
