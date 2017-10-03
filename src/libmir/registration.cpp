#include <iostream>
#include <limits>
#include <vector>

#include "mat.h"
#include "registration.h"
#include "sat.h"

// TODO move to some proper place
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
    static const int INFLUENCE_MARGIN = 1;

    static float histogram_bin_function(float i)
    {
        if (i < -2.f) {
            return 0.f;
        } else if (i < -1.f) {
            float k = (2.f + i);

            return k * k * k / 6.f;
        } else if (i < 0.f) {
            float k = 1.f + i;

            return (1.f + 3.f * k + 3 * k * k - 3.f * k * k * k) / 6.f;
        } else if (i < 1.f) {
            float k = 1.f - i;

            return (1.f + 3.f * k + 3 * k * k - 3.f * k * k * k) / 6.f;
        } else if (i < 2.f) {
            float k = (2.f - i);

            return k * k * k / 6.f;
        } else {
            assert(i >= 2.f);

            return 0.f;
        }
    }
};

/**
 * Build image histogram
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

template <typename PixelType, typename BinningMethod>
void image_histogram(const Mat& image, std::vector<float>& histogram)
{
    // This function works for single channel images only
    assert(image.step.buf[1] == 1);

    Mat::ConstIterator<PixelType> it_img(image);

    histogram.resize(NUM_HISTOGRAM_CENTRAL_BINS +
                     2 * BinningMethod::INFLUENCE_MARGIN);
    std::fill(histogram.begin(), histogram.end(), 0.f);

    const PixelType color_max = std::numeric_limits<PixelType>::max();

    const int bin_width =
        (static_cast<int>(color_max) + 1) / NUM_HISTOGRAM_CENTRAL_BINS;

    const float a_bin_map =
        NUM_HISTOGRAM_CENTRAL_BINS / static_cast<float>(color_max);
    const float b_bin_map = -0.5;

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const PixelType& pixel_val = it_img(y, x, 0);

            int pixel_bin_rounded = static_cast<int>(pixel_val) / bin_width;
            float pixel_bin       = a_bin_map * pixel_val + b_bin_map;

            int lower_bin =
                std::max(-BinningMethod::INFLUENCE_MARGIN,
                         pixel_bin_rounded - BinningMethod::INFLUENCE_MARGIN);
            int upper_bin =
                std::min(pixel_bin_rounded + BinningMethod::INFLUENCE_MARGIN,
                         NUM_HISTOGRAM_CENTRAL_BINS - 1 +
                             BinningMethod::INFLUENCE_MARGIN);

            for (int neighbor = lower_bin; neighbor <= upper_bin; ++neighbor) {
                const float distance_to_neighbor =
                    pixel_bin - static_cast<float>(neighbor);
                histogram[neighbor + BinningMethod::INFLUENCE_MARGIN] +=
                    BinningMethod::histogram_bin_function(distance_to_neighbor);
            }
        }
    }

    // Normalize histogram
    double histogram_summation = 0.0;
    for (const auto& h_value : histogram) {
        histogram_summation += h_value;
    }

    float histogram_normalization =
        static_cast<float>(1.0 / histogram_summation);
    for (auto& h_value : histogram) {
        h_value *= histogram_normalization;
    }

    return;
}

/**
 * Build joint image histogram
 *
 * Refer to image_histogram for detailed information about how binning is
 * performed.
 * The output is TODO(rowmajor ou colmajor), with rows representing the bins for
 * image_a, and the columns representing the bins for image_b.
 */

template <typename PixelType, typename BinningMethod>
void joint_image_histogram(const Mat& image_a,
                           const Mat& image_b,
                           std::vector<float>& histogram)
{
    // This function works for single channel images only
    assert(image_a.step.buf[1] == 1);
    assert(image_b.step.buf[1] == 1);
    // The input images must have the same dimensions
    assert(image_a.rows == image_b.rows);
    assert(image_a.cols == image_b.cols);

    Mat::ConstIterator<PixelType> it_img_a(image_a);
    Mat::ConstIterator<PixelType> it_img_b(image_b);

    const int hist_row_size =
        NUM_HISTOGRAM_CENTRAL_BINS + 2 * BinningMethod::INFLUENCE_MARGIN;
    const int total_hist_size = hist_row_size * hist_row_size;
    histogram.resize(total_hist_size);
    std::fill(histogram.begin(), histogram.end(), 0.f);

    const PixelType color_max = std::numeric_limits<PixelType>::max();

    const int bin_width =
        (static_cast<int>(color_max) + 1) / NUM_HISTOGRAM_CENTRAL_BINS;

    const float a_bin_map =
        NUM_HISTOGRAM_CENTRAL_BINS / static_cast<float>(color_max);
    const float b_bin_map = -0.5;

    const auto hist_pos = [hist_row_size](int x, int y) {
        return y * hist_row_size + x;
    };

    for (int y = 0; y < image_a.rows; ++y) {
        for (int x = 0; x < image_a.cols; ++x) {
            const PixelType& pixel_val_a = it_img_a(y, x, 0);
            const PixelType& pixel_val_b = it_img_b(y, x, 0);

            int pixel_bin_a_rounded = static_cast<int>(pixel_val_a) / bin_width;
            int pixel_bin_b_rounded = static_cast<int>(pixel_val_b) / bin_width;
            float pixel_bin_a       = a_bin_map * pixel_val_a + b_bin_map;
            float pixel_bin_b       = a_bin_map * pixel_val_b + b_bin_map;

            int lower_bin_a =
                std::max(-BinningMethod::INFLUENCE_MARGIN,
                         pixel_bin_a_rounded - BinningMethod::INFLUENCE_MARGIN);
            int lower_bin_b =
                std::max(-BinningMethod::INFLUENCE_MARGIN,
                         pixel_bin_b_rounded - BinningMethod::INFLUENCE_MARGIN);

            int upper_bin_a =
                std::min(pixel_bin_a_rounded + BinningMethod::INFLUENCE_MARGIN,
                         NUM_HISTOGRAM_CENTRAL_BINS - 1 +
                             BinningMethod::INFLUENCE_MARGIN);
            int upper_bin_b =
                std::min(pixel_bin_b_rounded + BinningMethod::INFLUENCE_MARGIN,
                         NUM_HISTOGRAM_CENTRAL_BINS - 1 +
                             BinningMethod::INFLUENCE_MARGIN);

            for (int neighbor_b = lower_bin_b; neighbor_b <= upper_bin_b;
                 ++neighbor_b) {
                const float distance_to_neighbor_b =
                    pixel_bin_b - static_cast<float>(neighbor_b);
                const float hist_bin_b = BinningMethod::histogram_bin_function(
                    distance_to_neighbor_b);

                for (int neighbor_a = lower_bin_a; neighbor_a <= upper_bin_a;
                     ++neighbor_a) {
                    const float distance_to_neighbor_a =
                        pixel_bin_a - static_cast<float>(neighbor_a);

                    const float hist_bin_a =
                        BinningMethod::histogram_bin_function(
                            distance_to_neighbor_a);

                    const int hist_row =
                        neighbor_b + BinningMethod::INFLUENCE_MARGIN;
                    const int hist_col =
                        neighbor_a + BinningMethod::INFLUENCE_MARGIN;

                    histogram[hist_pos(hist_row, hist_col)] +=
                        hist_bin_b * hist_bin_a;
                }
            }
        }
    }

    // Normalize histogram
    double histogram_summation = 0.0;
    for (const auto& h_value : histogram) {
        histogram_summation += h_value;
    }

    float histogram_normalization =
        static_cast<float>(1.0 / histogram_summation);
    for (auto& h_value : histogram) {
        h_value *= histogram_normalization;
    }

    return;
}

bool register_translation(const Mat& source,
                          const Mat& destination,
                          Eigen::Vector2f& registration)
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

    return true;
}
