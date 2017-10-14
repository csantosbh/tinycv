#include <iostream>
#include <limits>
#include <vector>

#include "mat.h"
#include "registration.h"
#include "sat.h"

// TODO move to some proper place
static const int NUM_HISTOGRAM_CENTRAL_BINS = 8;

using Matrix3fRowMajor = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

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


// deprecated
template <typename PixelType, typename BinningMethod, typename MaskIterator>
void image_histogram(const Mat& image,
                     const MaskIterator& it_mask,
                     std::vector<float>& histogram)
{
    // This function works for single channel images only
    assert(image.channels() == 1);

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
            if (it_mask(y, x, 0) != 0) {
                const PixelType& pixel_val = it_img(y, x, 0);

                int pixel_bin_rounded = static_cast<int>(pixel_val) / bin_width;
                float pixel_bin       = a_bin_map * pixel_val + b_bin_map;

                int lower_bin = std::max(-BinningMethod::INFLUENCE_MARGIN,
                                         pixel_bin_rounded -
                                             BinningMethod::INFLUENCE_MARGIN);
                int upper_bin = std::min(pixel_bin_rounded +
                                             BinningMethod::INFLUENCE_MARGIN,
                                         NUM_HISTOGRAM_CENTRAL_BINS - 1 +
                                             BinningMethod::INFLUENCE_MARGIN);

                for (int neighbor = lower_bin; neighbor <= upper_bin;
                     ++neighbor) {
                    const float distance_to_neighbor =
                        pixel_bin - static_cast<float>(neighbor);
                    histogram[neighbor + BinningMethod::INFLUENCE_MARGIN] +=
                        BinningMethod::histogram_bin_function(
                            distance_to_neighbor);
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

    const PixelType color_max = std::numeric_limits<PixelType>::max();

    const int bin_width =
        (static_cast<int>(color_max) + 1) / NUM_HISTOGRAM_CENTRAL_BINS;

    const float a_bin_map =
        NUM_HISTOGRAM_CENTRAL_BINS / static_cast<float>(color_max);
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

    const auto updateImageHistogram =
        [a_bin_map, b_bin_map](int y,
                               int x,
                               bool pixel_mask,
                               const Mat::ConstIterator<PixelType>& it_img,
                               float& pixel_bin,
                               int& lower_bin,
                               int& upper_bin,
                               WeightArray& bin_weights,
                               std::vector<float>& histogram) {
            if (pixel_mask) {
                const PixelType& pixel_val = it_img(y, x, 0);

                int pixel_bin_rounded = static_cast<int>(pixel_val) / bin_width;
                pixel_bin             = a_bin_map * pixel_val + b_bin_map;

                lower_bin = std::max(-BinningMethod::INFLUENCE_MARGIN,
                                     pixel_bin_rounded -
                                         BinningMethod::INFLUENCE_MARGIN);

                upper_bin = std::min(pixel_bin_rounded +
                                         BinningMethod::INFLUENCE_MARGIN,
                                     NUM_HISTOGRAM_CENTRAL_BINS - 1 +
                                         BinningMethod::INFLUENCE_MARGIN);

                int bin_weights_idx = 0;
                for (int neighbor = lower_bin; neighbor <= upper_bin;
                     ++neighbor) {
                    const float distance_to_neighbor =
                        pixel_bin - static_cast<float>(neighbor);

                    bin_weights[bin_weights_idx] =
                        BinningMethod::histogram_bin_function(
                            distance_to_neighbor);
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

/**
 * Mask iterator for the case where we know that all pixels are valid
 *
 * Designed for use with templated parameters, so the compiler can optimize away
 * conditional branches.
 */
struct PositiveMaskIterator
{
    uint8_t operator()(int, int, int) const
    {
        return 255;
    }
};

template <typename MaskIteratorA, typename MaskIteratorB>
double mutual_information_impl(const Mat& image_a,
                               const MaskIteratorA& it_mask_a,
                               const Mat& image_b,
                               const MaskIteratorB& it_mask_b)
{
    using std::vector;
    using BinningMethod = BSpline4;
    using PixelType     = uint8_t;

    vector<float> histogram_a;
    vector<float> histogram_b;
    vector<float> histogram_ab;
    joint_image_histogram<PixelType,
                          BinningMethod,
                          MaskIteratorA,
                          MaskIteratorB>(image_a,
                                         it_mask_a,
                                         image_b,
                                         it_mask_b,
                                         histogram_a,
                                         histogram_b,
                                         histogram_ab);

    const int number_practical_bins = static_cast<int>(histogram_a.size());

    const auto joint_hist_pos = [number_practical_bins](int x, int y) {
        return y * number_practical_bins + x;
    };

    double mi_summation = 0.0;
    for (int j = 0; j < number_practical_bins; ++j) {
        for (int i = 0; i < number_practical_bins; ++i) {
            double prob_ij = histogram_ab[joint_hist_pos(i, j)];

            /*
             * We know that P(a=i,b=j) < P(a=i) and P(a=i,b=j) < P(b=j), since
             * (a=i,b=j) is a subset of both (a=i) and (b=j) events.
             *
             * Therefore, if P(a=i)=0 or P(b=j)=0, then P(a=i,b=j)=0.
             *
             * Now consider P(a=i,b=j)=0. Then, the MI term
             *
             *  MI(i,j) = P(a=i,b=j) * log(P(a=i,b=j) / (P(a=i) * P(b=j)))
             *
             * must evaluate to 0.
             *
             * Proof:
             * Let k = P(a=i,b=j), l=P(a=i) and m=P(b=j), for the sake of
             * simplicity. Then:
             *
             *  MI(i,j) = lim{k->0+} k * log(k/(l * m)).
             *
             * If l > 0 and m > 0, then it is trivial to see that MI(i,j) = 0.
             * If, however, both l and m are zero, we have
             *
             *  MI(i,j) = lim{k->0+} k * log(k/(k * k))
             *          = lim{k->0+} k * log(k) - k * log(k) - k * log(k)
             *
             * Each term k * log(k) can be written as log(k) / (1/k), so one can
             * use L'hopital rule to find out that each term of the sum
             * converges to 0.
             */
            if (prob_ij > 0.0) {
                double prob_ai = histogram_a[i];
                double prob_bj = histogram_b[j];

                double logterm = std::log(prob_ij / (prob_ai * prob_bj));
                mi_summation += prob_ij * logterm;
            }

            assert(!std::isnan(mi_summation));
        }
    }

    return mi_summation;
}

/**
 * Compute the mutual information between image_a and image_b
 */
double mutual_information(const Mat& image_a, const Mat& image_b)
{
    return mutual_information_impl<PositiveMaskIterator, PositiveMaskIterator>(
        image_a, {}, image_b, {});
}

/**
 * Compute the mutual information between image_a and image_b
 *
 * @param it_mask_b  Arbitrary mask for determining valid pixels of the image_b.
 *                   A pixel is valid iff its corresponding mask pixel is not 0.
 */
double mutual_information(const Mat& image_a,
                          const Mat& image_b,
                          const Mat::ConstIterator<uint8_t>& it_mask_b)
{
    return mutual_information_impl<PositiveMaskIterator,
                                   Mat::ConstIterator<uint8_t>>(
        image_a, {}, image_b, it_mask_b);
}

/**
 * Compute the mutual information between image_a and image_b
 *
 * @param it_mask_a  Arbitrary mask for determining valid pixels of the image_a.
 *                   A pixel is valid iff its corresponding mask pixel is not 0.
 * @param it_mask_b  Arbitrary mask for determining valid pixels of the image_b.
 *                   A pixel is valid iff its corresponding mask pixel is not 0.
 */
double mutual_information(const Mat& image_a,
                          const Mat::ConstIterator<uint8_t>& it_mask_a,
                          const Mat& image_b,
                          const Mat::ConstIterator<uint8_t>& it_mask_b)
{
    return mutual_information_impl<Mat::ConstIterator<uint8_t>,
                                   Mat::ConstIterator<uint8_t>>(
        image_a, it_mask_a, image_b, it_mask_b);
}

/**
 * Sample pixel at arbitrary coordinate using bilinear interpolation
 *
 * Handles multichannel images gracefully.
 *
 * @param coordinates  Pointer to array of two floats <x,y>.
 * @param output       Pointer to output pixel.
 */
template <typename PixelType, int channels>
void bilinear_sample(const Mat::ConstIterator<PixelType>& it_img,
                     const float* coordinates,
                     PixelType* output)
{
    float horiz_alpha      = coordinates[0] - std::floor(coordinates[0]);
    float vert_alpha       = coordinates[1] - std::floor(coordinates[1]);
    float horiz_alpha_comp = 1.f - horiz_alpha;
    float vert_alpha_comp  = 1.f - vert_alpha;

    int coordinates_i[] = {static_cast<int>(coordinates[0]),
                           static_cast<int>(coordinates[1])};

    int left   = std::max(0, coordinates_i[0]);
    int right  = std::min(coordinates_i[0] + 1, it_img.m.cols - 1);
    int top    = std::max(0, coordinates_i[1]);
    int bottom = std::min(coordinates_i[1] + 1, it_img.m.rows - 1);

    assert(0 <= left);
    assert(left <= right);
    assert(right < it_img.m.cols);

    assert(0 <= top);
    assert(top <= bottom);
    assert(bottom < it_img.m.rows);

    const PixelType* tl = &it_img(top, left, 0);
    const PixelType* tr = &it_img(top, right, 0);
    const PixelType* bl = &it_img(bottom, left, 0);
    const PixelType* br = &it_img(bottom, right, 0);

    for (int channel = 0; channel < channels; ++channel) {
        output[channel] = static_cast<PixelType>(
            vert_alpha_comp *
                (horiz_alpha * tr[channel] + horiz_alpha_comp * tl[channel]) +
            vert_alpha *
                (horiz_alpha * br[channel] + horiz_alpha_comp * bl[channel]));
    }
}

struct BoundingBox
{
    BoundingBox()
    {
    }

    explicit BoundingBox(const BoundingBox& other)
        : left_top(other.left_top)
        , right_bottom(other.right_bottom)
    {
    }

    BoundingBox(BoundingBox&& other)
        : left_top(other.left_top)
        , right_bottom(other.right_bottom)
    {
    }

    BoundingBox(const std::initializer_list<std::array<float, 2>>& corners)
        : left_top(*corners.begin())
        , right_bottom(*(corners.end() - 1))
    {
    }

    BoundingBox(const Mat& image)
        : left_top({0, 0})
        , right_bottom({static_cast<float>(image.cols - 1),
                        static_cast<float>(image.rows - 1)})
    {
    }

    BoundingBox& operator=(const BoundingBox& other)
    {
        left_top     = other.left_top;
        right_bottom = other.right_bottom;

        return *this;
    }

    BoundingBox& operator=(BoundingBox&& other)
    {
        left_top     = other.left_top;
        right_bottom = other.right_bottom;

        return *this;
    }

    int flooring_width() const
    {
        return static_cast<int>(right_bottom[0] - left_top[0]) + 1;
    }

    int flooring_height() const
    {
        return static_cast<int>(right_bottom[1] - left_top[1]) + 1;
    }

    std::array<float, 2> left_top;
    std::array<float, 2> right_bottom;
};

BoundingBox bounding_box_transform(const BoundingBox& bb,
                                   const float* homography_ptr)
{
    using Eigen::Vector3f;
    using std::vector;

    Eigen::Map<const Matrix3fRowMajor> homography(homography_ptr);

    // clang-format off
    const vector<Vector3f> image_corner{
        {bb.left_top[0],     bb.left_top[1],     1.f},
        {bb.right_bottom[0], bb.left_top[1],     1.f},
        {bb.right_bottom[0], bb.right_bottom[1], 1.f},
        {bb.left_top[0],     bb.right_bottom[1], 1.f}
    };

    BoundingBox output_bb{
        {std::numeric_limits<float>::max(),
         std::numeric_limits<float>::max()},
        {std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::lowest()}
    };
    // clang-format on

    vector<Vector3f> transformed_corners(4);
    for (size_t i = 0; i < image_corner.size(); ++i) {
        transformed_corners[i] = homography * image_corner[i];
        transformed_corners[i] /= transformed_corners[i][2];

        // Update bounding box
        for (int c = 0; c < 2; ++c) {
            output_bb.left_top[c] =
                std::min(output_bb.left_top[c], transformed_corners[i][c]);
            output_bb.right_bottom[c] =
                std::max(output_bb.right_bottom[c], transformed_corners[i][c]);
        }
    }

    return output_bb;
}

BoundingBox bounding_box_intersect(const BoundingBox& bb_a,
                                   const BoundingBox& bb_b)
{
    using std::max;
    using std::min;

    return BoundingBox({{max(bb_a.left_top[0], bb_b.left_top[0]),
                         max(bb_a.left_top[1], bb_b.left_top[1])},
                        {min(bb_a.right_bottom[0], bb_b.right_bottom[0]),
                         min(bb_a.right_bottom[1], bb_b.right_bottom[1])}});
}

template <typename T>
Mat image_crop(const Mat& image, const BoundingBox& crop_bb)
{
    Mat output;

    assert(crop_bb.left_top[0] >= 0.f);
    assert(crop_bb.left_top[1] >= 0.f);

    assert(crop_bb.left_top[0] <= crop_bb.right_bottom[0]);
    assert(crop_bb.left_top[1] <= crop_bb.right_bottom[1]);

    assert(crop_bb.left_top[0] <= crop_bb.right_bottom[0]);
    assert(crop_bb.left_top[1] <= crop_bb.right_bottom[1]);

    output.create_from_buffer<T>(
        static_cast<T*>(image.data) +
            static_cast<int>(crop_bb.left_top[1]) * image.row_stride() +
            static_cast<int>(crop_bb.left_top[0]) * image.channels(),
        crop_bb.flooring_height(),
        crop_bb.flooring_width(),
        image.channels(),
        image.row_stride());
    output.data_mgr_ = image.data_mgr_;

    return output;
}

enum class InterpolationMode { Bilinear, NearestNeighbor };

template <typename PixelType,
          int channels,
          InterpolationMode interpolation_mode>
void image_transform(const Mat& image,
                     const float* homography_ptr,
                     const BoundingBox& output_bb,
                     Mat& output_image,
                     Mat& output_mask)
{
    using Eigen::Vector2f;
    using Eigen::Vector3f;
    using std::vector;

    assert(image.type() == Mat::Type::UINT8);
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

                if (interpolation_mode == InterpolationMode::Bilinear) {
                    bilinear_sample<PixelType, channels>(
                        it_img,
                        transformed_coord.data(),
                        &it_transf_img(y_buff, x_buff, 0));
                } else {
                    assert(interpolation_mode ==
                           InterpolationMode::NearestNeighbor);
                    for (int c = 0; c < channels; ++c) {
                        it_transf_img(y_buff, x_buff, c) =
                            it_img(static_cast<int>(transformed_coord[1]),
                                   static_cast<int>(transformed_coord[0]),
                                   c);
                    }
                }

                it_mask(y_buff, x_buff, 0) = 255;
            } else {
                it_mask(y_buff, x_buff, 0) = 0;
            }
        }
    }

    return;
}

void generate_mi_space(const Mat& source)
{
    Mat src_sat;
    generate_sat<1>(source, src_sat);
    Mat small;
    scale_from_sat<uint8_t, 1>(src_sat, 0.3f, small);

    const float dt = 10.0f;
    for (float y = -dt; y <= dt; y += 0.1f) {
        for (float x = -dt; x <= dt; x += 0.1f) {

            // clang-format off
            std::vector<float> homography {
                1.f, 0.f, x,
                0.f, 1.f, y,
                0.f, 0.f, 1.f
            };
            // clang-format on

            Mat transformed_mask;
            Mat transformed_img;

            BoundingBox input_bb = BoundingBox(small);

            BoundingBox output_bb = bounding_box_intersect(
                bounding_box_transform(input_bb, homography.data()), input_bb);
            Mat cropped_img = image_crop<uint8_t>(small, output_bb);
            image_transform<uint8_t, 1, InterpolationMode::Bilinear>(
                small,
                homography.data(),
                output_bb,
                transformed_img,
                transformed_mask);
            double mi = mutual_information(
                cropped_img, transformed_img, transformed_mask);

            std::cout << mi << " ";
        }
        std::cout << std::endl;
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
                                */

    generate_mi_space(source);
    return true;
}
