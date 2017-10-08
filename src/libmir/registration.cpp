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
    assert(image_a.channels() == 1);
    assert(image_b.channels() == 1);
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

double mutual_information(const Mat& image_a, const Mat& image_b)
{
    using std::vector;

    vector<float> histogram_a;
    image_histogram<uint8_t, BSpline4>(image_a, histogram_a);

    vector<float> histogram_b;
    image_histogram<uint8_t, BSpline4>(image_b, histogram_b);

    vector<float> histogram_ab;
    joint_image_histogram<uint8_t, BSpline4>(image_a, image_b, histogram_ab);

    const int number_practical_bins = static_cast<int>(histogram_a.size());

    const auto joint_hist_pos = [number_practical_bins](int x, int y) {
        return y * number_practical_bins + x;
    };

    double mi_summation = 0.0;
    for (int j = 0; j < number_practical_bins; ++j) {
        for (int i = 0; i < number_practical_bins; ++i) {
            double prob_ij = histogram_ab[joint_hist_pos(i, j)];

            double prob_ai = histogram_a[i];
            double prob_bj = histogram_b[j];

            mi_summation += prob_ij * std::log(prob_ij / (prob_ai * prob_bj));
        }
    }

    return mi_summation;
}

template <typename PixelType>
PixelType bilinear_sample(const Mat::ConstIterator<PixelType>& it_img,
                          float* coordinates,
                          int channel)
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
    int bottom = std::min(coordinates_i[1], it_img.m.rows - 1);

    assert(0 <= left);
    assert(left <= right);
    assert(right < it_img.m.cols);

    assert(0 <= top);
    assert(top <= bottom);
    assert(bottom < it_img.m.rows);

    PixelType tl = it_img(top, left, channel);
    PixelType tr = it_img(top, right, channel);
    PixelType bl = it_img(bottom, left, channel);
    PixelType br = it_img(bottom, right, channel);

    return static_cast<PixelType>(
        vert_alpha_comp * (horiz_alpha * tr + horiz_alpha_comp * tl) +
        vert_alpha * (horiz_alpha * br + horiz_alpha_comp * bl));
}

struct BoundingBox
{
    BoundingBox()
    {
    }

    BoundingBox(const BoundingBox& other)
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

    float width() const
    {
        return right_bottom[0] - left_top[0];
    }

    float height() const
    {
        return right_bottom[1] - left_top[1];
    }

    std::array<float, 2> left_top;
    std::array<float, 2> right_bottom;
};

void bounding_box_transform(const float* homography_ptr, BoundingBox& bb)
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

    bb.left_top     = {std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max()};
    bb.right_bottom = {std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest()};
    // clang-format on

    vector<Vector3f> transformed_corners(4);
    for (size_t i = 0; i < image_corner.size(); ++i) {
        transformed_corners[i] = homography * image_corner[i];
        transformed_corners[i] /= transformed_corners[i][2];

        // Update bounding box
        for (int c = 0; c < 2; ++c) {
            bb.left_top[c] =
                std::min(bb.left_top[c], transformed_corners[i][c]);
            bb.right_bottom[c] =
                std::max(bb.right_bottom[c], transformed_corners[i][c]);
        }
    }
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

    output.create_from_buffer<T>(static_cast<T*>(image.data) +
                                     static_cast<int>(crop_bb.left_top[0]) *
                                         image.channels(),
                                 static_cast<int>(crop_bb.height()),
                                 static_cast<int>(crop_bb.width()),
                                 image.channels(),
                                 image.row_stride());

    return output;
}

void image_transform(const Mat& image,
                     const float* homography_ptr,
                     const BoundingBox& output_bb,
                     Mat& output_image)
{
    using Eigen::Vector2f;
    using Eigen::Vector3f;
    using std::vector;
    using PixelType = uint8_t;

    assert(image.type() == Mat::Type::UINT8);

    // Compute bounding box of the transformed image by transforming its corners
    Eigen::Map<const Matrix3fRowMajor> homography(homography_ptr);

    int output_width = static_cast<int>(
        std::floor(output_bb.right_bottom[0] - output_bb.left_top[0]));
    int output_height = static_cast<int>(
        std::floor(output_bb.right_bottom[1] - output_bb.left_top[1]));

    output_image.create<PixelType>(output_width, output_height, 1);
    // TODO: output_image.fill(0);
    memset(output_image.data, 0, output_width * output_height);

    Matrix3fRowMajor homography_inv = homography.inverse();
    // Converts from output space to transformed bounding box space
    Matrix3fRowMajor transf_bb_pivot;
    // clang-format off
    transf_bb_pivot << 1.f, 0.f, output_bb.left_top[0],
                       0.f, 1.f, output_bb.left_top[1],
                       0.f, 0.f, 1.f;
    // clang-format on
    homography_inv = homography_inv * transf_bb_pivot;

    Mat::ConstIterator<PixelType> it_img(image);
    Mat::Iterator<PixelType> it_transf_img(output_image);

    float last_input_col = static_cast<float>(image.cols) - 1.f;
    float last_input_row = static_cast<float>(image.rows) - 1.f;

    for (int y_buff = 0; y_buff < output_image.rows; ++y_buff) {
        for (int x_buff = 0; x_buff < output_image.cols; ++x_buff) {
            for (int c = 0; c < static_cast<int>(output_image.channels());
                 ++c) {
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
                    it_transf_img(y_buff, x_buff, c) =
                        bilinear_sample(it_img, transformed_coord.data(), c);
                    // TODO set mask to ~0
                }
            }
        }
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

    // clang-format off
    std::vector<float> homography {
        1.f, 0.f, 10.f,
        0.f, 1.f, -10.f,
        -0.0001f, 0.0001f, 1.f
    };
    // clang-format on
    Mat transformed_source;
    BoundingBox interest_bb(source);
    bounding_box_transform(homography.data(), interest_bb);
    interest_bb = bounding_box_intersect(interest_bb, BoundingBox(source));
    Mat cropped_source = image_crop<uint8_t>(source, interest_bb);
    image_transform(source, homography.data(), interest_bb, transformed_source);

    return true;
}
