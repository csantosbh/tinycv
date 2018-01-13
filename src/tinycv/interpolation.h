#ifndef _TINYCV_INTERPOLATION_H_
#define _TINYCV_INTERPOLATION_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "mat.h"
#include "math.h"

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
    assert(it_img.m.type() == Mat::get_type_enum<PixelType>());

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

template <typename PixelType, int channels>
void bilateral_sample(const Mat::ConstIterator<PixelType>& it_img,
                      const float* coordinates,
                      PixelType* output)
{
    assert(it_img.m.type() == Mat::get_type_enum<PixelType>());

    float horiz_alpha      = coordinates[0] - std::floor(coordinates[0]);
    float vert_alpha       = coordinates[1] - std::floor(coordinates[1]);
    float horiz_alpha_comp = 1.f - horiz_alpha;
    float vert_alpha_comp  = 1.f - vert_alpha;

    // clang-format off
    const std::array<std::array<float, 2>, 4> window_neighbor_coords {{
        {-horiz_alpha, -vert_alpha},     {horiz_alpha_comp, -vert_alpha},
        {-horiz_alpha, vert_alpha_comp}, {horiz_alpha_comp, vert_alpha_comp}
    }};
    // clang-format on

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

    const auto norm_l2_squared = [](int dimensions, const auto& coords) {
        float result = 0.0;

        for (int i = 0; i < dimensions; ++i) {
            result += coords[i] * coords[i];
        }

        return result;
    };

    const auto norm_l1 = [](int dimensions, const auto& coords) {
        float result = 0.0;

        for (int i = 0; i < dimensions; ++i) {
            result += std::abs(coords[i]);
        }

        return result;
    };

    // cpp-format off
    const std::array<const PixelType*, 4> neighbors{
        {&it_img(top, left, 0),
         &it_img(top, right, 0),
         &it_img(bottom, left, 0),
         &it_img(bottom, right, 0)}};
    // cpp-format on

    const float sigma_d = 0.500f;
    const float sigma_r = 8.0f;

    const float sigma_d_squared = sigma_d * sigma_d;
    const float sigma_r_squared = sigma_r * sigma_r;

    std::array<float, 4> weights;
    std::array<float, channels> median;

    // Compute median pixel
    for (int c = 0; c < channels; ++c) {
        std::vector<size_t> neighbor_indices = {0, 1, 2, 3};

        std::sort(neighbor_indices.begin(),
                  neighbor_indices.end(),
                  [&neighbors, c](size_t idx_a, size_t idx_b) {
                      return neighbors[idx_a][c] < neighbors[idx_b][c];
                  });

        median[c] = neighbors[neighbor_indices[1]][c];
    }

    float weight_summation = 0.f;

    // Compute intensity differences and bilateral weights
    for (int i = 0; i < 4; ++i) {
        std::array<float, channels> intensity_diffs;
        for (int c = 0; c < channels; ++c) {
            intensity_diffs[c] =
                static_cast<float>(neighbors[i][c]) - median[c];
        }

        weights[i] = std::exp(-norm_l2_squared(2, window_neighbor_coords[i]) /
                                  (2.0f * sigma_d_squared) -
                              norm_l1(channels, intensity_diffs) /
                                  (2.0f * sigma_r_squared));

        weight_summation += weights[i];
    }

    const float weight_summation_comp = 1.f / weight_summation;

    // Compute pixel value
    for (int c = 0; c < channels; ++c) {
        float pixel_value = 0.f;

        for (int i = 0; i < 4; ++i) {
            pixel_value += neighbors[i][c] * weights[i];
        }

        output[c] = static_cast<PixelType>(pixel_value * weight_summation_comp);
    }
}

template <typename PixelType, int channels>
void jitter_sample(const Mat::ConstIterator<PixelType>& it_img,
                   const float* coordinates,
                   PixelType* output)
{
    float horiz_alpha = coordinates[0] - std::floor(coordinates[0]);
    float vert_alpha  = coordinates[1] - std::floor(coordinates[1]);

    // TODO create own rand functions and distribution types
    float rand_x =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    float rand_y =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

    int offset_x = rand_x < horiz_alpha ? 1 : 0;
    int offset_y = rand_y < vert_alpha ? 1 : 0;

    int x = clamp(
        static_cast<int>(coordinates[0]) + offset_x, 0, it_img.m.cols - 1);
    int y = clamp(
        static_cast<int>(coordinates[1]) + offset_y, 0, it_img.m.rows - 1);

    assert(0 <= x);
    assert(x < it_img.m.cols);

    assert(0 <= y);
    assert(y < it_img.m.rows);

    for (int channel = 0; channel < channels; ++channel) {
        output[channel] = it_img(y, x, channel);
    }
}

template <typename PixelType, int channels>
void nearest_neighbor_sample(const Mat::ConstIterator<PixelType>& it_img,
                             const float* coordinates,
                             PixelType* output)
{
    int x = clamp(
        static_cast<int>(std::round(coordinates[0])), 0, it_img.m.cols - 1);
    int y = clamp(
        static_cast<int>(std::round(coordinates[1])), 0, it_img.m.rows - 1);

    assert(0 <= x);
    assert(x < it_img.m.cols);

    assert(0 <= y);
    assert(y < it_img.m.rows);

    for (int channel = 0; channel < channels; ++channel) {
        output[channel] = it_img(y, x, channel);
    }
}

#endif
