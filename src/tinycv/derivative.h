#ifndef _TINYCV_DERIVATIVE_H_
#define _TINYCV_DERIVATIVE_H_

#include <cassert>

#include "math.h"
#include "transform.h"

enum class ImageDerivativeAxis { dX, dY };
enum class FilterOrder { Fifth, Seventh };

template <int channels, FilterOrder filter_order>
class DerivativeHoloborodko
{
  public:
    /**
     * Size of the border of the derivative image that will be discarded when
     * calling derivative()
     */
    static constexpr int border_crop_size()
    {
        return filter_order == FilterOrder::Fifth ? 2 : 3;
    }

    /**
     * Compute the image derivative along the specified axis.
     *
     * Note that the output image borders WILL BE CROPPED by an amount
     * proportional to the chosen filter order. Call border_crop_size() to query
     * the length of the border that will be cropped.
     */
    template <typename InputPixelType, typename OutputPixelType>
    static void
    derivative(const Mat& image, ImageDerivativeAxis axis, Mat& output_image)
    {
        if (filter_order == FilterOrder::Fifth) {
            derivative_impl<InputPixelType, OutputPixelType>(image,
                                                             axis,
                                                             {-1, -2, 0, 2, 1},
                                                             {1, 1, 2, 1, 1},
                                                             1.f / 32.f,
                                                             output_image);
        } else {
            assert(filter_order == FilterOrder::Seventh);

            derivative_impl<InputPixelType, OutputPixelType>(
                image,
                axis,
                {-1, -4, -5, 0, 5, 4, 1},
                {1, 1, 4, 6, 4, 1, 1},
                1.f / 512.f,
                output_image);
        }
    }

  private:
    template <typename InputPixelType, typename OutputPixelType>
    static void
    derivative_impl(const Mat& image,
                    ImageDerivativeAxis axis,
                    const std::initializer_list<float>& high_pass_component,
                    const std::initializer_list<float>& low_pass_component,
                    const float norm_factor,
                    Mat& output_image)
    {
        assert(std::is_signed<OutputPixelType>::value);
        assert(high_pass_component.size() == low_pass_component.size());

        Mat horizontal_kernel;
        Mat vertical_kernel;

        const int kernel_length = static_cast<int>(high_pass_component.size());

        horizontal_kernel.create<float>(1, kernel_length, 1);
        vertical_kernel.create<float>(1, kernel_length, 1);

        if (axis == ImageDerivativeAxis::dX) {
            horizontal_kernel << high_pass_component;
            vertical_kernel << low_pass_component;
        } else {
            assert(axis == ImageDerivativeAxis::dY);

            horizontal_kernel << low_pass_component;
            vertical_kernel << high_pass_component;
        }

        image_convolve<InputPixelType, OutputPixelType, channels>(
            image,
            vertical_kernel,
            horizontal_kernel,
            norm_factor,
            output_image);
    }
};

template <int channels>
class DerivativeNaive
{
  public:
    static constexpr int border_crop_size()
    {
        ///
        // In the case of the Naive derivative, we should in principle only need
        // to remove the horizontal or vertical borders. However, since we
        // always compute both horizontal and vertical derivatives, both borders
        // will end up being cropped anyway
        return 1;
    }

    template <typename InputPixelType, typename OutputPixelType>
    static void
    derivative(const Mat& image, ImageDerivativeAxis axis, Mat& output_image)
    {
        const int kernel_border = border_crop_size();
        assert(std::is_signed<OutputPixelType>::value);

        output_image.create<OutputPixelType>(image.rows - 2 * kernel_border,
                                             image.cols - 2 * kernel_border,
                                             image.channels());

        Mat::ConstIterator<InputPixelType> input_img_it(image);
        std::function<void(const Point<int>&, Point<int>&, Point<int>&)>
            coord_deriv_provider;


        if (axis == ImageDerivativeAxis::dX) {
            coord_deriv_provider = [](const Point<int>& coord_current,
                                      Point<int>& coord_prev,
                                      Point<int>& coord_next) {
                coord_prev = {coord_current.x - 1, coord_current.y};
                coord_next = {coord_current.x + 1, coord_current.y};
            };
        } else {
            coord_deriv_provider = [](const Point<int>& coord_current,
                                      Point<int>& coord_prev,
                                      Point<int>& coord_next) {
                coord_prev = {coord_current.x, coord_current.y - 1};
                coord_next = {coord_current.x, coord_current.y + 1};
            };
        }

        const auto derivative_core =
            [&input_img_it, &coord_deriv_provider, kernel_border](
                Mat::Iterator<OutputPixelType>& it, int y, int x, int c) {
                Point<int> coord_prev;
                Point<int> coord_next;
                coord_deriv_provider({kernel_border + x, kernel_border + y},
                                     coord_prev,
                                     coord_next);

                it(y, x, c) = static_cast<OutputPixelType>(
                                  input_img_it(coord_next.y, coord_next.x, c) -
                                  input_img_it(coord_prev.y, coord_prev.x, c)) /
                              static_cast<OutputPixelType>(2);
            };

        output_image.for_each<Mat::Iterator<OutputPixelType>>(derivative_core);
    }
};

#endif
