#include <cassert>
#include <cmath>
#include <limits>

#include "sat.h"
#include "math.h"

template<int channels>
void generate_sat(const Mat& image,
                  Mat& sat)
{
    assert(image.type() == Mat::Type::UINT8);
    assert(channels == image.step.buf[1]);

    sat.create<SatType>(image.cols + 1,
                        image.rows + 1,
                        channels);

    // Horizontal step
    uint8_t* imagePtr = static_cast<uint8_t*>(image.data);
    SatType* satPtr = static_cast<SatType*>(sat.data);

    // First row
    for(int x = 0; x < sat.cols; ++x) {
        for(int c = 0; c < channels; ++c) {
            *satPtr++ = 0;
        }
    }

    for(int y = 1; y < sat.rows; ++y) {
        // First column
        for(int c = 0; c < channels; ++c) {
            *satPtr++ = 0;
        }

        // Second column
        for(int c = 0; c < channels; ++c) {
            *satPtr++ = *imagePtr++;
        }

        // Remaining columns
        for(int x = 2; x < sat.cols; ++x) {
            for(int c = 0; c < channels; ++c) {
                *satPtr = *(satPtr - channels) + *imagePtr++;
                satPtr++;
            }
        }
    }

    // Vertical step
    satPtr = static_cast<SatType*>(sat.data) +
             2 * sat.step.buf[0];

    for(int y = 2; y < sat.rows; ++y) {
        // Skip first column
        satPtr += sat.step.buf[1];

        // Iterate over next columns
        for(int x = 1; x < sat.cols; ++x) {
            for(int c = 0; c < channels; ++c) {
                *satPtr = *(satPtr - sat.step.buf[0]) + *satPtr;
                satPtr++;
            }
        }
    }
}

template<typename T,
         int channels>
void scale_from_sat(const Mat& source,
                    float scale,
                    Mat& destination)
{
    assert(scale < 1.0f);
    assert(channels == source.step.buf[1]);
    assert(source.type() == SatTypeEnum);

    int dstCols = static_cast<int>(
                      round(scale *
                            static_cast<float>(source.cols)));
    int dstRows = static_cast<int>(
                      round(scale *
                            static_cast<float>(source.rows)));

    destination.create<T>(dstCols, dstRows, channels);

    /*
     *
     *                     |- k_x -|
     *      +--------------.-------.---------+
     *      |              |       .   |     |
     *      |              |       .   |     |
     *      |              |       .   |     |
     *      |              |       .   |     |
     *      |            Pa|       .   |Pb   |
     *  --- |--------------*       .   *     |
     *  |   |                      .   |     |
     * k_y  |                      .   |     |
     *  |   |                      .   |     |
     *  --- .......................*Pm |     |
     *      |                          |     |
     *      |--------------*-----------*Pc   |
     *      |            Pd                  |
     *      +--------------------------------+
     *
     * SAT of floating point position Pm:
     *    Pm = Pa +
     *         (Pb-Pa) * k_x +
     *         (Pd-Pa) * k_y +
     *         (Pc-Pb-Pd+Pa) * k_x * k_y
     *
     */

    const double step_x = (static_cast<double>(source.cols) - 1) /
                         static_cast<double>(destination.cols);
    const double step_y = (static_cast<double>(source.rows) - 1) /
                         static_cast<double>(destination.rows);
    const float pixel_area_conj = 1.f /
                                  static_cast<float>(step_x * step_y);


    Mat::ConstIterator<SatType> it_sat(source);
    Mat::Iterator<T> it_dst(destination);

    auto sample_sat = [](const Mat::ConstIterator<SatType>& src,
                         double x,
                         double y,
                         std::array<float, channels>& dst)
    {
        const float k_x = static_cast<float>(x - std::floor(x));
        const float k_y = static_cast<float>(y - std::floor(y));

        const int a_x = static_cast<int>(std::floor(x));
        const int a_y = static_cast<int>(std::floor(y));
        const int c_x = static_cast<int>(std::ceil(x));
        const int c_y = static_cast<int>(std::ceil(y));

        for(int c = 0; c < channels; ++c) {
            SatType Pa = src(a_y, a_x, c);
            SatType Pb = src(a_y, c_x, c);
            SatType Pc = src(c_y, c_x, c);
            SatType Pd = src(c_y, a_x, c);

            assert(Pb >= Pa);
            assert(Pd >= Pa);
            assert(Pc >= Pb);
            assert(Pc >= Pd);
            assert(Pc + Pa >= Pb + Pd);

            dst[c] = static_cast<float>(Pa) +
                     static_cast<float>(Pb - Pa) * k_x +
                     static_cast<float>(Pd - Pa) * k_y +
                     static_cast<float>(Pc - Pb - Pd + Pa) * k_x * k_y;
        }
    };

    std::array<float, channels> Sa;
    std::array<float, channels> Sb;
    std::array<float, channels> Sc;
    std::array<float, channels> Sd;

    for(int dst_y = 0; dst_y < destination.rows; ++dst_y) {
        const double src_lower_y = step_y *
                                   static_cast<double>(dst_y);
        const double src_upper_y = clamp(src_lower_y + step_y,
                                         0.0, source.rows - 1.0);

        for(int dst_x = 0; dst_x < destination.cols; ++dst_x) {
            const double src_lower_x = step_x *
                                       static_cast<double>(dst_x);
            const double src_upper_x = clamp(src_lower_x + step_x,
                                       0.0, source.cols - 1.0);

            sample_sat(it_sat,
                       src_lower_x,
                       src_lower_y,
                       Sa);

            sample_sat(it_sat,
                       src_upper_x,
                       src_lower_y,
                       Sb);

            sample_sat(it_sat,
                       src_upper_x,
                       src_upper_y,
                       Sc);

            sample_sat(it_sat,
                       src_lower_x,
                       src_upper_y,
                       Sd);

            for(int c = 0; c < channels; ++c) {
                assert(Sb[c] >= Sa[c]);
                assert(Sd[c] >= Sa[c]);
                assert(Sc[c] >= Sb[c]);
                assert(Sc[c] >= Sd[c]);

                float area = (Sc[c] - Sb[c] - Sd[c] + Sa[c]) *
                             pixel_area_conj;

                it_dst(dst_y,
                       dst_x,
                       c) = discretize_pixel<T>(area);
            }
        }
    }
}

template
void generate_sat<1>(const Mat& image,
                     Mat& sat);

template
void generate_sat<3>(const Mat& image,
                     Mat& sat);

template
void scale_from_sat<uint8_t, 1>(const Mat& source,
                                float scale,
                                Mat& destination);

template
void scale_from_sat<uint8_t, 3>(const Mat& source,
                                float scale,
                                Mat& destination);
