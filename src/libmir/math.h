#ifndef _LIBMIR_MATH_H_
#define _LIBMIR_MATH_H_

#include <algorithm>

template <typename PointType>
struct Point
{
    PointType x;
    PointType y;

    const PointType* ptr() const
    {
        return reinterpret_cast<const PointType*>(this);
    }
};

template <typename T>
T clamp(T value, T lowest, T highest)
{
    return std::max(lowest, std::min(highest, value));
}

template <typename T>
T discretize_pixel(float pixel)
{
    return static_cast<T>(clamp(pixel, 0.f, 255.f));
}

template <typename InputType, typename OutputType>
OutputType fast_positive_round(InputType input)
{
    return static_cast<OutputType>(input + 0.5);
}

#endif
