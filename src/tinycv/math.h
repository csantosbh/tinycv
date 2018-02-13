#ifndef _TINYCV_MATH_H_
#define _TINYCV_MATH_H_

#include <algorithm>


namespace tinycv
{

template <typename PrimitiveType>
struct Point
{
    PrimitiveType x;
    PrimitiveType y;

    const PrimitiveType* cptr() const
    {
        return reinterpret_cast<const PrimitiveType*>(this);
    }

    PrimitiveType* ptr()
    {
        return reinterpret_cast<PrimitiveType*>(this);
    }
};

template <typename PrimitiveType>
Point<PrimitiveType> operator-(const Point<PrimitiveType>& point)
{
    return {-point.x, -point.y};
}

template <typename PrimitiveType>
Point<PrimitiveType> operator-(const Point<PrimitiveType>& a,
                               const Point<PrimitiveType>& b)
{
    return {a.x - b.x, a.y - b.y};
}

template <typename PrimitiveType>
Point<PrimitiveType> operator+(const Point<PrimitiveType>& a,
                               const Point<PrimitiveType>& b)
{
    return {a.x + b.x, a.y + b.y};
}

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

template <typename T>
T sign(T value)
{
    return value < 0 ? -1 : value == 0 ? 0 : 1;
}
}

#endif
