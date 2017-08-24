#ifndef _LIBMIR_MATH_H_
#define _LIBMIR_MATH_H_

#include <algorithm>

template<typename T>
T clamp(T value,
        T lowest,
        T highest)
{
    return std::max(lowest,
                    std::min(highest,
                             value));
}

template<typename T>
T discretize_pixel(float pixel) {
    return static_cast<T>(clamp(pixel,
                                0.f,
                                255.f));
}

#endif
