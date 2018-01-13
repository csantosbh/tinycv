#ifndef _TINYCV_DRAW_LINE_H_
#define _TINYCV_DRAW_LINE_H_

#include <array>

#include "tinycv/mat.h"
#include "tinycv/math.h"

template <typename PixelType, int NumChannels>
void draw_line(const Point<int>& a,
               const Point<int>& b,
               const std::array<PixelType, NumChannels>& color,
               Mat& canvas);

#endif
