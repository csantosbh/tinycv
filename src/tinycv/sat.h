#ifndef _TINYCV_SAT_H_
#define _TINYCV_SAT_H_

#include "mat.h"


namespace tinycv
{

template <int channels>
void generate_sat(const Mat& image, Mat& sat);

template <typename T, int channels>
void scale_from_sat(const Mat& source, float scale, Mat& destination);
}

#endif
