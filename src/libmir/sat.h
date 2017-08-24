#ifndef _LIBMIR_SAT_H_
#define _LIBMIR_SAT_H_

#include "mat.h"

template<int channels>
void generate_sat(const Mat& image,
                  Mat& sat);

template<typename T,
         int channels>
void scale_from_sat(const Mat& source,
                    float scale,
                    Mat& destination);

#endif
