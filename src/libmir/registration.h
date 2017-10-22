#ifndef _LIBMIR_REGISTRATION_H_
#define _LIBMIR_REGISTRATION_H_

#include "mat.h"

bool register_translation(const Mat& source,
                          const Mat& destination,
                          float* translation);

#endif
