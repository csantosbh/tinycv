#ifndef _LIBMIR_REGISTRATION_H_
#define _LIBMIR_REGISTRATION_H_

#include <Eigen/Eigen>

#include "mat.h"

bool register_translation(const Mat& source,
                          const Mat& destination,
                          Eigen::Vector2f& registration);

#endif
