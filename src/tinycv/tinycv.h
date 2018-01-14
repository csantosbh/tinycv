#ifndef _TINYCV_TINYCV_H_
#define _TINYCV_TINYCV_H_

#include "registration.h"
#include "transform.h"

#include "draw/line.h"

#define TINYCV_IMPLEMENTATION
#define TINYCV_HEADER_ONLY

#if defined(TINYCV_IMPLEMENTATION) && defined(TINYCV_HEADER_ONLY)
#include "bounding_box.cpp"
#include "histogram.cpp"
#include "mat.cpp"
#include "registration.cpp"
#include "sat.cpp"

#include "draw/line.cpp"
#endif

#endif
