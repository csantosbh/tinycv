#include "registration.h"
#include "mat.h"
#include "sat.h"

bool register_translation(const Mat& source,
                          const Mat& destination,
                          Eigen::Vector2f& registration)
{

    Mat src_sat;

    generate_sat<3>(source, src_sat);

    float scale = 0.99f;

    for(int i = 0; i < 20; ++i) {
        Mat small;
        scale_from_sat<uint8_t, 3>(src_sat, scale, small);
        scale *= 0.97f;
    }

    return true;
}
