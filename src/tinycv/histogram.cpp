#include "histogram.h"

float KroneckerFunction::histogram_bin_function(const float i)
{
    return (i >= -0.5f && i <= 0.5f) ? 1 : 0;
}

float BSpline4::histogram_bin_function(const float i)
{
    float result;

    if (i < -2.f) {
        // i < -2.0
        result = 0.f;
    } else if (i < -1.f) {
        // -2.0 <= i < -1.0
        float k = (2.f + i);

        result = k * k * k / 6.f;
    } else if (i < 0.f) {
        // -1.0 <= i < 0.0
        float k = 1.f + i;

        result = (1.f + 3.f * k + 3 * k * k - 3.f * k * k * k) / 6.f;
    } else if (i < 1.f) {
        // 0.0 <= i < 1.0
        float k = 1.f - i;

        result = (1.f + 3.f * k + 3 * k * k - 3.f * k * k * k) / 6.f;
    } else if (i < 2.f) {
        // 1.0 <= i < 2.0
        float k = (2.f - i);

        result = k * k * k / 6.f;
    } else {
        // 2.0 <= i
        assert(2.f <= i);

        result = 0.f;
    }

    return result;
}

float BSpline4::hbf_derivative(const float i)
{
    const auto b_spline_3 = [](const float i) -> float {
        float result;

        if (i < -1.5) {
            // i < -1.5
            result = 0.f;
        } else if (i < -0.5) {
            // -1.5 <= i < -0.5
            float k = (1.5f + i);

            result = k * k / 2.f;
        } else if (i < 0.f) {
            // -0.5 <= i < 0.0
            float k = 0.5f + i;

            result = 1 + i - k * k;
        } else if (i < 0.5f) {
            // 0.0 <= i < 0.5
            float k = 0.5f - i;

            result = 1 - i - k * k;
        } else if (i < 1.5) {
            // 0.5 <= i < 1.5
            float k = 1.5f - i;

            result = k * k / 2.f;
        } else {
            // 1.5 <= i
            assert(1.5 <= i);

            result = 0.f;
        }

        return result;
    };

    return b_spline_3(i + 0.5f) - b_spline_3(i - 0.5f);
}

float BSpline4::hbf_second_derivative(const float i)
{
    const auto b_spline_2 = [](const float i) {
        float result;

        if (i < -1.f) {
            // i < -1
            result = 0.f;
        } else if (i < 0.f) {
            // -1 <= i < 0
            result = 1.f + i;
        } else if (i < 1.f) {
            // 0 <= i < 1
            result = 1.f - i;
        } else {
            // 1 <= i
            result = 0.f;
        }

        return result;
    };

    return b_spline_2(i + 1.0f) - 2.f * b_spline_2(i) + b_spline_2(i - 1.0f);
}

double sum_histogram(const Mat::ConstIterator<float>& histogram)
{
    assert(!histogram.m.empty());

    double histogram_summation = 0.0;

    for (int y = 0; y < histogram.m.rows; ++y) {
        for (int x = 0; x < histogram.m.cols; ++x) {
            for (int c = 0; c < histogram.m.channels(); ++c) {
                histogram_summation += histogram(y, x, c);
            }
        }
    }

    return histogram_summation;
}

void normalize_histogram(const double histogram_summation,
                         Mat::Iterator<float>& histogram)
{
    assert(!histogram.m.empty());

    const float histogram_normalization =
        static_cast<float>(1.0 / histogram_summation);

    for (int y = 0; y < histogram.m.rows; ++y) {
        for (int x = 0; x < histogram.m.cols; ++x) {
            for (int c = 0; c < histogram.m.channels(); ++c) {
                histogram(y, x, c) *= histogram_normalization;
            }
        }
    }
}
