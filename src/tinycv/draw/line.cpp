#include <functional>

#include "line.h"


namespace tinycv
{

template <typename PixelType>
void iterate_line(const Point<int>& a,
                  const Point<int>& b,
                  const std::function<void(const Point<int>&)>& callback)
{
    const int horizontal_steps = abs(b.x - a.x) + 1;
    const int vertical_steps   = abs(b.y - a.y) + 1;

    const int dx = sign(b.x - a.x);
    const int dy = sign(b.y - a.y);

    Point<int> point_it{a.x, a.y};
    float residual = 0.f;

    if (horizontal_steps > vertical_steps) {
        const float horiz_steps_per_vert_step =
            static_cast<float>(horizontal_steps) /
            static_cast<float>(vertical_steps);

        for (int v_step = 0; v_step < vertical_steps; ++v_step) {
            const float last_h_step_f = horiz_steps_per_vert_step + residual;
            const int last_h_step     = static_cast<int>(last_h_step_f);
            residual                  = last_h_step_f - floor(last_h_step_f);

            for (int h_step = 0; h_step < last_h_step; ++h_step) {
                callback(point_it);
                point_it.x += dx;
            }
            point_it.y += dy;
        }
    } else {
        const float vert_steps_per_horiz_step =
            static_cast<float>(vertical_steps) /
            static_cast<float>(horizontal_steps);

        for (int h_step = 0; h_step < horizontal_steps; ++h_step) {
            const float last_v_step_f = vert_steps_per_horiz_step + residual;
            const int last_v_step     = static_cast<int>(last_v_step_f);
            residual                  = last_v_step_f - floor(last_v_step_f);

            for (int v_step = 0; v_step < last_v_step; ++v_step) {
                callback(point_it);
                point_it.y += dy;
            }
            point_it.x += dx;
        }
    }

    callback(b);

    assert(abs(point_it.x - b.x) <= 1 && abs(point_it.y - b.y) <= 1);

    return;
}

template <typename PixelType, int NumChannels>
void draw_line(const Point<int>& a,
               const Point<int>& b,
               const std::array<PixelType, NumChannels>& color,
               Mat& canvas)
{
    assert(canvas.data != nullptr);

    Mat::Iterator<PixelType> canvas_it(canvas);
    iterate_line<PixelType>(
        a, b, [&canvas_it, &color](const Point<int>& point) {
            if (point.x >= 0 && point.x < canvas_it.m.cols && point.y >= 0 &&
                point.y < canvas_it.m.rows) {
                for (int channel = 0; channel < NumChannels; ++channel) {
                    canvas_it(point.y, point.x, channel) = color[channel];
                }
            }
        });
}
}
