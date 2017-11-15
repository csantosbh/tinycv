#ifndef _LIBMIR_MUTUAL_INFORMATION_H_
#define _LIBMIR_MUTUAL_INFORMATION_H_

#include <vector>

#include "histogram.h"
#include "mat.h"

using MaskType = uint8_t;

/**
 * Mask iterator for the case where we know that all pixels are valid
 *
 * Designed for use with templated parameters, so the compiler can optimize away
 * conditional branches.
 */
struct PositiveMaskIterator
{
    uint8_t operator()(int, int, int) const
    {
        return 255;
    }

    bool is_mask_of(const Mat&) const
    {
        return true;
    }
};

template <typename PixelType, typename MaskIteratorA, typename MaskIteratorB>
double mutual_information_impl(const Mat& image_a,
                               const MaskIteratorA& it_mask_a,
                               const Mat& image_b,
                               const MaskIteratorB& it_mask_b)
{
    using std::vector;
    using BinningMethod = BSpline4;

    assert(image_a.rows == image_b.rows);
    assert(image_a.cols == image_b.cols);

    assert(image_a.type() == image_b.type());

    assert(it_mask_a.is_mask_of(image_a));
    assert(it_mask_b.is_mask_of(image_b));


    Mat histogram_a;
    Mat histogram_b;
    Mat histogram_ab;
    joint_image_histogram<PixelType,
                          BinningMethod,
                          MaskIteratorA,
                          MaskIteratorB>(image_a,
                                         it_mask_a,
                                         image_b,
                                         it_mask_b,
                                         histogram_a,
                                         histogram_b,
                                         histogram_ab);

    const int number_practical_bins = static_cast<int>(histogram_a.cols);

    // Histogram iterators
    Mat::ConstIterator<float> hist_a_it(histogram_a);
    Mat::ConstIterator<float> hist_b_it(histogram_b);
    Mat::ConstIterator<float> hist_ab_it(histogram_ab);

    double mi_summation = 0.0;
    for (int j = 0; j < number_practical_bins; ++j) {
        for (int i = 0; i < number_practical_bins; ++i) {
            double prob_ij = hist_ab_it(i, j, 0);

            /*
             * We know that P(a=i,b=j) < P(a=i) and P(a=i,b=j) < P(b=j), since
             * (a=i,b=j) is a subset of both (a=i) and (b=j) events.
             *
             * Therefore, if P(a=i)=0 or P(b=j)=0, then P(a=i,b=j)=0.
             *
             * Now consider P(a=i,b=j)=0. Then, the MI term
             *
             *  MI(i,j) = P(a=i,b=j) * log(P(a=i,b=j) / (P(a=i) * P(b=j)))
             *
             * must evaluate to 0.
             *
             * Proof:
             * Let k = P(a=i,b=j), l=P(a=i) and m=P(b=j), for the sake of
             * simplicity. Then:
             *
             *  MI(i,j) = lim{k->0+} k * log(k/(l * m)).
             *
             * If l > 0 and m > 0, then it is trivial to see that MI(i,j) = 0.
             * If, however, both l and m are zero, we have
             *
             *  MI(i,j) = lim{k->0+} k * log(k/(k * k))
             *          = lim{k->0+} k * log(k) - k * log(k) - k * log(k)
             *
             * Each term k * log(k) can be written as log(k) / (1/k), so one can
             * use L'hopital rule to find out that each term of the sum
             * converges to 0.
             */
            if (prob_ij > 0.0) {
                double prob_ai = hist_a_it(0, i, 0);
                double prob_bj = hist_b_it(0, j, 0);

                double logterm = std::log(prob_ij / (prob_ai * prob_bj));
                mi_summation += prob_ij * logterm;
            }

            assert(!std::isnan(mi_summation));
        }
    }

    return mi_summation;
}

/**
 * Compute the mutual information between image_a and image_b
 */
template <typename PixelType>
double mutual_information(const Mat& image_a, const Mat& image_b)
{
    return mutual_information_impl<PixelType,
                                   PositiveMaskIterator,
                                   PositiveMaskIterator>(
        image_a, {}, image_b, {});
}

/**
 * Compute the mutual information between image_a and image_b
 *
 * @param it_mask_b  Arbitrary mask for determining valid pixels of the image_b.
 *                   A pixel is valid iff its corresponding mask pixel is not 0.
 */
template <typename PixelType>
double mutual_information(const Mat& image_a,
                          const Mat& image_b,
                          const Mat::ConstIterator<MaskType>& it_mask_b)
{
    return mutual_information_impl<PixelType,
                                   PositiveMaskIterator,
                                   Mat::ConstIterator<MaskType>>(
        image_a, {}, image_b, it_mask_b);
}

/**
 * Compute the mutual information between image_a and image_b
 *
 * @param it_mask_a  Arbitrary mask for determining valid pixels of the image_a.
 *                   A pixel is valid iff its corresponding mask pixel is not 0.
 * @param it_mask_b  Arbitrary mask for determining valid pixels of the image_b.
 *                   A pixel is valid iff its corresponding mask pixel is not 0.
 */
template <typename PixelType>
double mutual_information(const Mat& image_a,
                          const Mat::ConstIterator<PixelType>& it_mask_a,
                          const Mat& image_b,
                          const Mat::ConstIterator<PixelType>& it_mask_b)
{
    return mutual_information_impl<PixelType,
                                   Mat::ConstIterator<MaskType>,
                                   Mat::ConstIterator<MaskType>>(
        image_a, it_mask_a, image_b, it_mask_b);
}

template <typename TransformClass>
void mutual_information_gradient(const Mat& reference,
                                 const Mat& steepest_img,
                                 const Mat& tracked,
                                 const Mat& tracked_mask,
                                 Mat& gradient)
{
    using std::vector;
    using BinningMethod = BSpline4;
    using PixelType     = float;
    using GradientType  = float;

    assert(reference.rows == tracked.rows);
    assert(reference.cols == tracked.cols);

    assert(reference.type() == tracked.type());

    assert(tracked_mask.is_mask_of(tracked));

    if (gradient.empty()) {
        gradient.create<GradientType>(1, TransformClass::number_parameters, 1);
    } else {
        assert(gradient.rows == 1);
        assert(gradient.cols == TransformClass::number_parameters);
        assert(gradient.channels() == 1);
    }

    gradient.fill<GradientType>(0);

    Mat histogram_r;
    Mat histogram_rt;
    Mat histogram_rt_grad;

    joint_hist_gradient<PixelType,
                        float,
                        BinningMethod,
                        PositiveMaskIterator,
                        Mat::ConstIterator<uint8_t>>(reference,
                                                     {},
                                                     steepest_img,
                                                     tracked,
                                                     tracked_mask,
                                                     histogram_r,
                                                     histogram_rt,
                                                     histogram_rt_grad);

    const int number_practical_bins = static_cast<int>(histogram_r.cols);
    const int number_parameters     = histogram_rt_grad.channels();

    Mat::ConstIterator<float> hist_r_it(histogram_r);
    Mat::ConstIterator<float> hist_rt_it(histogram_rt);
    Mat::ConstIterator<float> hist_rt_grad_it(histogram_rt_grad);

    Mat::Iterator<float> gradient_it(gradient);

    for (int i = 0; i < number_practical_bins; ++i) {
        for (int j = 0; j < number_practical_bins; ++j) {
            for (int param = 0; param < number_parameters; ++param) {
                float grad_at_ij = hist_rt_grad_it(i, j, param);
                float hist_at_ij = hist_rt_it(i, j, 0);
                float hist_at_j  = hist_r_it(0, j, 0);

                assert(hist_at_ij <= hist_at_j);

                if (hist_at_ij > 0.f) {
                    assert(hist_at_j > 0.f);

                    gradient_it(0, param, 0) +=
                        grad_at_ij * std::log(hist_at_ij / hist_at_j);
                } else {
                    assert(hist_at_ij == 0.f);
                }
            }
        }
    }

    return;
}

#endif
