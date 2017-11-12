#ifndef _LIBMIR_MUTUAL_INFORMATION_H_
#define _LIBMIR_MUTUAL_INFORMATION_H_

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
};

template <typename PixelType, typename MaskIteratorA, typename MaskIteratorB>
double mutual_information_impl(const Mat& image_a,
                               const MaskIteratorA& it_mask_a,
                               const Mat& image_b,
                               const MaskIteratorB& it_mask_b)
{
    using std::vector;
    using BinningMethod = BSpline4;

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

#endif
