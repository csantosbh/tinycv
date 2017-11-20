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
    for (int i = 0; i < number_practical_bins; ++i) {
        for (int j = 0; j < number_practical_bins; ++j) {
            const float prob_ij = hist_ab_it(i, j, 0);

            /*
             * We know that P(a=j,b=i) < P(a=j) and P(a=j,b=i) < P(b=i), since
             * (a=j,b=i) is a subset of both (a=j) and (b=i) events.
             *
             * Therefore, if P(a=j)=0 or P(b=i)=0, then P(a=j,b=i)=0.
             *
             * Now consider P(a=j,b=i)=0. Then, the MI term
             *
             *  MI(j,i) = P(a=j,b=i) * log(P(a=j,b=i) / (P(a=j) * P(b=i)))
             *
             * must evaluate to 0.
             *
             * Proof:
             * Let k = P(a=j,b=i), l=P(a=j) and m=P(b=i), for the sake of
             * simplicity. Then:
             *
             *  MI(j,i) = lim{k->0+} k * log(k/(l * m)).
             *
             * If l > 0 and m > 0, then it is trivial to see that MI(j,i) = 0.
             * If, however, both l and m are zero, we have
             *
             *  MI(j,i) = lim{k->0+} k * log(k/(k * k))
             *          = lim{k->0+} k * log(k) - k * log(k) - k * log(k)
             *
             * Each term k * log(k) can be written as log(k) / (1/k), so one can
             * use L'hopital rule to find out that each term of the sum
             * converges to 0.
             */
            if (prob_ij > 0.0) {
                double prob_aj = hist_a_it(0, j, 0);
                double prob_bi = hist_b_it(0, i, 0);

                assert(prob_aj > 0.f);
                assert(prob_bi > 0.f);

                double logterm = std::log(prob_ij / (prob_aj * prob_bi));
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

    double histogram_r_sum;
    double histogram_rt_sum;
    joint_hist_gradient<PixelType,
                        float,
                        BinningMethod,
                        TransformClass,
                        PositiveMaskIterator,
                        Mat::ConstIterator<MaskType>>(reference,
                                                      {},
                                                      steepest_img,
                                                      tracked,
                                                      tracked_mask,
                                                      histogram_r_sum,
                                                      histogram_rt_sum,
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

template <typename TransformClass>
void mutual_information_hessian(const Mat& reference,
                                const Mat& steepest_ref_grad,
                                const Mat& steepest_ref_hess,
                                const Mat& tracked,
                                const Mat& tracked_mask,
                                const Mat& histogram_r,
                                const Mat& histogram_rt,
                                const Mat& histogram_rt_grad,
                                const double histogram_r_sum,
                                const double histogram_rt_sum,
                                Mat& hessian)
{
    const int number_practical_bins = static_cast<int>(histogram_r.cols);
    const int transform_params      = TransformClass::number_parameters;

    using HessianPixelType = float;
    using BinningMethod    = BSpline4;
    using PixelType        = float;
    using SteepestType     = float;

    // clang-format off
    using SteepestColType  = Eigen::Matrix<HessianPixelType,
                                           transform_params,
                                           1>;
    using SteepestRowType = Eigen::Matrix<HessianPixelType,
                                           1,
                                           transform_params>;
    using HessianMatType   = Eigen::Matrix<HessianPixelType,
                                           transform_params,
                                           transform_params,
                                           Eigen::RowMajor>;
    // clang-format on

    // The input images must have the same dimensions
    assert(reference.cols == tracked.cols);
    assert(reference.rows == tracked.rows);

    // The steepest images must have the same dimensions as the input images
    assert(reference.cols == steepest_ref_grad.cols);
    assert(reference.rows == steepest_ref_grad.rows);
    assert(reference.cols == steepest_ref_hess.cols);
    assert(reference.rows == steepest_ref_hess.rows);

    // The steepest images must have the right amount of channels
    assert(steepest_ref_grad.channels() == transform_params);
    assert(steepest_ref_hess.channels() == transform_params * transform_params);

    // The input mask must be valid
    assert(tracked_mask.is_mask_of(tracked));

    // The histograms must have the right number of bins
    assert(histogram_r.rows == 1 && histogram_r.cols == number_practical_bins);
    assert(histogram_rt.rows == number_practical_bins &&
           histogram_rt.cols == number_practical_bins);
    assert(histogram_rt_grad.rows == number_practical_bins &&
           histogram_rt_grad.cols == number_practical_bins);

    // The histograms must have the correct number of channels
    assert(histogram_r.channels() == 1);
    assert(histogram_rt.channels() == 1);
    assert(histogram_rt_grad.channels() == transform_params);

    // Allocate output if it hasn't been allocated yet
    if (hessian.empty()) {
        hessian.create<HessianPixelType>(transform_params, transform_params, 1);
    } else {
        assert(hessian.cols == transform_params &&
               hessian.rows == transform_params && hessian.channels() == 1);
    }

    // Initialize hessian
    hessian.fill<HessianPixelType>(0);

    // Create histogram iterators
    Mat::ConstIterator<float> histogram_r_it(histogram_r);
    Mat::ConstIterator<float> histogram_rt_it(histogram_rt);
    Mat::ConstIterator<float> histogram_rt_grad_it(histogram_rt_grad);

    // Create output iterator
    Mat::Iterator<HessianPixelType> hessian_it(hessian);

    // Create joint histogram hessian and reference histogram gradient
    Mat histogram_rt_hess;
    Mat histogram_r_grad;

    joint_hist_hessian<PixelType,
                       SteepestType,
                       BinningMethod,
                       TransformClass,
                       PositiveMaskIterator,
                       Mat::ConstIterator<MaskType>>(
        reference,
        {},
        steepest_ref_grad,
        steepest_ref_hess,
        tracked,
        Mat::ConstIterator<MaskType>(tracked_mask),
        histogram_r_sum,
        histogram_rt_sum,
        histogram_r_grad,
        histogram_rt_hess);

    assert(histogram_rt_hess.rows == number_practical_bins &&
           histogram_rt_hess.cols == number_practical_bins);
    assert(histogram_rt_hess.channels() == transform_params * transform_params);

    // Create iterators for joint hessian and ref gradient
    Mat::Iterator<HessianPixelType> histogram_rt_hess_it(histogram_rt_hess);
    Mat::ConstIterator<HessianPixelType> histogram_r_grad_it(histogram_r_grad);

    // Create Eigen support structures that don't depend on histogram bin
    Eigen::Map<HessianMatType> hessian_mat(&hessian_it(0, 0, 0));

    for (int i = 0; i < number_practical_bins; ++i) {
        for (int j = 0; j < number_practical_bins; j++) {
            // Create Eigen support strucutres that depend on histogram bin
            Eigen::Map<const SteepestColType> histogram_rt_grad_col(
                &histogram_rt_grad_it(i, j, 0));

            Eigen::Map<const SteepestRowType> histogram_rt_grad_row(
                &histogram_rt_grad_it(i, j, 0));

            Eigen::Map<const SteepestRowType> histogram_r_grad_row(
                &histogram_r_grad_it(0, j, 0));

            Eigen::Map<const HessianMatType> histogram_rt_hess_mat(
                &histogram_rt_hess_it(i, j, 0));

            if (histogram_rt_it(i, j, 0) > 0.f) {
                // Make sure there are no divisions by zero
                assert(histogram_r_it(0, j, 0) > 0.f);
                assert(histogram_rt_it(i, j, 0) > 0.f);

                // Sanity check
                assert(histogram_r_it(0, j, 0) >= histogram_rt_it(i, j, 0));

                // Evaluate hessian expression
                hessian_mat +=
                    histogram_rt_grad_col *
                        (histogram_rt_grad_row / histogram_rt_it(i, j, 0) -
                         histogram_r_grad_row / histogram_r_it(0, j, 0)) +
                    histogram_rt_hess_mat * std::log(histogram_rt_it(i, j, 0) /
                                                     histogram_r_it(0, j, 0));
            }
        }
    }
}

#endif
