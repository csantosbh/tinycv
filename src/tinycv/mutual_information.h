#ifndef _TINYCV_MUTUAL_INFORMATION_H_
#define _TINYCV_MUTUAL_INFORMATION_H_

#include "histogram.h"
#include "mat.h"

using MaskPixelType = uint8_t;

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

template <typename PixelType,
          typename BinningMethod,
          typename MaskIteratorA,
          typename MaskIteratorB>
double mutual_information_impl(const Mat& img_reference,
                               const MaskIteratorA& mask_reference_it,
                               const Mat& img_tracked,
                               const MaskIteratorB& mask_tracked_it)
{
    const int number_hist_bins =
        HistogramConfig::num_total_bins<BinningMethod>();

    // This function works for single channel images only
    assert(img_reference.channels() == 1);
    assert(img_tracked.channels() == 1);

    // The input images must have the same dimensions
    assert(img_reference.rows == img_tracked.rows);
    assert(img_reference.cols == img_tracked.cols);

    // The reference and tracked images must be of the same type
    assert(img_reference.type() == img_tracked.type());

    // The input masks must be valid
    assert(mask_reference_it.is_mask_of(img_reference));
    assert(mask_tracked_it.is_mask_of(img_tracked));

    Mat histogram_r;
    Mat histogram_t;
    Mat histogram_rt;
    joint_image_histogram<PixelType,
                          BinningMethod,
                          MaskIteratorA,
                          MaskIteratorB>(img_reference,
                                         mask_reference_it,
                                         img_tracked,
                                         mask_tracked_it,
                                         histogram_r,
                                         histogram_t,
                                         histogram_rt);

    // Histogram iterators
    Mat::ConstIterator<float> hist_r_it(histogram_r);
    Mat::ConstIterator<float> hist_t_it(histogram_t);
    Mat::ConstIterator<float> hist_rt_it(histogram_rt);

    double mi_summation = 0.0;
    for (int i = 0; i < number_hist_bins; ++i) {
        for (int j = 0; j < number_hist_bins; ++j) {
            const float prob_ij = hist_rt_it(i, j, 0);

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
                double prob_rj = hist_r_it(0, j, 0);
                double prob_ti = hist_t_it(0, i, 0);

                assert(prob_rj > 0.f);
                assert(prob_ti > 0.f);

                double logterm = std::log(prob_ij / (prob_rj * prob_ti));
                mi_summation += prob_ij * logterm;
            }

            assert(!std::isnan(mi_summation));
        }
    }

    return mi_summation;
}

/**
 * Compute the mutual information between img_reference and img_tracked
 */
template <typename PixelType, typename BinningMethod>
double mutual_information(const Mat& img_reference, const Mat& img_tracked)
{
    return mutual_information_impl<PixelType,
                                   BinningMethod,
                                   PositiveMaskIterator,
                                   PositiveMaskIterator>(
        img_reference, {}, img_tracked, {});
}

/**
 * Compute the mutual information between image_a and image_b
 *
 * @param mask_tracked_it  Arbitrary mask for determining valid pixels of the
 *                         img_tracked. A pixel is valid iff its corresponding
 *                         mask pixel is not 0.
 */
template <typename PixelType, typename BinningMethod>
double
mutual_information(const Mat& img_reference,
                   const Mat& img_tracked,
                   const Mat::ConstIterator<MaskPixelType>& mask_tracked_it)
{
    return mutual_information_impl<PixelType,
                                   BinningMethod,
                                   PositiveMaskIterator,
                                   Mat::ConstIterator<MaskPixelType>>(
        img_reference, {}, img_tracked, mask_tracked_it);
}

/**
 * Compute the mutual information between image_r and image_t
 *
 * @param mask_reference_it  Arbitrary mask for determining valid pixels of the
 *                           img_reference. A pixel is valid iff its
 *                           corresponding mask pixel is not 0.
 * @param mask_tracked_it  Arbitrary mask for determining valid pixels of the
 *                         img_tracked. A pixel is valid iff its corresponding
 *                         mask pixel is not 0.
 */
template <typename PixelType, typename BinningMethod>
double
mutual_information(const Mat& img_reference,
                   const Mat::ConstIterator<PixelType>& mask_reference_it,
                   const Mat& img_tracked,
                   const Mat::ConstIterator<PixelType>& mask_tracked_it)
{
    return mutual_information_impl<PixelType,
                                   BinningMethod,
                                   Mat::ConstIterator<MaskPixelType>,
                                   Mat::ConstIterator<MaskPixelType>>(
        img_reference, mask_reference_it, img_tracked, mask_tracked_it);
}

/**
 * Compute the gradient of the mutual information
 */
template <typename PixelType,
          typename GradPixelType,
          typename BinningMethod,
          typename TransformClass>
void mutual_information_gradient(const Mat& img_reference,
                                 const Mat& steepest_grad_r,
                                 const Mat& img_tracked,
                                 const Mat& mask_tracked,
                                 Mat& mi_gradient)
{
    const int number_hist_bins =
        HistogramConfig::num_total_bins<BinningMethod>();
    const int number_transform_params = TransformClass::number_parameters;

    // The input images must have the same dimensions
    assert(img_reference.rows == img_tracked.rows);
    assert(img_reference.cols == img_tracked.cols);

    // The steepest images must have the same dimensions as the input images
    assert(img_reference.cols == steepest_grad_r.cols);
    assert(img_reference.rows == steepest_grad_r.rows);

    // The reference and tracked images must be of the same type
    assert(img_reference.type() == Mat::get_type_enum<PixelType>());
    assert(img_reference.type() == img_tracked.type());

    // The images derived from the gradient must be of the same type
    assert(steepest_grad_r.type() == Mat::get_type_enum<GradPixelType>());

    // The input mask must be valid
    assert(mask_tracked.is_mask_of(img_tracked));

    if (mi_gradient.empty()) {
        mi_gradient.create<GradPixelType>(1, number_transform_params, 1);
    } else {
        assert(mi_gradient.rows == 1);
        assert(mi_gradient.cols == number_transform_params);
        assert(mi_gradient.channels() == 1);
    }

    mi_gradient.fill<GradPixelType>(0);

    Mat histogram_r;
    Mat histogram_rt;
    Mat histogram_rt_grad;

    double histogram_r_sum;
    double histogram_rt_sum;
    joint_hist_gradient<PixelType,
                        GradPixelType,
                        BinningMethod,
                        TransformClass,
                        PositiveMaskIterator,
                        Mat::ConstIterator<MaskPixelType>>(img_reference,
                                                           {},
                                                           steepest_grad_r,
                                                           img_tracked,
                                                           mask_tracked,
                                                           histogram_r_sum,
                                                           histogram_rt_sum,
                                                           histogram_r,
                                                           histogram_rt,
                                                           histogram_rt_grad);

    Mat::ConstIterator<float> hist_r_it(histogram_r);
    Mat::ConstIterator<float> hist_rt_it(histogram_rt);
    Mat::ConstIterator<float> hist_rt_grad_it(histogram_rt_grad);

    Mat::Iterator<float> gradient_it(mi_gradient);

    for (int i = 0; i < number_hist_bins; ++i) {
        for (int j = 0; j < number_hist_bins; ++j) {
            for (int param = 0; param < number_transform_params; ++param) {
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

template <typename PixelType,
          typename GradPixelType,
          typename BinningMethod,
          typename TransformClass,
          typename MaskIteratorT>
void mutual_information_hessian(const Mat& img_reference,
                                const Mat& steepest_grad_r,
                                const Mat& steepest_hess_r,
                                const Mat& img_tracked,
                                const MaskIteratorT& mask_tracked_it,
                                const Mat& histogram_r,
                                const Mat& histogram_rt,
                                const Mat& histogram_rt_grad,
                                const double histogram_r_sum,
                                const double histogram_rt_sum,
                                Mat& mi_hessian)
{
    const int number_hist_bins =
        HistogramConfig::num_total_bins<BinningMethod>();
    const int number_transform_params = TransformClass::number_parameters;

    // clang-format off
    using SteepestColType  = Eigen::Matrix<GradPixelType,
                                           number_transform_params,
                                           1>;
    using SteepestRowType = Eigen::Matrix<GradPixelType,
                                           1,
                                           number_transform_params>;
    using HessianMatType   = Eigen::Matrix<GradPixelType,
                                           number_transform_params,
                                           number_transform_params,
                                           Eigen::RowMajor>;
    // clang-format on

    // This function works for single channel images only
    assert(img_reference.channels() == 1);
    assert(img_tracked.channels() == 1);

    // The input images must have the same dimensions
    assert(img_reference.cols == img_tracked.cols);
    assert(img_reference.rows == img_tracked.rows);

    // The steepest images must have the same dimensions as the input images
    assert(img_reference.cols == steepest_grad_r.cols);
    assert(img_reference.rows == steepest_grad_r.rows);
    assert(img_reference.cols == steepest_hess_r.cols);
    assert(img_reference.rows == steepest_hess_r.rows);

    // The steepest images must have the right amount of channels
    assert(steepest_grad_r.channels() == number_transform_params);
    assert(steepest_hess_r.channels() ==
           number_transform_params * number_transform_params);

    // The input mask must be valid
    assert(mask_tracked_it.is_mask_of(img_tracked));

    // The histograms must have the right number of bins
    assert(histogram_r.rows == 1 && histogram_r.cols == number_hist_bins);
    assert(histogram_rt.rows == number_hist_bins &&
           histogram_rt.cols == number_hist_bins);
    assert(histogram_rt_grad.rows == number_hist_bins &&
           histogram_rt_grad.cols == number_hist_bins);

    // The histograms must have the correct number of channels
    assert(histogram_r.channels() == 1);
    assert(histogram_rt.channels() == 1);
    assert(histogram_rt_grad.channels() == number_transform_params);

    // The reference and tracked images must be of the same type
    assert(img_reference.type() == img_tracked.type());

    // The images derived from the gradient must be of the same type
    assert(steepest_grad_r.type() == Mat::get_type_enum<GradPixelType>());
    assert(steepest_grad_r.type() == steepest_hess_r.type());

    // Allocate output if it hasn't been allocated yet
    if (mi_hessian.empty()) {
        mi_hessian.create<GradPixelType>(
            number_transform_params, number_transform_params, 1);
    } else {
        assert(mi_hessian.cols == number_transform_params &&
               mi_hessian.rows == number_transform_params &&
               mi_hessian.channels() == 1);
    }

    // Initialize hessian
    mi_hessian.fill<GradPixelType>(0);

    // Create histogram iterators
    Mat::ConstIterator<float> histogram_r_it(histogram_r);
    Mat::ConstIterator<float> histogram_rt_it(histogram_rt);
    Mat::ConstIterator<float> histogram_rt_grad_it(histogram_rt_grad);

    // Create output iterator
    Mat::Iterator<GradPixelType> hessian_it(mi_hessian);

    // Create hessian of the joint histogram and the gradient of the marginal
    // reference histogram
    Mat histogram_rt_hess;
    Mat histogram_r_grad;

    joint_hist_hessian<PixelType,
                       GradPixelType,
                       BinningMethod,
                       TransformClass,
                       PositiveMaskIterator,
                       MaskIteratorT>(
        img_reference,
        {},
        steepest_grad_r,
        steepest_hess_r,
        img_tracked,
        mask_tracked_it,
        histogram_r_sum,
        histogram_rt_sum,
        histogram_r_grad,
        histogram_rt_hess);

    assert(histogram_rt_hess.rows == number_hist_bins &&
           histogram_rt_hess.cols == number_hist_bins);
    assert(histogram_rt_hess.channels() ==
           number_transform_params * number_transform_params);

    // Create iterators for joint hessian and ref gradient
    Mat::Iterator<GradPixelType> histogram_rt_hess_it(histogram_rt_hess);
    Mat::ConstIterator<GradPixelType> histogram_r_grad_it(histogram_r_grad);

    // Create Eigen support structures that don't depend on histogram bin
    Eigen::Map<HessianMatType> hessian_mat(&hessian_it(0, 0, 0));

    for (int i = 0; i < number_hist_bins; ++i) {
        for (int j = 0; j < number_hist_bins; j++) {
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
