#include <cassert>

#include "mat.h"

using namespace std;

Mat::Mat()
    : data(nullptr)
    , cols(0)
    , rows(0)
{
}

Mat::Mat(const Mat& other)
    : data(other.data)
    , cols(other.cols)
    , rows(other.rows)
    , step(other.step)
    , data_mgr_(other.data_mgr_)
    , flags_(other.flags_)
{
}

Mat& Mat::operator=(const Mat& o)
{
    data      = o.data;
    cols      = o.cols;
    rows      = o.rows;
    step      = o.step;
    data_mgr_ = o.data_mgr_;
    flags_    = o.flags_;

    return *this;
}

Mat& Mat::operator=(Mat&& o)
{
    data      = o.data;
    cols      = o.cols;
    rows      = o.rows;
    step      = o.step;
    data_mgr_ = std::move(o.data_mgr_);
    flags_    = o.flags_;

    return *this;
}

bool Mat::empty() const
{
    return data == nullptr;
}

void Mat::release()
{
    // Intentionally, "data" is not changed so we are left with an invalid
    // pointer.
    data_mgr_.reset();
    data = nullptr;
}

int Mat::channels() const
{
    return static_cast<int>(step.buf[1]);
}

size_t Mat::row_stride() const
{
    return step.buf[0];
}

template <typename T>
void Mat::compute_flags(int channels)
{
    const int CV_CN_SHIFT = 4;

    flags_ = (channels - 1) << CV_CN_SHIFT;

    if (is_same<T, uint8_t>::value) {
        flags_ |= static_cast<int>(Type::UINT8);
    } else if (is_same<T, uint16_t>::value) {
        flags_ |= static_cast<int>(Type::UINT16);
    } else if (is_same<T, int16_t>::value) {
        flags_ |= static_cast<int>(Type::INT16);
    } else if (is_same<T, int32_t>::value) {
        flags_ |= static_cast<int>(Type::INT32);
    } else if (is_same<T, float>::value) {
        flags_ |= static_cast<int>(Type::FLOAT32);
    } else if (is_same<T, double>::value) {
        flags_ |= static_cast<int>(Type::FLOAT64);
    } else if (is_same<T, uint32_t>::value) {
        flags_ |= static_cast<int>(Type::UINT32);
    }
}

Mat::Type Mat::type() const
{
    return static_cast<Mat::Type>(flags_ & 0x0f);
}

template <typename T>
Mat& Mat::create(int height, int width, int channels)
{
    cols = height;
    rows = width;
    step = {static_cast<size_t>(cols * channels),
            static_cast<size_t>(channels)};

    compute_flags<T>(channels);

    data_mgr_ = shared_ptr<T>(new T[rows * cols * channels],
                              [](T* buf) { delete[] buf; });
    data      = data_mgr_.get();

    return *this;
}

template <typename PixelType>
Mat& Mat::create_from_buffer(PixelType* ptr,
                             int height,
                             int width,
                             int channels,
                             size_t stride)
{
    data = ptr;
    cols = width;
    rows = height;
    step = {stride, static_cast<size_t>(channels)};

    compute_flags<PixelType>(channels);

    return *this;
}

template <typename T>
Mat::Iterator<T>& Mat::Iterator<T>::operator=(Mat::Iterator<T>& o)
{
    m = o.m;

    return *this;
}

template <typename T>
T& Mat::Iterator<T>::operator()(int row, int col, int chan)
{
    assert(row >= 0);
    assert(row < m.rows);
    assert(col >= 0);
    assert(col < m.cols);

    return (static_cast<T*>(
        m.data))[row * m.step.buf[0] + col * m.step.buf[1] + chan];
}

template <typename T>
Mat::ConstIterator<T>& Mat::ConstIterator<T>::
operator=(Mat::ConstIterator<T>& o)
{
    m = o.m;

    return *this;
}

template <typename T>
const T& Mat::ConstIterator<T>::operator()(int row, int col, int chan) const
{
    assert(row >= 0);
    assert(row < m.rows);
    assert(col >= 0);
    assert(col < m.cols);

    return (static_cast<T*>(
        m.data))[row * m.step.buf[0] + col * m.step.buf[1] + chan];
}

template Mat& Mat::create<uint8_t>(int rows, int cols, int channels);

template Mat& Mat::create<int32_t>(int rows, int cols, int channels);

template Mat& Mat::create_from_buffer<uint8_t>(uint8_t* ptr,
                                               int rows,
                                               int cols,
                                               int channels,
                                               size_t stride);

template uint8_t& Mat::Iterator<uint8_t>::
operator()(int row, int col, int chan);

template SatType& Mat::Iterator<SatType>::
operator()(int row, int col, int chan);

template const uint8_t& Mat::ConstIterator<uint8_t>::
operator()(int row, int col, int chan) const;

template const SatType& Mat::ConstIterator<SatType>::
operator()(int row, int col, int chan) const;
