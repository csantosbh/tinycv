#include <cassert>

#include "mat.h"

using namespace std;

Mat::Mat()
    : data(nullptr)
    , cols(0)
    , rows(0)
{
}

Mat::Mat(const Mat& other, CopyMode copy_mode)
    : data(other.data)
    , cols(other.cols)
    , rows(other.rows)
    , step(other.step)
    , data_mgr_(other.data_mgr_)
    , flags_(other.flags_)
{
    if (copy_mode == CopyMode::Deep) {
        const size_t buffer_length = rows * cols * channels() * step.buf[2];

        data_mgr_ = shared_ptr<void>(new uint8_t[buffer_length],
                                     [](void* buf) { delete[] buf; });
        data      = data_mgr_.get();

        memcpy(data, other.data, buffer_length);
    }
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

template <typename T>
void Mat::compute_flags(int channels)
{
    const int CV_CN_SHIFT = 4;

    flags_ = (channels - 1) << CV_CN_SHIFT |
             static_cast<int>(Mat::get_type_enum<T>());
}

Mat::Type Mat::type() const
{
    return static_cast<Mat::Type>(flags_ & 0x0f);
}

template <typename T>
Mat& Mat::create(int height, int width, int channels)
{
    rows = height;
    cols = width;
    step = {static_cast<size_t>(cols * channels),
            static_cast<size_t>(channels),
            sizeof(T)};

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
    step = {stride, static_cast<size_t>(channels), sizeof(PixelType)};

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
Mat::ConstIterator<T>& Mat::ConstIterator<T>::
operator=(Mat::ConstIterator<T>& o)
{
    m = o.m;

    return *this;
}

template Mat& Mat::create<uint8_t>(int rows, int cols, int channels);

template Mat& Mat::create<int16_t>(int rows, int cols, int channels);

template Mat& Mat::create<int32_t>(int rows, int cols, int channels);

template Mat& Mat::create<float>(int rows, int cols, int channels);

template Mat& Mat::create_from_buffer<uint8_t>(uint8_t* ptr,
                                               int rows,
                                               int cols,
                                               int channels,
                                               size_t stride);

template Mat& Mat::create_from_buffer<float>(float* ptr,
                                             int rows,
                                             int cols,
                                             int channels,
                                             size_t stride);

template uint8_t& Mat::Iterator<uint8_t>::
operator()(int row, int col, int chan);

template int16_t& Mat::Iterator<int16_t>::
operator()(int row, int col, int chan);

template SatType& Mat::Iterator<SatType>::
operator()(int row, int col, int chan);

template float& Mat::Iterator<float>::operator()(int row, int col, int chan);

template const uint8_t& Mat::ConstIterator<uint8_t>::
operator()(int row, int col, int chan) const;

template const int16_t& Mat::ConstIterator<int16_t>::
operator()(int row, int col, int chan) const;

template const SatType& Mat::ConstIterator<SatType>::
operator()(int row, int col, int chan) const;

template const float& Mat::ConstIterator<float>::
operator()(int row, int col, int chan) const;
