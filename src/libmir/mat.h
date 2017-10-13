#ifndef _LIBMIR_MAT_H_
#define _LIBMIR_MAT_H_

#include <memory>

struct BoundingBox;

class Mat
{
  public:
    enum class Type {
        UINT8   = 0,
        UINT16  = 2,
        INT16   = 3,
        INT32   = 4,
        FLOAT32 = 5,
        FLOAT64 = 6,
        UINT32  = 7
    };

    void* data;
    int cols; // Width
    int rows; // Height

    Mat();

    Mat(const Mat& other);

    Mat& operator=(const Mat& o);

    Mat& operator=(Mat&& o);

    template <typename PixelType>
    Mat& create_from_buffer(PixelType* ptr,
                            int rows,
                            int cols,
                            int channels,
                            size_t stride);

    template <typename T>
    Mat& create(int rows, int cols, int channels);

    Type type() const;

    bool empty() const;

    void release();

    int channels() const;

    size_t row_stride() const;

    template <typename T>
    struct Iterator
    {
        Iterator(Mat& m_)
            : m(m_)
        {
        }

        Iterator<T>& operator=(Iterator<T>& o);

        T& operator()(int row, int col, int chan);

        Mat& m;
    };

    template <typename T>
    struct ConstIterator
    {
        ConstIterator(const Mat& m_)
            : m(m_)
        {
        }
        ConstIterator(const ConstIterator& cvIt)
            : m(cvIt.m)
        {
        }

        ConstIterator<T>& operator=(ConstIterator<T>& o);

        const T& operator()(int row, int col, int chan) const;

        const Mat& m;
    };

    /// Friends
    template<typename T>
    friend Mat image_crop(const Mat& image, const BoundingBox& crop_bb);

  private:
    struct
    {
        size_t buf[2]; // buf[0] = width of the containing buffer*channels;
                       // buf[1] = channels
    } step;

    std::shared_ptr<void> data_mgr_;

    int flags_; // OpenCV-compatible flags

    template <typename T>
    void compute_flags(int channels);
};

constexpr Mat::Type SatTypeEnum = Mat::Type::INT32;
using SatType                   = int32_t;

#endif
