#ifndef _LIBMIR_MAT_H_
#define _LIBMIR_MAT_H_

#include <cassert>
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

    template <typename PixelType>
    Mat& create(int rows, int cols, int channels);

    template <typename PixelType>
    Mat& operator<<(const std::initializer_list<PixelType>& fill_data)
    {
        const int num_channels = channels();

        assert(fill_data.size() ==
               static_cast<size_t>(rows * cols * num_channels));

        auto src_it = fill_data.begin();
        Iterator<PixelType> dst_it(*this);

        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                for (int c = 0; c < num_channels; ++c) {
                    dst_it(y, x, c) = *src_it;
                    ++src_it;
                }
            }
        }

        return *this;
    }

    Type type() const;

    bool empty() const;

    void release();

    int channels() const;

    size_t row_stride() const;

    template <typename PixelType>
    static Type get_type_enum()
    {
        if (std::is_same<PixelType, uint8_t>::value) {
            return Type::UINT8;
        } else if (std::is_same<PixelType, uint16_t>::value) {
            return Type::UINT16;
        } else if (std::is_same<PixelType, int16_t>::value) {
            return Type::INT16;
        } else if (std::is_same<PixelType, int32_t>::value) {
            return Type::INT32;
        } else if (std::is_same<PixelType, float>::value) {
            return Type::FLOAT32;
        } else if (std::is_same<PixelType, double>::value) {
            return Type::FLOAT64;
        } else if (std::is_same<PixelType, uint32_t>::value) {
            return Type::UINT32;
        }
    }

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
    template <typename T>
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
