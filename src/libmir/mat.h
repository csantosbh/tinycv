#ifndef _LIBMIR_MAT_H_
#define _LIBMIR_MAT_H_

#include <memory>

class Mat {
public:
    enum class Type {
        UINT8 = 0,
        UINT16 = 2,
        INT16 = 3,
        INT32 = 4,
        FLOAT32 = 5,
        FLOAT64 = 6,
        UINT32 = 7
    };

    void* data;
    int cols; // Width
    int rows; // Height
    struct {
       size_t buf[2]; // Buf[0] = width of the containing
                      // buffer*channels; buff[1] = channels
    } step;

    Mat();

    Mat& operator=(Mat&& o);

    template<typename T>
    Mat& create_from_buffer(T* ptr,
                            int rows,
                            int cols,
                            int channels,
                            int stride);

    template<typename T>
    Mat& create(int rows,
                int cols,
                int channels);

    Type type() const;

    bool empty() const;

    void release();

    template<typename T>
    struct Iterator {
        Iterator(Mat& m_) : m(m_) {}

        Iterator<T>& operator=(Iterator<T>& o);

        T& operator()(int row,
                      int col,
                      int chan);

        Mat &m;
    };

    template<typename T>
    struct ConstIterator {
        ConstIterator(const Mat& m_) : m(m_) {}
        ConstIterator(const ConstIterator& cvIt) : m(cvIt.m_) {}

        ConstIterator<T>& operator=(ConstIterator<T>& o);

        const T& operator()(int row,
                            int col,
                            int chan) const;

        const Mat &m;
    };

private:
    std::shared_ptr<void> dataMgr;

    int flags_; // OpenCV-compatible flags

    template<typename T>
    void compute_flags(int channels);
};

constexpr Mat::Type SatTypeEnum = Mat::Type::INT32;
using SatType = int32_t;

#endif
