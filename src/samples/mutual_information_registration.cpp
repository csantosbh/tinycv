#include <iostream>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "tinycv/third_party/stb/stb_image.h"

#define TINYCV_IMPLEMENTATION
#include "tinycv/tinycv.h"

using stbibuf = std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>;

void load_stbibuf(stbibuf& output,
                  const char* filename,
                  int& width,
                  int& height,
                  int& channels)
{
    output = stbibuf(stbi_load(filename, &width, &height, &channels, 1),
                     [](uint8_t* buf) { stbi_image_free(buf); });

    channels = 1;
}

int main(int argc, char** argv)
{
    using std::vector;
    using namespace tinycv;

    int width;
    int height;
    int channels;

    /// Load input images
    stbibuf stb_source;
    load_stbibuf(stb_source, "../source.jpg", width, height, channels);

    stbibuf stb_destination;
    load_stbibuf(
        stb_destination, "../destination.jpg", width, height, channels);

    /// Convert input images to the Mat structure
    Mat source;
    source.create_from_buffer<uint8_t>(
        stb_source.get(), height, width, channels, width * channels);

    Mat destination;
    destination.create_from_buffer<uint8_t>(
        stb_destination.get(), height, width, channels, width * channels);

    /// Register input images
    Mat homography;

    register_homography(destination, source, homography);

    /// Warp source image with homography
    Mat transf_test;
    Mat transf_mask;
    image_transform<uint8_t,
                    1,
                    HomographyTransform<float>,
                    bilinear_sample<uint8_t, 1>>(
        source, homography, BoundingBox(source), transf_test, transf_mask);

    /// Draw contour of transformed <source> into <destination>

    // Corners are defined in clockwise order, starting at top left
    vector<Point<float>> source_corners{
        {0.f, 0.f},
        {static_cast<float>(source.cols) - 1.f, 0.f},
        {static_cast<float>(source.cols) - 1.f,
         static_cast<float>(source.rows) - 1.f},
        {0.f, static_cast<float>(source.rows) - 1.f}};

    for (size_t i = 0; i < source_corners.size(); ++i) {
        const Point<float> transf_corner_a =
            HomographyTransform<float>::transform(source_corners[i],
                                                  homography);

        const Point<float> transf_corner_b =
            HomographyTransform<float>::transform(
                source_corners[(i + 1) % source_corners.size()], homography);

        draw_line<uint8_t, 1>({static_cast<int>(round(transf_corner_a.x)),
                               static_cast<int>(round(transf_corner_a.y))},
                              {static_cast<int>(round(transf_corner_b.x)),
                               static_cast<int>(round(transf_corner_b.y))},
                              {0},
                              destination);
    }

    return 0;
}
