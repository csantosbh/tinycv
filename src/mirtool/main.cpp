#include <iostream>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
using stbibuf = std::unique_ptr<uint8_t,
                                std::function<void(uint8_t*)> >;

#include "../libmir/registration.h"

using namespace std;

void load_stbibuf(stbibuf& output,
                  const char* filename,
                  int& width,
                  int& height,
                  int& channels) {
    output = stbibuf(stbi_load(filename, &width, &height, &channels, 1),
                   [](uint8_t* buf) {
        stbi_image_free(buf);
    });

    channels = 1;
}

int main(int argc,
         char** argv) {
    int width;
    int height;
    int channels;

    stbibuf stb_source;
    load_stbibuf(stb_source, "../images/house.jpg",
                 width, height, channels);

    stbibuf stb_destination;
    load_stbibuf(stb_destination,
                 "../images/house.jpg",
                 width, height, channels);

    Mat source;
    source.create_from_buffer<uint8_t>(stb_source.get(),
                                     height, width, channels,
                                     width * channels);

    Mat destination;
    destination.create_from_buffer<uint8_t>(stb_destination.get(),
                                          height, width, channels,
                                          width * channels);

    Eigen::Vector2f translation;
    register_translation(source, destination, translation);

    return 0;
}
