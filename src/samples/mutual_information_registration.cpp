/**
 * Tracks a quadrilateral pattern through a sequence of images.
 *
 * Usage: Provide the left-top and right-bottom coordinates of the bounding box
 * of the target (as it lies in the first frame of the sequence) via command
 * line arguments.
 *
 * The names of all files to be processed must be given via stdin. Sample:
 *
 * ls *.png | ./mutual_information_registration <left_top_x> <left_top_y> \
 * <right_bottom_x> <right_bottom_y>
 */

#include <iostream>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "tinycv/third_party/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tinycv/third_party/stb/stb_image_write.h"

#define TINYCV_IMPLEMENTATION
#include "tinycv/tinycv.h"

using stbibuf = std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>;

tinycv::Mat imread(const char* filename)
{
    int width;
    int height;
    int channels;

    stbibuf stb_output =
        stbibuf(stbi_load(filename, &width, &height, &channels, 0),
                [](uint8_t* buf) { stbi_image_free(buf); });

    tinycv::Mat mat_output;
    mat_output.create_from_buffer<uint8_t>(
        stb_output, height, width, channels, width * channels);

    return mat_output;
}

void imwrite(const tinycv::Mat& buffer, const char* filename)
{
    stbi_write_png(filename,
                   buffer.cols,
                   buffer.rows,
                   buffer.channels(),
                   buffer.data,
                   static_cast<int>(buffer.row_stride()));
}

void check_cmd_parameters(int argc, char** argv)
{
    if (argc != 5) {
        std::cout << "Usage: " << argv[0]
                  << " <left_top_x> <left_top_y> <right_bottom_x> "
                     "<right_bottom_y>\nProvide the left-top and right-bottom "
                     "coordinates of the bounding box of the target (as it "
                     "lies in the first frame of the sequence) via command "
                     "line arguments.\nThe names of all files to be processed "
                     "must be given via stdin"
                  << std::endl;

        exit(0);
    }
}

int main(int argc, char** argv)
{
    using std::vector;
    using namespace tinycv;

    check_cmd_parameters(argc, argv);

    std::string curr_frame_name;
    std::getline(std::cin, curr_frame_name);

    /// Load input images
    Mat current_frame_rgb = imread(curr_frame_name.c_str());

    assert(current_frame_rgb.channels() == 3);

    /// Convert RGB frame to grayscale
    Mat current_frame;
    rgb_to_gray<uint8_t, uint8_t>(current_frame_rgb, current_frame);

    /// Create coordinates of bounding box of reference object
    Point<float> left_top{std::stof(argv[1]), std::stof(argv[2])};
    Point<float> right_bottom{std::stof(argv[3]), std::stof(argv[4])};

    /// Configure reference ROI
    NonLinearRegistration<HomographyTransform<float>> aligner;
    Mat ref_frame_roi = image_crop<uint8_t>(
        current_frame, BoundingBox({left_top, right_bottom}));
    aligner.set_reference(ref_frame_roi);

    /// Initialize transform initial guess
    Mat initial_guess;
    initial_guess.create<float>(
        1, HomographyTransform<float>::number_parameters, 1);

    Point<float> roi_diag = right_bottom - left_top;
    HomographyTransform<float>::from_matches(
        {{0, 0}, {roi_diag.x, 0}, roi_diag, {0, roi_diag.y}},
        {left_top,
         {right_bottom.x, left_top.y},
         right_bottom,
         {left_top.x, right_bottom.y}},
        initial_guess);

    /// Iterate through remaining frames
    int cnt=1;
    while (std::getline(std::cin, curr_frame_name)) {
        cnt++;
        /// Load new frame and convert to grayscale
        current_frame_rgb = imread(curr_frame_name.c_str());
        assert(current_frame_rgb.channels() == 3);
        rgb_to_gray<uint8_t, uint8_t>(current_frame_rgb, current_frame);

        /// Register subsequent frames
        Mat homography;
        aligner.register_image(current_frame, initial_guess, homography);
        initial_guess = homography;

        // The <homography> transforms from <current_frame> to <ref_frame_roi>.
        // In order to draw the tracked object in the current frame, we want a
        // homography that transforms from <ref_frame_roi> to <current_frame>,
        // that is, the inverse of <homography>.
        Mat draw_homography;
        HomographyTransform<float>::inverse(homography, draw_homography);

        // Print registration parameter
        using TransformColType = Eigen::Matrix<GradPixelType, 8, 1>;
        std::cout << "\n"
                  << Eigen::Map<TransformColType>(
                         static_cast<GradPixelType*>(homography.data))
                         .transpose()
                  << std::endl;

        /// Draw contour of transformed <ref_frame_roi> into <current_frame_rgb>
        // Corners are defined in clockwise order, starting at top left
        const vector<Point<float>> ref_roi_corners{
            {{0, 0}, {roi_diag.x, 0}, roi_diag, {0, roi_diag.y}}};

        for (size_t i = 0; i < ref_roi_corners.size(); ++i) {
            const Point<float> transf_corner_a =
                HomographyTransform<float>::transform(ref_roi_corners[i],
                                                      draw_homography);

            const Point<float> transf_corner_b =
                HomographyTransform<float>::transform(
                    ref_roi_corners[(i + 1) % ref_roi_corners.size()],
                    draw_homography);

            draw_line<uint8_t, 3>({static_cast<int>(round(transf_corner_a.x)),
                                   static_cast<int>(round(transf_corner_a.y))},
                                  {static_cast<int>(round(transf_corner_b.x)),
                                   static_cast<int>(round(transf_corner_b.y))},
                                  {255, 0, 0},
                                  current_frame_rgb);
        }

        /// Save drawn image
        std::stringstream drawn_frame_name;
        drawn_frame_name << curr_frame_name << ".tracked.png";
        imwrite(current_frame_rgb, drawn_frame_name.str().c_str());
    }

    return 0;
}
