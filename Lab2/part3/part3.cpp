#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <vector>
#include "lodepng.h"  // include your downloaded decoder

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

// ----- Non-blocking keyboard input -----
int kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

// ----- Load PNG using lodepng (C interface, no zlib needed) -----
cv::Mat load_png_lodepng(const std::string &filename) {
    unsigned char *image = NULL;
    unsigned width, height;

    unsigned error = lodepng_decode32_file(&image, &width, &height, filename.c_str());
    if (error) {
        std::cerr << "Error decoding PNG: " << lodepng_error_text(error) << std::endl;
        if (image) free(image);
        return cv::Mat();
    }

    cv::Mat img_rgba(height, width, CV_8UC4, image);
    cv::Mat img_bgr;
    cv::cvtColor(img_rgba, img_bgr, cv::COLOR_RGBA2BGR);

    free(image);
    return img_bgr;
}

int main() {
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    int fb_width = fb_info.xres_virtual;
    int fb_height = fb_info.yres_virtual;
    int fb_bpp = fb_info.bits_per_pixel;

    if (fb_width == 0 || fb_height == 0) {
        std::cerr << "Failed to get framebuffer info." << std::endl;
        return 1;
    }

    std::ofstream ofs("/dev/fb0", std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open /dev/fb0" << std::endl;
        return 1;
    }

    // --- Load PNG via lodepng (no libpng needed) ---
    cv::Mat img = load_png_lodepng("advance.png");
    if (img.empty()) {
        std::cerr << "Failed to decode advanced.png!" << std::endl;
        return 1;
    }

    // Fit height to screen, keep aspect ratio
    double scale = (double)fb_height / img.rows;
    int scaled_w = (int)(img.cols * scale);
    int scaled_h = fb_height;
    cv::Mat scaled;
    cv::resize(img, scaled, cv::Size(scaled_w, scaled_h));

    // Create doubled image for seamless wraparound
    cv::Mat doubled;
    cv::hconcat(scaled, scaled, doubled);  // concatenate horizontally

    int x_offset = 0;
    int direction = 1;  // 1 = right, -1 = left
    int total_w = doubled.cols;

    std::cout << "Electronic scroll board running (loop mode).\n";
    std::cout << "J → move left,  L → move right,  Q → quit\n";

    while (true) {
        // Wrap offset to create infinite scrolling
        if (direction == 1) {
            x_offset += 50;
            if (x_offset >= scaled_w) x_offset = 0;
        } else {
            x_offset -= 50;
            if (x_offset < 0) x_offset = scaled_w - 1;
        }

        // Crop visible region (within doubled image)
        cv::Mat view = doubled(cv::Rect(x_offset, 0, fb_width, fb_height));

        cv::Mat bgr565;
        cv::cvtColor(view, bgr565, cv::COLOR_BGR2BGR565);

        // Write to framebuffer
        for (int y = 0; y < fb_height; y++) {
            std::streamoff pos = static_cast<std::streamoff>(y) *
                                 static_cast<std::streamoff>(fb_info.xres_virtual) *
                                 static_cast<std::streamoff>(fb_bpp / 8);
            ofs.seekp(pos, std::ios::beg);
            const char *row_ptr = reinterpret_cast<const char *>(bgr565.ptr(y));
            std::size_t bytes_to_write =
                static_cast<std::size_t>(fb_width) * static_cast<std::size_t>(fb_bpp / 8);
            ofs.write(row_ptr, static_cast<std::streamsize>(bytes_to_write));
        }

        // Keyboard control
        if (kbhit()) {
            char c = getchar();
            if (c == 'j' || c == 'J') direction = -1;  // move left
            if (c == 'l' || c == 'L') direction = 1;   // move right
            if (c == 'q' || c == 'Q') break;           // quit
        }

        usleep(30000);  // smooth refresh
    }

    ofs.close();
    std::cout << "Program exited.\n";
    return 0;
}

framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;
    int fd = open(framebuffer_device_path, O_RDWR);
    if (fd >= 0) {
        if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == 0) {
            fb_info.xres_virtual = screen_info.xres_virtual;
            fb_info.yres_virtual = screen_info.yres_virtual;
            fb_info.bits_per_pixel = screen_info.bits_per_pixel;
        } else {
            fb_info.xres_virtual = fb_info.yres_virtual = fb_info.bits_per_pixel = 0;
        }
        close(fd);
    } else {
        fb_info.xres_virtual = fb_info.yres_virtual = fb_info.bits_per_pixel = 0;
    }
    return fb_info;
}
