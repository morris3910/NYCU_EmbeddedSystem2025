#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>

struct framebuffer_info
{
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

// ----- helper: check if directory exists -----
bool dir_exists(const char *path)
{
    DIR *dir = opendir(path);
    if (dir)
    {
        closedir(dir);
        return true;
    }
    else
    {
        return false;
    }
}

// ----- helper: create directory recursively -----
bool make_dir(const char *path)
{
    if (mkdir(path, 0777) == 0)
        return true;
    if (errno == EEXIST)
        return true;
    return false;
}

// ----- non-blocking keyboard -----
int kbhit()
{
    termios oldt, newt;
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
    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

int main(int argc, const char *argv[])
{
    cv::Mat frame;
    cv::VideoCapture camera(2);

    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0", std::ios::out | std::ios::binary);

    if (!camera.isOpened())
    {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }

    int fb_width = fb_info.xres_virtual;
    int fb_height = fb_info.yres_virtual;
    double target_aspect = 4.0 / 3.0;

    // ---------- Create new screenshot directory ----------
    std::string base_path = ".";
    int folder_id = 0;
    char path_buf[256];

    while (true)
    {
        snprintf(path_buf, sizeof(path_buf), "%s/screenshot_%d", base_path.c_str(), folder_id);
        if (!dir_exists(path_buf))
        {
            make_dir(path_buf);
            break;
        }
        folder_id++;
    }

    std::string save_dir = path_buf;
    std::cout << "Saving screenshots to: " << save_dir << std::endl;

    int screenshot_count = 0;

    while (true)
    {
        if (!camera.read(frame))
            continue;

        double cam_aspect = static_cast<double>(frame.cols) / frame.rows;
        int new_width, new_height;
        if (cam_aspect > target_aspect)
        {
            new_width = fb_width;
            new_height = static_cast<int>(fb_width / target_aspect);
        }
        else
        {
            new_height = fb_height;
            new_width = static_cast<int>(fb_height * target_aspect);
        }

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(new_width, new_height));

        cv::Mat display(fb_height, fb_width, CV_8UC3, cv::Scalar(0, 0, 0));
        int x_offset = (fb_width - resized.cols) / 2;
        int y_offset = (fb_height - resized.rows) / 2;
        resized.copyTo(display(cv::Rect(x_offset, y_offset, resized.cols, resized.rows)));

        cv::Mat frame_bgr565;
        cv::cvtColor(display, frame_bgr565, cv::COLOR_BGR2BGR565);

        for (int y = 0; y < fb_height; y++)
        {
            std::streamoff row_offset =
                static_cast<std::streamoff>(y) *
                static_cast<std::streamoff>(fb_info.xres_virtual) *
                static_cast<std::streamoff>(fb_info.bits_per_pixel / 8);
            ofs.seekp(row_offset, std::ios::beg);

            const char *row_ptr = reinterpret_cast<const char *>(frame_bgr565.ptr(y));
            std::size_t bytes_to_write =
                static_cast<std::size_t>(fb_width) * static_cast<std::size_t>(fb_info.bits_per_pixel / 8);
            ofs.write(row_ptr, static_cast<std::streamsize>(bytes_to_write));
        }

        // ---------- Non-blocking key detection ----------
        if (kbhit())
        {
            char c = getchar();
            if (c == 'c')
            {
                char filename[512];
                snprintf(filename, sizeof(filename), "%s/%d.bmp", save_dir.c_str(), screenshot_count++);
                cv::imwrite(filename, frame);
                std::cout << "Captured: " << filename << std::endl;
            }
        }
    }

    camera.release();
    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;
    int fd = open(framebuffer_device_path, O_RDWR);
    if (fd >= 0)
    {
        if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == 0)
        {
            fb_info.xres_virtual = screen_info.xres_virtual;
            fb_info.yres_virtual = screen_info.yres_virtual;
            fb_info.bits_per_pixel = screen_info.bits_per_pixel;
        }
        else
        {
            fb_info.xres_virtual = fb_info.yres_virtual = fb_info.bits_per_pixel = 0;
        }
        close(fd);
    }
    else
    {
        fb_info.xres_virtual = fb_info.yres_virtual = fb_info.bits_per_pixel = 0;
    }
    return fb_info;
}
