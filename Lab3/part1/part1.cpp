#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>
#include <map>

using namespace cv;
using namespace cv::face;
using namespace std;

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

bool dir_exists(const char *path) {
    DIR *dir = opendir(path);
    if (dir) {
        closedir(dir);
        return true;
    }
    return false;
}

int kbhit() {
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
    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

// load labels.txt
map<int, string> loadLabels(const string &path) {
    map<int, string> labels;
    ifstream file(path.c_str());
    if (!file.is_open()) {
        cerr << "Cannot open labels.txt" << endl;
        return labels;
    }
    int id;
    string name;
    while (file >> id >> name) {
        labels[id] = name;
    }
    return labels;
}

int main(int argc, const char *argv[]) {
    Mat frame;
    VideoCapture camera(2);
    if (!camera.isOpened()) {
        cerr << "cannot open camara" << endl;
        return 1;
    }
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    camera.set(CAP_PROP_BUFFERSIZE, 1);

    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    ofstream ofs("/dev/fb0", ios::out | ios::binary);
    if (!ofs.is_open()) {
        cerr << "cannot open /dev/fb0" << endl;
        return 1;
    }

    // ====== load Haar model ======
    CascadeClassifier face_cascade;
    if (!face_cascade.load("./haarcascade_frontalface_default.xml")) {
        cerr << "cannot load Haar model！" << endl;
        return 1;
    }

    // ====== load LBPH and labels ======
    string model_path = "./lbph_model.yml";
    string label_path = "./labels.txt";

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->read(model_path);
    map<int, string> label_map = loadLabels(label_path);

    cout << "Successfully loaded LBPH model and labels" << endl;

    int fb_width = fb_info.xres_virtual;
    int fb_height = fb_info.yres_virtual;
    double target_aspect = 4.0 / 3.0;

    vector<Rect> faces;

    while (true) {
        if (!camera.read(frame)) continue;

        // ---- face detect ----
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, Size(80, 80), Size(250, 250));

        // ---- face identify ----
        for (size_t i = 0; i < faces.size(); i++) {
	    Rect face = faces[i]; 
	    Mat roi = gray(face);
	    resize(roi, roi, Size(128, 128));

	    int label;
	    double confidence;
	    model->predict(roi, label, confidence);

	    string name = "Unknown";
	    Scalar color = Scalar(0, 0, 255);
	    if (confidence < 80 && label_map.count(label)) {
		name = label_map[label];
		color = Scalar(0, 255, 0);
	    }

	    char conf_text[32];
	    sprintf(conf_text, "(%.1f)", confidence);

	    string text = name + " " + conf_text;
	    rectangle(frame, face, color, 2);
	    putText(frame, text, Point(face.x, face.y - 10),
		    FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
	}

        // ---- Resize to framebuffer ----
        double scale = 0.5;
	int display_width  = static_cast<int>(fb_info.xres_virtual * scale);
	int display_height = static_cast<int>(fb_info.yres_virtual * scale);

	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(display_width, display_height));

	// 黑色背景畫布
	cv::Mat display(fb_info.yres_virtual, fb_info.xres_virtual, CV_8UC3, cv::Scalar(0, 0, 0));

	// 置中顯示
	int x_offset = (fb_info.xres_virtual - resized.cols) / 2;
	int y_offset = (fb_info.yres_virtual - resized.rows) / 2;
	resized.copyTo(display(cv::Rect(x_offset, y_offset, resized.cols, resized.rows)));


        // ---- transfer to BGR565 and write into framebuffer ----
        Mat frame_bgr565;
        cvtColor(display, frame_bgr565, COLOR_BGR2BGR565);

        for (int y = 0; y < fb_height; y++) {
            streamoff row_offset = static_cast<streamoff>(y) *
                                   static_cast<streamoff>(fb_info.xres_virtual) *
                                   static_cast<streamoff>(fb_info.bits_per_pixel / 8);
            ofs.seekp(row_offset, ios::beg);
            const char *row_ptr = reinterpret_cast<const char *>(frame_bgr565.ptr(y));
            size_t bytes_to_write =
                static_cast<size_t>(fb_width) * static_cast<size_t>(fb_info.bits_per_pixel / 8);
            ofs.write(row_ptr, static_cast<streamsize>(bytes_to_write));
        }

        // ---- q to exit ----
        if (kbhit()) {
            char c = getchar();
            if (c == 'q') {
                cout << "exit" << endl;
                break;
            }
        }
    }

    camera.release();
    ofs.close();
    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    struct framebuffer_info fb_info;
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
