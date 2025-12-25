#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <termios.h>
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>   // NEW

using namespace std;
using namespace cv;

//================ YOLO Settings ================
const int INPUT_SIZE = 320;
const int NUM_CLASSES = 80;
const float CONF_THRESH = 0.1f;
const float NMS_THRESH = 0.45f;
const int SKIP_FRAMES = 2;    // 每 2 frame 才推論一次 YOLO

struct Object {
    Rect rect;
    int label;
    float prob;
};

//================ 只保留的 8 個類別 ================
// COCO index:
// bottle=39, cup=41, spoon=44, banana=46,
// keyboard=66, cell phone=67, book=73, scissors=76

// 判斷這個類別是不是我們要的 8 種之一  // NEW
bool is_target_class(int cls)
{
    switch (cls) {
        case 39: // bottle
        case 41: // cup
        case 44: // spoon
        case 46: // banana
        case 66: // keyboard
        case 67: // cell phone
        case 73: // book
        case 76: // scissors
            return true;
        default:
            return false;
    }
}

// 把類別 id 轉成對應名字              // NEW
std::string class_name(int cls)
{
    switch (cls) {
        case 39: return "bottle";
        case 41: return "cup";
        case 44: return "spoon";
        case 46: return "banana";
        case 66: return "keyboard";
        case 67: return "cell phone";
        case 73: return "book";
        case 76: return "scissors";
        default: return "unknown";
    }
}

//================ NMS ================
float intersection_area(const Object &a, const Object &b) {
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void nms_custom(const vector<Object> &objs, vector<Object> &picked, float thr) {
    picked.clear();
    if (objs.empty()) return;

    vector<int> idx(objs.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return objs[a].prob > objs[b].prob;
    });

    for (int i : idx) {
        const Object &a = objs[i];
        bool keep = true;

        for (const auto &b : picked) {
            float inter = intersection_area(a, b);
            float uni = a.rect.area() + b.rect.area() - inter;
            if (uni <= 0.f) continue;

            if (inter / uni > thr) {
                keep = false;
                break;
            }
        }
        if (keep) picked.push_back(a);
    }
}

//================ Framebuffer ================
struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres;
    uint32_t yres;
};

framebuffer_info get_fb_info(const char *path) {
    framebuffer_info fb{};
    fb_var_screeninfo info{};

    int fd = open(path, O_RDWR);
    if (fd >= 0) {
        if (ioctl(fd, FBIOGET_VSCREENINFO, &info) == 0) {
            fb.xres = info.xres_virtual;
            fb.yres = info.yres_virtual;
            fb.bits_per_pixel = info.bits_per_pixel;
        }
        close(fd);
    }
    return fb;
}

//================ Keyboard ================
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

//================ Letterbox ================
ncnn::Mat letterbox(const Mat &img, int target, float &scale, int &pad_x, int &pad_y) {
    int w = img.cols, h = img.rows;

    float r = min((float)target / w, (float)target / h);
    int nw = round(w * r);
    int nh = round(h * r);

    scale = r;
    pad_x = (target - nw) / 2;
    pad_y = (target - nh) / 2;

    Mat resized;
    resize(img, resized, Size(nw, nh));

    Mat canvas(target, target, CV_8UC3, Scalar(0, 0, 0));
    resized.copyTo(canvas(Rect(pad_x, pad_y, nw, nh)));

    ncnn::Mat in = ncnn::Mat::from_pixels(canvas.data, ncnn::Mat::PIXEL_BGR, target, target);
    const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(nullptr, norm);

    return in;
}

//================ Main ================
int main() {
    // Load YOLO model
    ncnn::Net net;
    net.opt.num_threads = 4;
    net.opt.use_fp16_storage = true;
    net.opt.use_vulkan_compute = false;

    if (net.load_param("./yolov8n320.ncnn.param") ||
        net.load_model("./yolov8n320.ncnn.bin")) {
        cerr << "Failed to load YOLO model\n";
        return 1;
    }

    // Open camera
    VideoCapture cam(2);
    if (!cam.isOpened()) {
        cerr << "Camera not found\n";
        return 1;
    }
    cam.set(CAP_PROP_FRAME_WIDTH, 640);
    cam.set(CAP_PROP_FRAME_HEIGHT, 480);
    cam.set(CAP_PROP_BUFFERSIZE, 1);

    // Framebuffer mmap
    framebuffer_info fb = get_fb_info("/dev/fb0");
    int fb_w = fb.xres, fb_h = fb.yres;

    int fb_fd = open("/dev/fb0", O_RDWR);
    long screensize = fb_w * fb_h * fb.bits_per_pixel / 8;
    unsigned short *fbp = (unsigned short *)mmap(
        0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);

    if ((long)fbp == -1) {
        cerr << "Framebuffer mmap failed\n";
        return 1;
    }

    Mat frame;
    vector<Object> last_detection;   // 上一個 YOLO 的偵測結果
    int frame_count = 0;

    while (true) {
        cam.read(frame);
        if (frame.empty()) continue;

        frame_count++;

        // ---- 只有每 SKIP_FRAMES frame 才跑 YOLO ----
        if (frame_count % SKIP_FRAMES == 0) {
            float scale; int pad_x, pad_y;
            ncnn::Mat in = letterbox(frame, INPUT_SIZE, scale, pad_x, pad_y);

            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);

            ncnn::Mat out;
            ex.extract("out0", out);

            int attrs = out.h;
            int num = out.w;
            bool has_obj = (attrs == 5 + NUM_CLASSES);

            vector<Object> props;

            for (int i = 0; i < num; i++) {
                float cx = out.row(0)[i];
                float cy = out.row(1)[i];
                float w  = out.row(2)[i];
                float h  = out.row(3)[i];

                float obj = has_obj ? out.row(4)[i] : 1.f;
                if (obj < CONF_THRESH) continue;

                int cls_start = has_obj ? 5 : 4;
                int best_cls = -1;
                float best_score = 0.f;

                for (int c = 0; c < NUM_CLASSES; c++) {
                    float s = out.row(cls_start + c)[i];
                    if (s > best_score) {
                        best_score = s;
                        best_cls = c;
                    }
                }

                float score = obj * best_score;
                if (score < CONF_THRESH) continue;

                // 只保留我們指定的 8 個類別     // NEW
                if (!is_target_class(best_cls)) continue;

                float x0 = (cx - w/2 - pad_x) / scale;
                float y0 = (cy - h/2 - pad_y) / scale;
                float x1 = (cx + w/2 - pad_x) / scale;
                float y1 = (cy + h/2 - pad_y) / scale;

                Object o;
                o.rect = Rect(Point(x0, y0), Point(x1, y1));
                o.label = best_cls;
                o.prob = score;

                props.push_back(o);
            }

            nms_custom(props, last_detection, NMS_THRESH);
        }

        // ---- 畫上次 YOLO 偵測出的框 + 類別名稱 ----
        for (auto &o : last_detection) {
            // 畫框
            rectangle(frame, o.rect, Scalar(0, 255, 0), 2);

            // 準備文字：類別名稱 + 機率百分比     // NEW
            int prob_percent = (int)(o.prob * 100 + 0.5f);
            string label_text = class_name(o.label) + " " + to_string(prob_percent) + "%";

            int baseLine = 0;
            Size textSize = getTextSize(label_text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int x = o.rect.x;
            int y = o.rect.y - 5;
            if (y < textSize.height) y = textSize.height + 5;

            // 先畫一個背景方塊讓文字比較清楚   // NEW
            rectangle(frame,
                      Point(x, y - textSize.height - 2),
                      Point(x + textSize.width + 2, y + baseLine),
                      Scalar(0, 255, 0), FILLED);

            // 再畫白色文字                // NEW
            putText(frame, label_text, Point(x + 1, y - 2),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        // ---- 顯示到 framebuffer ----
        Mat resized;
        resize(frame, resized, Size(fb_w, fb_h));

        Mat bgr565;
        cvtColor(resized, bgr565, COLOR_BGR2BGR565);

        memcpy(fbp, bgr565.data, screensize);

        if (kbhit() && getchar() == 'q') break;
    }

    munmap(fbp, screensize);
    close(fb_fd);

    return 0;
}
