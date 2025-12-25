#include <fcntl.h>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <cstdio>
#include <cctype>
#include <cerrno>
#include <cstring>

#include <ncnn/net.h>
#include <ncnn/mat.h>

using namespace cv;
using namespace std;

struct Object {
    Rect rect;
    int label;
    float prob;
};

// ======= COCO 80 class names (YOLOv8 default) =======
static const std::vector<std::string> COCO_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","dart","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","controller","vase","scissors","teddy bear","hair drier","toothbrush"
};

static std::string to_display_label_coco(std::string name)
{
    // 指定替換規則（完全比對 COCO 原始名稱）
    if (name == "cup")         return "Mug";
    if (name == "book")        return "Sticky note";
    if (name == "sports ball") return "Baseball";
    if (name == "tv")          return "tvmonitor";
    if (name == "cell phone")  return "Phone";
    if (name == "remote")      return "Controller";
    if (name == "bird")        return "Pigeon";

    // 其他：維持原本字串，只把第一個英文字母大寫
    for (size_t i = 0; i < name.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(name[i]);
        if (std::isalpha(ch)) {
            name[i] = static_cast<char>(std::toupper(ch));
            break;
        }
    }
    return name;
}

static inline std::string get_coco_name(int label)
{
    if (label >= 0 && label < (int)COCO_NAMES.size())
        return to_display_label_coco(COCO_NAMES[label]);
    return "Cls" + std::to_string(label);
}

// ======= Custom 4 class names =======
static const std::vector<std::string> CUSTOM_NAMES = {
    "Tissue",
    "Dart",
    "Pencil",
    "Poker card"
};

static inline std::string get_custom_name(int label)
{
    if (label >= 0 && label < (int)CUSTOM_NAMES.size())
        return CUSTOM_NAMES[label];
    return "Cls" + std::to_string(label);
}

// ================== IoU / NMS ==================
static float intersection_area(const Object& a, const Object& b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_custom(const vector<Object>& objects, vector<Object>& picked, float nms_thresh)
{
    picked.clear();
    if (objects.empty()) return;

    vector<int> idx(objects.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(), [&](int a, int b){
        return objects[a].prob > objects[b].prob;
    });

    for (int i : idx)
    {
        const Object& a = objects[i];
        bool keep = true;
        for (const auto& b : picked)
        {
            float inter = intersection_area(a, b);
            float uni   = a.rect.area() + b.rect.area() - inter;
            if (uni <= 0.f) continue;

            float iou = inter / uni;
            if (iou > nms_thresh)
            {
                keep = false;
                break;
            }
        }
        if (keep) picked.push_back(a);
    }
}

// ================== Framebuffer ==================
struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

static framebuffer_info get_framebuffer_info(const char *path)
{
    framebuffer_info fb{};
    fb_var_screeninfo info{};
    int fd = open(path, O_RDWR);
    if (fd >= 0)
    {
        if (ioctl(fd, FBIOGET_VSCREENINFO, &info) == 0)
        {
            fb.xres_virtual   = info.xres_virtual;
            fb.yres_virtual   = info.yres_virtual;
            fb.bits_per_pixel = info.bits_per_pixel;
        }
        close(fd);
    }
    return fb;
}

// ================== Letterbox 前處理 ==================
static ncnn::Mat letterbox(const Mat& img, int target_size, float& scale, int& pad_x, int& pad_y)
{
    int w = img.cols;
    int h = img.rows;

    float r = std::min((float)target_size / w, (float)target_size / h);
    int new_w = (int)std::round(w * r);
    int new_h = (int)std::round(h * r);

    scale = r;
    pad_x = (target_size - new_w) / 2;
    pad_y = (target_size - new_h) / 2;

    Mat resized;
    resize(img, resized, Size(new_w, new_h));

    Mat canvas(target_size, target_size, CV_8UC3, Scalar(0, 0, 0));
    resized.copyTo(canvas(Rect(pad_x, pad_y, new_w, new_h)));

    // 統一用 BGR2RGB
    ncnn::Mat in = ncnn::Mat::from_pixels(
        canvas.data, ncnn::Mat::PIXEL_BGR2RGB,
        target_size, target_size
    );

    const float norm_vals[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};
    in.substract_mean_normalize(nullptr, norm_vals);

    return in;
}

// ================== 安全寫 JPG（避免偶發壞檔） ==================
static bool safe_imwrite_jpg(const std::string& out_file, const cv::Mat& img, int quality = 95)
{
    std::string tmp = out_file + ".tmp";
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, quality };

    bool ok = false;
    try {
        ok = cv::imwrite(tmp, img, params);
    } catch (const cv::Exception& e) {
        std::cerr << "[ERR] imwrite exception: " << e.what() << "\n";
        ok = false;
    }

    if (!ok) {
        std::cerr << "[ERR] imwrite failed: " << tmp << "\n";
        return false;
    }

    if (::rename(tmp.c_str(), out_file.c_str()) != 0) {
        std::cerr << "[ERR] rename failed: " << std::strerror(errno) << "\n";
        return false;
    }

    ::sync(); // 板子上比較保險
    return true;
}

// ================== 推論 + 畫框（通用） ==================
template <typename NameFunc>
static int infer_and_draw(
    ncnn::Net& net,
    cv::Mat& img_inplace,
    int input_size,
    int num_classes,
    float conf_thresh,
    float nms_thresh,
    const char* in_blob,
    const char* out_blob,
    NameFunc get_name,
    const cv::Scalar& box_color,
    const std::string& prefix
) {
    float scale = 1.f;
    int pad_x = 0, pad_y = 0;
    ncnn::Mat in = letterbox(img_inplace, input_size, scale, pad_x, pad_y);

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input(in_blob, in) != 0) {
        std::cerr << "[ERR] ex.input failed: " << in_blob << "\n";
        return -1;
    }

    ncnn::Mat out;
    if (ex.extract(out_blob, out) != 0) {
        std::cerr << "[ERR] ex.extract failed: " << out_blob << "\n";
        return -1;
    }

    int img_w = img_inplace.cols;
    int img_h = img_inplace.rows;

    int attrs = out.h;
    int num_proposals = out.w;

    bool has_obj_conf = false;
    if (attrs == 5 + num_classes) {
        has_obj_conf = true;
    } else if (attrs == 4 + num_classes) {
        has_obj_conf = false;
    } else {
        // 重要：shape 不符直接退出，避免越界造成記憶體亂掉 -> 可能導致 JPG 偶發壞檔
        std::cerr << "[ERR] unexpected out.h=" << attrs
                  << " expected " << (4 + num_classes) << " or " << (5 + num_classes)
                  << " (classes=" << num_classes << ")\n";
        return -1;
    }

    vector<Object> proposals;
    proposals.reserve(256);

    for (int i = 0; i < num_proposals; i++)
    {
        float cx = out.row(0)[i];
        float cy = out.row(1)[i];
        float w  = out.row(2)[i];
        float h  = out.row(3)[i];

        float obj_conf = 1.f;
        int cls_start_row = 4;

        if (has_obj_conf) {
            obj_conf      = out.row(4)[i];
            cls_start_row = 5;
            if (obj_conf < conf_thresh) continue;
        }

        int best_cls = -1;
        float best_cls_score = 0.f;

        for (int c = 0; c < num_classes; c++)
        {
            float cls_score = out.row(cls_start_row + c)[i];
            if (cls_score > best_cls_score) {
                best_cls_score = cls_score;
                best_cls = c;
            }
        }

        float score = obj_conf * best_cls_score;
        if (score < conf_thresh) continue;

        float x0 = cx - w * 0.5f;
        float y0 = cy - h * 0.5f;
        float x1 = cx + w * 0.5f;
        float y1 = cy + h * 0.5f;

        x0 -= pad_x; x1 -= pad_x;
        y0 -= pad_y; y1 -= pad_y;

        x0 /= scale; x1 /= scale;
        y0 /= scale; y1 /= scale;

        x0 = std::max(0.f, std::min((float)img_w - 1.f, x0));
        y0 = std::max(0.f, std::min((float)img_h - 1.f, y0));
        x1 = std::max(0.f, std::min((float)img_w - 1.f, x1));
        y1 = std::max(0.f, std::min((float)img_h - 1.f, y1));

        Object obj;
        obj.rect  = Rect(Point((int)x0, (int)y0), Point((int)x1, (int)y1));
        obj.label = best_cls;
        obj.prob  = score;
        proposals.push_back(obj);
    }

    vector<Object> objects;
    nms_custom(proposals, objects, nms_thresh);

    // draw
    for (const auto& o : objects)
    {
        rectangle(img_inplace, o.rect, box_color, 2);

        std::string name = get_name(o.label);
        char text[160];
        std::snprintf(text, sizeof(text), "%s%s %.2f", prefix.c_str(), name.c_str(), o.prob);

        Point org = o.rect.tl();
        org.y = std::max(0, org.y - 5);
        putText(img_inplace, text, org, FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
    }

    return (int)objects.size();
}

// ================== 主程式：讀一次圖，先 COCO 再 finetune ==================
int main()
{
    // ======= COCO model (先跑) =======
    string coco_param = "./yolov8x.ncnn.param";
    string coco_bin   = "./yolov8x.ncnn.bin";
    const int COCO_INPUT = 640;
    const int COCO_CLASSES = 80;

    // ======= finetune model (後跑) =======
    string ft_param = "./yolov8s.ncnn.param";
    string ft_bin   = "./yolov8s.ncnn.bin";
    const int FT_INPUT = 960;
    const int FT_CLASSES = 4;

    // ======= IO =======
    string image_file = "./sample.jpg";     // 讀進來的圖（只讀一次）
    string out_file   = "./result.jpg"; // 同一張輸出：先 COCO 再 finetune

    const float CONF_THRESH = 0.25f;
    const float NMS_THRESH  = 0.45f;

    const char* IN_BLOB  = "in0";
    const char* OUT_BLOB = "out0";

    // 1) load models
    ncnn::Net net_coco;
    net_coco.opt.num_threads = 4;
    net_coco.opt.use_vulkan_compute = false;
    net_coco.opt.use_fp16_storage = false;
    net_coco.opt.use_fp16_arithmetic = false;
    net_coco.opt.use_int8_storage = false;
    net_coco.opt.use_int8_arithmetic = false;

    if (net_coco.load_param(coco_param.c_str()) != 0 || net_coco.load_model(coco_bin.c_str()) != 0) {
        std::cerr << "[ERR] load COCO model failed: " << coco_param << " / " << coco_bin << "\n";
        return -1;
    }

    ncnn::Net net_ft;
    net_ft.opt.num_threads = 4;
    net_ft.opt.use_vulkan_compute = false;
    net_ft.opt.use_fp16_storage = false;
    net_ft.opt.use_fp16_arithmetic = false;
    net_ft.opt.use_int8_storage = false;
    net_ft.opt.use_int8_arithmetic = false;

    if (net_ft.load_param(ft_param.c_str()) != 0 || net_ft.load_model(ft_bin.c_str()) != 0) {
        std::cerr << "[ERR] load finetune model failed: " << ft_param << " / " << ft_bin << "\n";
        return -1;
    }

    std::cout << "[OK] models loaded\n";

    // 2) read image once
    Mat img = imread(image_file);
    if (img.empty()) {
        std::cerr << "[ERR] imread failed: " << image_file << "\n";
        return -1;
    }
    std::cout << "[OK] image: " << img.cols << " x " << img.rows << "\n";

    // 3) run COCO first (green)
    int coco_cnt = infer_and_draw(
        net_coco, img,
        COCO_INPUT, COCO_CLASSES,
        CONF_THRESH, NMS_THRESH,
        IN_BLOB, OUT_BLOB,
        [](int label){ return get_coco_name(label); },
        Scalar(0, 255, 0),
        ""   // 你也可以改成 "[COCO] " 做前綴
    );
    if (coco_cnt < 0) return -1;
    std::cout << "[OK] COCO done, boxes=" << coco_cnt << "\n";

    // 4) run finetune second (red)
    int ft_cnt = infer_and_draw(
        net_ft, img,
        FT_INPUT, FT_CLASSES,
        CONF_THRESH, NMS_THRESH,
        IN_BLOB, OUT_BLOB,
        [](int label){ return get_custom_name(label); },
        Scalar(0, 0, 255),
        ""   // 你也可以改成 "[FT] " 做前綴
    );
    if (ft_cnt < 0) return -1;
    std::cout << "[OK] finetune done, boxes=" << ft_cnt << "\n";

    // 5) save output (safe)
	imwrite(out_file, img);
    std::cout << "[OK] saved: " << out_file << "\n";

    // 6) framebuffer display
    framebuffer_info fb = get_framebuffer_info("/dev/fb0");
    int fb_w = fb.xres_virtual;
    int fb_h = fb.yres_virtual;

    int fb_fd = open("/dev/fb0", O_RDWR);
    if (fb_fd < 0) {
        std::cerr << "[WARN] open /dev/fb0 failed\n";
        return 0;
    }

    long int screensize = fb_w * fb_h * fb.bits_per_pixel / 8;
    char* fbp = (char*)mmap(0, screensize, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fb_fd, 0);
    if ((long)fbp == -1) {
        std::cerr << "[WARN] framebuffer mmap failed\n";
        close(fb_fd);
        return 0;
    }

    Mat disp;
    resize(img, disp, Size(fb_w, fb_h));

    if (fb.bits_per_pixel == 16) {
        Mat bgr565;
        cvtColor(disp, bgr565, COLOR_BGR2BGR565);
        memcpy(fbp, bgr565.data, screensize);
    } else {
        Mat bgra;
        cvtColor(disp, bgra, COLOR_BGR2BGRA);
        memcpy(fbp, bgra.data, screensize);
    }

    munmap(fbp, screensize);
    close(fb_fd);

    std::cout << "[OK] framebuffer displayed\n";
    return 0;
}
