#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

using namespace cv;
using namespace std;

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t yres_virtual;
};

framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    framebuffer_info fb_info;
    fb_var_screeninfo screen_info;
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

int main() {
    string cfgFile = "./yolov3.cfg";
    string weightsFile = "./yolov3_best.weights";
    string imagePath = "./final_demo.jpg";
    string outputPath = "./final_result_4.jpg";

    // ---- 讀取圖片 ----
    Mat img = imread(imagePath);
    if (img.empty()) {
        cerr << "❌ 讀取圖片失敗：" << imagePath << endl;
        return -1;
    }

    int H = img.rows;
    int W = img.cols;

    cout << "圖片大小: " << W << "x" << H << endl;

    // ---- 用 cfg + weights 載入 YOLOv3 ----
    dnn::Net net = dnn::readNetFromDarknet(cfgFile, weightsFile);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // YOLOv3 輸出層名稱
    vector<String> outNames = net.getUnconnectedOutLayersNames();

    // ---- 製作 blob ----
    Mat blob;
    dnn::blobFromImage(img, blob, 1/255.0, Size(608, 608), Scalar(), true, false);
    net.setInput(blob);

    // ---- forward ----
    vector<Mat> outs;
    net.forward(outs, outNames);

    vector<Rect> boxes;
    vector<float> confidences;

    float confThreshold = 0.1f;
    float nmsThreshold = 0.3f;

    // ---- 處理每個 output ----
    for (int i = 0; i < (int)outs.size(); i++) {
        Mat &out = outs[i];
        float *data = (float *)out.data;

        for (int j = 0; j < out.rows; j++, data += out.cols) {
            float confidence = data[4];

            if (confidence > confThreshold) {
                // 單一 class = helmet
                float score = data[5] * confidence;
                if (score > confThreshold) {

                    float centerX = data[0] * W;
                    float centerY = data[1] * H;
                    float width   = data[2] * W;
                    float height  = data[3] * H;

                    int left = (int)(centerX - width / 2);
                    int top  = (int)(centerY - height / 2);

                    boxes.push_back(Rect(left, top, (int)width, (int)height));
                    confidences.push_back(score);
                }
            }
        }
    }

    // ---- NMS ----
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (int k = 0; k < (int)indices.size(); k++) {
        int idx = indices[k];
        rectangle(img, boxes[idx], Scalar(0, 255, 0), 3);
        putText(img, "Helmet", boxes[idx].tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,255,0), 2);
    }

    imwrite(outputPath, img);
    cout << "結果輸出到：" << outputPath << endl;

    // ---- Framebuffer 顯示 ----
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    ofstream ofs("/dev/fb0", ios::out | ios::binary);

    if (!ofs.is_open()) {
        cerr << "⚠️ 無法開啟 framebuffer" << endl;
        return 0;
    }

    int fb_w = fb_info.xres_virtual;
    int fb_h = fb_info.yres_virtual;

    Mat resized;

    double fb_aspect = (double)fb_w / fb_h;
    double img_aspect = (double)W / H;

    if (img_aspect > fb_aspect)
        resize(img, resized, Size(fb_w, fb_w / img_aspect));
    else
        resize(img, resized, Size(fb_h * img_aspect, fb_h));

    Mat canvas(fb_h, fb_w, CV_8UC3, Scalar(0,0,0));
    int x_off = (fb_w - resized.cols) / 2;
    int y_off = (fb_h - resized.rows) / 2;
    resized.copyTo(canvas(Rect(x_off, y_off, resized.cols, resized.rows)));

    Mat bgr565;
    cvtColor(canvas, bgr565, COLOR_BGR2BGR565);

    for (int y = 0; y < fb_h; y++) {
        streamoff offset = (streamoff)y * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);
        ofs.seekp(offset, ios::beg);
        ofs.write((char*)bgr565.ptr(y), fb_w * (fb_info.bits_per_pixel / 8));
    }

    ofs.close();
    cout << "📺 HDMI 顯示完成！" << endl;

    return 0;
}
