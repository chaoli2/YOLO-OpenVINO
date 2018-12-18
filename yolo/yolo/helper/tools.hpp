#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <object.hpp>

using namespace std;

namespace tools{

void addRectangles(cv::Mat data, size_t height, size_t width, std::vector<DetectedObject> detectedObjects) {
    std::vector<Color> colors = {
        {128, 64,  128},
        {232, 35,  244},
        {70,  70,  70},
        {156, 102, 102},
        {153, 153, 190},
        {153, 153, 153},
        {30,  170, 250},
        {0,   220, 220},
        {35,  142, 107},
        {152, 251, 152},
        {180, 130, 70},
        {60,  20,  220},
        {0,   0,   255},
        {142, 0,   0},
        {70,  0,   0},
        {100, 60,  0},
        {90,  0,   0},
        {230, 0,   0},
        {32,  11,  119},
        {0,   74,  111},
        {81,  0,   81}
    };

    int center_w = width / 2;
    int center_h = height / 2;
    cout << "width: " << data.cols << " height: " << data.rows << endl;
    cout << "intput_width: " << width << " intput_height: " << height << endl;
    cout << "center_w: " << center_w << " center_h: " << center_h << endl;

    cv::Mat3b Dst(width, height);
    cv::resize(data, Dst, cv::Size(width, height));

    for (size_t i = 0; i < detectedObjects.size(); i++) {
        int cls = detectedObjects[i].objectType % colors.size();
        cv::Vec3b color;
        color[0] = colors.at(cls).red();
        color[1] = colors.at(cls).green();
        color[2] = colors.at(cls).blue();

        int xmin = detectedObjects[i].xmin + center_w;
        int xmax = detectedObjects[i].xmax + center_w;
        int ymin = detectedObjects[i].ymin + center_h;
        int ymax = detectedObjects[i].ymax + center_h;

        cv::Point p1 (xmin, ymax);
        cv::Point p2 (xmax, ymin);

        cout << i << " - [" << xmin << "," << ymin << "]\t[" << xmax << "," << ymax << "]\t" 
        << detectedObjects[i].objectType << "\tprob: " << detectedObjects[i].prob 
        << endl;

        cv::rectangle(Dst, p1, p2, color);
    }
    cv::imshow("Yolo", Dst);
    cv::waitKey(0);
}

void addRectangles(unsigned char *data, size_t height, size_t width, std::vector<DetectedObject> detectedObjects) {
    std::vector<Color> colors = {
        {128, 64,  128},
        {232, 35,  244},
        {70,  70,  70},
        {156, 102, 102},
        {153, 153, 190},
        {153, 153, 153},
        {30,  170, 250},
        {0,   220, 220},
        {35,  142, 107},
        {152, 251, 152},
        {180, 130, 70},
        {60,  20,  220},
        {0,   0,   255},
        {142, 0,   0},
        {70,  0,   0},
        {100, 60,  0},
        {90,  0,   0},
        {230, 0,   0},
        {32,  11,  119},
        {0,   74,  111},
        {81,  0,   81}
    };

    for (size_t i = 0; i < detectedObjects.size(); i++) {
        int cls = detectedObjects[i].objectType % colors.size();

        int xmin = detectedObjects[i].xmin * width;
        int xmax = detectedObjects[i].xmax * width;
        int ymin = detectedObjects[i].ymin * height;
        int ymax = detectedObjects[i].ymax * height;

        size_t shift_first = ymin*width*3;
        size_t shift_second = ymax*width*3;
        for (int x = xmin; x < xmax; x++) {
            data[shift_first + x*3] = colors.at(cls).red();
            data[shift_first + x*3 + 1] = colors.at(cls).green();
            data[shift_first + x*3 + 2] = colors.at(cls).blue();
            data[shift_second + x*3] = colors.at(cls).red();
            data[shift_second + x*3 + 1] = colors.at(cls).green();
            data[shift_second + x*3 + 2] = colors.at(cls).blue();
        }

        shift_first = xmin*3;
        shift_second = xmax*3;
        for (int y = ymin; y < ymax; y++) {
            data[shift_first + y*width*3] = colors.at(cls).red();
            data[shift_first + y*width*3 + 1] = colors.at(cls).green();
            data[shift_first + y*width*3 + 2] = colors.at(cls).blue();
            data[shift_second + y*width*3] = colors.at(cls).red();
            data[shift_second + y*width*3 + 1] = colors.at(cls).green();
            data[shift_second + y*width*3 + 2] = colors.at(cls).blue();
        }
    }
}

void dispObj(vector<float>& obj){
    cout << "size: " << obj.size() << endl;
    for(int i = 0; i <  obj.size(); i ++){
        cout << i << " - " << obj[i] << endl;
    }
}

/**
 * \brief This function analyses the YOLO net output for a single class
 * @param net_out - The output data
 * @param class_num - The class number
 * @return a list of found boxes
 */
void yoloNetParseOutput(const float *net_out) {
    cout << "YoloV2 Parse Output" << endl;
    float threshold = 0.2f;         // The confidence threshold
    int C = 80;                     // classes
    int B = 5;                      // bounding boxes
    int S = 19;                     // cell size

    // Parse Output
    vector<vector<float>> objs (S*S, vector<float>());
    for(int i = 0; i < B * (B + C); i++){
        for(int r = 0; r < S; r++){
            for(int c = 0; c < S; c++){
                int idx = c + r * S + i * S * S;
                assert(idx < S * S * B * (B + C));
                objs.at(c + r * S).push_back(net_out[idx]);
            }
        }
    }

    vector<helper::object::DetectedObject> DetectedObjects;
    // Parse Obj Boxes
    for(auto obj : objs){
        for(int b = 0; b < B; b++){
            for(int i = 0; i < B * C; i++){
                helper::object::DetectedObject obj ();

            }
        }
    }
    // dispObj(objs[19*19-1]);
}

std::vector<std::vector<float> > print_detections(int imw, int imh, int num, float thresh, box *boxes, float **probs, int classes) {
    std::vector<std::vector<float> > ret;
    int i;
    for (i = 0; i < num; ++i) {
        int idxClass = max_index(probs[i], classes);
        float prob = probs[i][idxClass];

        if (prob > thresh) {
            box b = boxes[i];
            int left = (b.x - b.w / 2.) * imw;
            int right = (b.x + b.w / 2.) * imw;
            int top = (b.y - b.h / 2.) * imh;
            int bot = (b.y + b.h / 2.) * imh;

            if (left < 0) left = 0;
            if (right > imw - 1) right = imw - 1;
            if (top < 0) top = 0;
            if (bot > imh - 1) bot = imh - 1;

            if(1){
                //dets: (left,top,right,bottom,classid,confident)
                std::vector<float> v(6);
                v[0] = b.x - b.w / 2.;
                v[1] = b.y - b.h / 2.;
                v[2] = b.x + b.w / 2.;
                v[3] = b.y + b.h / 2.;
                v[4] = idxClass;
                v[5] = prob;
                ret.push_back(v);
            }else{
                std::string label;
                label = std::to_string(idxClass);
                label = "label " +label;
                printf("%s: %.0f%% ", label.c_str(), prob * 100);
                printf("[(%d %d), (%d %d)]\n", left, top, right, bot);
            }
        }
    }
    return ret;
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh) {
    int i, j, k;
    sortable_bbox *s = (sortable_bbox *) calloc(total, sizeof(sortable_bbox));

    for (i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].cclass = 0;
        s[i].probs = probs;
    }

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            s[i].cclass = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for (i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for (j = i + 1; j < total; ++j) {
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh) {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }

    free(s);
}


void get_region_boxes(float *predictions, int lw, int lh, int lcoords, int lclasses, int lnum,
    int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int relative, const float *anchors) {
    int i,j,n;

    for (i = 0; i < lw * lh; ++i){
        int row = i / lw;
        int col = i % lw;
        for(n = 0; n < lnum; ++n){
            int index = n * lw * lh + i;
            for(j = 0; j < lclasses; ++j){
                probs[index][j] = 0;
            }
            int obj_index = entry_index(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords);
            int box_index = entry_index(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, 0);
            float scale = predictions[obj_index];

            boxes[index] = get_region_box(predictions, anchors, n, box_index, col, row, lw, lh, lw * lh);

            float max = 0;
            for(j = 0; j < lclasses; ++j){
                int class_index = entry_index(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1 + j);
                float prob = scale * predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if(prob > max) max = prob;
            }
            probs[index][lclasses] = max;

        }
    }
    correct_region_boxes(boxes, lw * lh * lnum, w, h, netw, neth, relative);
}


std::vector<std::vector<float> > yolov2_postprocess(float *data, int size, const float *anchors, float thresh, float nms, int classes) {

    std::vector<std::vector<float> > ret;
    int coords = 4;
    int num = 5;

    int imw = 416;
    int imh = 416;

    int lw = 19;
    int lh = 19;

    box *boxes = (box *) malloc(lw * lh * num * sizeof(box));
    float **probs = (float **) malloc(lw * lh * num * sizeof(float *));
    for (int j = 0; j < lw * lh * num ; ++j)
        probs[j] = (float *) malloc((classes + 1) * sizeof(float));

    get_region_boxes(data, lw, lh, coords, classes, num, imw, imh, imw, imh, thresh, probs, boxes, 1, anchors);
    do_nms_sort(boxes, probs, lw * lh * num, classes, nms);
    ret = print_detections(imw, imh, lw * lh * num, thresh, boxes, probs, classes);

    for (int j = 0; j < lw * lh * num ; ++j)
        free(probs[j]);
    free(probs);
    free(boxes);
    
    return ret;
}

}