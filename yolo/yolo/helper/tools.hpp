#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "object.hpp"

using namespace std;

namespace tools{

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    // float* prob;
    vector<float> prob = vector<float>(80);
    float objectness;
    int sort_class;

    friend ostream& operator<<(ostream& out, const detection& self){
        out << "Class: " << self.classes 
        << " [" << self.bbox.x << ", " << self.bbox.y 
        << "] [" << self.bbox.w << ", " << self.bbox.h << "]"
        << " score: " << self.objectness; 
        return out;
    }

} detection;

int entry_index(int IH, int IW, int location, int entry){
    int n =   location / (IW*IH);
    int loc = location % (IW*IH);
    // printf("l.w: %i l.h: %i l.coords: %i loc: %i entry: %i return: %i\n",
    //     IH,
    //     IW,
    //     4,
    //     location,
    //     entry,
    //     n*IW*IH*(4+80+1) + entry*IW*IH + loc);
    return n*IW*IH*(4+80+1) + entry*IW*IH + loc;
}

box get_region_box(const float *x, int n, int index, int i, int j, int w, int h, int stride){
    box b;
    float biases[10] = {
        0.572730, 0.677385, 
        1.874460, 2.062530, 
        3.338430, 5.474340, 
        7.882820, 3.527780, 
        9.770520, 9.168280
    };
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

void correct_region_boxes(vector<detection>& dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


vector<detection> yoloNetParseOutput(const float* output_data, int IH, int IW){
    vector<detection> dets (IW*IH*5);
    float thresh = 0.2;
    for(int i =0; i < IH*IW; i++){
        int row = i / IW;
        int col = i % IW;
        for(int n = 0; n < 5; n++){
            int index = n*IW*IH + i;
            int obj_index  = entry_index(IH, IW, n*IW*IH + i, 4);
            int box_index  = entry_index(IH, IW, n*IW*IH + i, 0);
            float scale = output_data[obj_index];
            dets[index].bbox = get_region_box(output_data, n, box_index, col, row, IW, IH, IW*IH);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].objectness)
            printf("scale:%f [%f, %f]-[%f, %f] score:%f obj_idx:%i box_idx:%i\n", 
                scale, 
                dets[index].bbox.x, dets[index].bbox.y,
                dets[index].bbox.w, dets[index].bbox.h,
                dets[index].objectness,
                obj_index,
                box_index);


            int class_index = entry_index(IH, IW, n*IW*IH + i, 5);
            if(dets[index].objectness){
                for(int j = 0; j < 80; ++j){
                    int class_index = entry_index(IH, IW, n*IW*IH + i, 4 + 1 + j);
                    float prob = scale*output_data[class_index];
                    // assert(prob >= 0);
                    // assert(prob <= 1);
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
                // printf("scale:%f [%f, %f]-[%f, %f] score:%f obj_idx:%i box_idx:%i\n", 
                //     scale, 
                //     dets[index].bbox.x, dets[index].bbox.y,
                //     dets[index].bbox.w, dets[index].bbox.h,
                //     dets[index].objectness,
                //     obj_index,
                //     box_index);
            }
        }

    }

    // for(int index = 0; index < 19*19*5; index++){
    //     if(dets[index].objectness > 0.4){
    //         printf("[%f, %f]-[%f, %f] score:%f\n", 
    //             dets[index].bbox.x, dets[index].bbox.y,
    //             dets[index].bbox.w, dets[index].bbox.h,
    //             dets[index].objectness);
    //     }
    // }
    correct_region_boxes(dets, IW*IH*5, 1, 1, IW, IH, 0);


    return dets;
}

bool nms_comparator(tools::detection a, tools::detection b)
{
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return true;
    return false;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(vector<tools::detection>& src_dets, int total, int classes, 
    float thresh, vector<tools::detection>& dets){

    for(int i = 0; i < src_dets.size(); ++i){
        if(src_dets[i].objectness){
            dets.push_back(src_dets.at(i));
        }
    }

    for(int k = 0; k < classes; ++k){
        for(int i = 0; i < dets.size(); ++i){
            dets[i].sort_class = k;
        }
        sort(dets.begin(), dets.end(), nms_comparator);
        for(int i = 0; i < dets.size(); ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(int j = i+1; j < dets.size(); ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }

    for(int i = 0; i < dets.size(); i++){
        printf("dets[%i]: classes: %i sort_class: %i objectness: %f\n", 
            i, dets[i].classes, dets[i].sort_class, dets[i].objectness);
    }

}

void draw_detections(cv::Mat im, vector<tools::detection>& dets){
    
    for(int i = 0; i < dets.size(); i++){
        if(dets[i].objectness){
            box b = dets[i].bbox;
            int classID = -1;
            for(int j = 0; j < 80; ++j){
                if (dets[i].prob[j] > 0.5){
                    if (classID < 0) {
                        classID = j;
                    }
                    printf("%i: %.0f%%\n", classID, dets[i].prob[j]*100);
                }
            }
            if (classID >= 0){
                int left  = (b.x-b.w/2.)*im.size().width;
                int right = (b.x+b.w/2.)*im.size().width;
                int top   = (b.y-b.h/2.)*im.size().height;
                int bot   = (b.y+b.h/2.)*im.size().height;

                if(left < 0) left = 0;
                if(right > im.size().width-1) right = im.size().width-1;
                if(top < 0) top = 0;
                if(bot > im.size().height-1) bot = im.size().height-1;

                // printf("Box[%i] [%i, %i] -> [%i, %i]\n", i, left, top, right, bot);
                cv::Point2d p1 (left, top);
                cv::Point2d p2 (right, bot);
                cv::rectangle(im, p1, p2, cv::Scalar(255, 0, 0), 2);

            }
        }
    }
    cv::cvtColor(im, im, cv::COLOR_RGB2BGR);
    cv::imshow("output", im);
    cv::waitKey(0);


}

}