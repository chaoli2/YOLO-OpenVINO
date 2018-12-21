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
    for(int i =0; i < IW*IH; i++){
        int row = i / IW;
        int col = i % IW;
        for(int n = 0; n < 5; n++){
            int index = n*IW*IH + i;
            int obj_index  = entry_index(IH, IW, n*IW*IH + i, 4);
            int box_index  = entry_index(IH, IW, n*IW*IH + i, 0);
            // cout << "obj_index: " << obj_index
            // << "\tbox_index: " << box_index << endl;
            float scale = output_data[obj_index];
            dets[index].bbox = get_region_box(output_data, n, box_index, col, row, IW, IH, IW*IH);
            dets[index].objectness = scale > thresh ? scale : 0;

            int class_index = entry_index(IH, IW, n*IW*IH + i, 5);
            if(dets[index].objectness){
                for(int j = 0; j < 80; ++j){
                    int class_index = entry_index(IH, IW, n*IW*IH + i, 4 + 1 + j);
                    float prob = scale*output_data[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
                printf("scale:%f [%f, %f]-[%f, %f] score:%f obj_idx:%i box_idx:%i\n", 
                    scale, 
                    dets[index].bbox.x, dets[index].bbox.y,
                    dets[index].bbox.w, dets[index].bbox.h,
                    dets[index].objectness,
                    obj_index,
                    box_index);
            }
        }

    }
    correct_region_boxes(dets, IW*IH*5, 1, 1, IW, IH, 0);

    return dets;
}

}