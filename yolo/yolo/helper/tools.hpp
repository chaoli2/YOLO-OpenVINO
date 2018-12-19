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
#include "CPost.hpp"

using namespace std;

namespace tools{

helper::object::Box get_region_box(const float *net_out, int n, int index, int i, int j, int w, int h, int stride) {
    // TINY_YOLOV2_ANCHORS
    const float biases[] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};
    float box_x = (i + net_out[index + 0*stride]) / w;
    float box_y = (j + net_out[index + 1*stride]) / h;
    float box_w = exp(net_out[index + 2*stride]) * biases[2*n]   / w;
    float box_h = exp(net_out[index + 3*stride]) * biases[2*n+1] / h;
    helper::object::Box b (box_x, box_y, box_w, box_h);

    return b;
}

void correct_region_boxes(vector<helper::object::Box>& boxes, int n, int w, int h, int netw, int neth, int relative) {
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
        helper::object::Box b = boxes.at(i);
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
        boxes.at(i) = b;
    }
}

bool SortBox(helper::object::Box& A, helper::object::Box& B){
    return A > B;
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

    vector<helper::object::Box> Boxes;
    for(int i = 0; i < S * S; i ++ ){
        int row = i / S;
        int col = i % S;
        for(int n = 0; n < B; n ++){
            int index = n * S * S + i;
            int obj_index = entry_index(S, S, 4, C, B, 0, n * S * S + i, 4);
            int box_index = entry_index(S, S, 4, C, B, 0, n * S * S + i, 0);
            float scale = net_out[obj_index];
            helper::object::Box Box = tools::get_region_box(net_out, n, box_index, col, row, S, S, S * S);

            float max = 0;
            int maxIdx = -1;
            for(int j = 0; j < C; ++j){
                int class_index = entry_index(S, S, 4, C, B, 0, n * S * S + i, 5 + j);
                float prob = scale * net_out[class_index];
                // probs[index][j] = (prob > threshold) ? prob : 0;
                Box.prob.push_back(prob);
                if(prob > max){
                    max = prob;
                    maxIdx = j;
                } 
            }
            Box.classIdx = maxIdx;
            Box.prob.push_back(max);
            Boxes.push_back(Box);
        }
    }
    int w = 608;
    int h = 404;
    tools::correct_region_boxes(Boxes, S * S * B, w, h, w, h, 1);

    sort(Boxes.begin(), Boxes.end(), SortBox);

    // for(int n = 0; n < Boxes.size(); n++){
    for(int n = 0; n < 5; n++){
        cout << "Boxes[" << n << "]: " << Boxes[n] << endl;
    }

}

}