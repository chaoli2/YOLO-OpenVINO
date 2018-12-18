#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "object.hpp"

using namespace std;

namespace tools{

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
            float xmin = obj.at(b*(C+5));
            float xmax = obj.at(b*(C+5) + 1);
            float ymin = obj.at(b*(C+5) + 2);
            float ymax = obj.at(b*(C+5) + 3);
            float score = obj.at(b*(C+5) + 4); //box confidence score
            helper::object::DetectedObject Box (xmin, xmax, ymin, ymax, score);
            for(int i = 0; i < C; i++){
                Box.ClassProb.push_back(obj.at(b*(C+5) + i));
            }
            DetectedObjects.push_back(Box);
        }
    }

    cout << DetectedObjects[0] << endl;


}

}