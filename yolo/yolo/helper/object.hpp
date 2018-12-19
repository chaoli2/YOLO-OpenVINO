#pragma once
#include <vector>
#include <iostream>

using namespace std;

namespace helper{

namespace object{

class ObjectBox {
public:
    vector<float> ClassProb;
    float xmin, xmax, ymin, ymax, score;

    ObjectBox(float xmin, float ymin, float xmax, float ymax, float score)
        : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), score(score) {}

    friend ostream& operator<<(ostream& out, const ObjectBox& self){
        out << "xmin: " << self.xmin
        << " xmax: " << self.xmax 
        << " ymin: " << self.ymin 
        << " ymax: " << self.ymax 
        << " score: " << self.score;

        for(int i = 0; i <  self.ClassProb.size(); i ++){
            out << i << " - " << self.ClassProb.at(i) << endl;
        }
        
        return out;
    }
};

class Box {
public:
    float x, y, w, h;
    int left; 
    int right;
    int top;
    int bot;

    Box(float x, float y, float w, float h)
        : x(x), y(y), w(w), h(h) {
        int imw = 608;
        int imh = 608;
        
        int left = (b.x - b.w / 2.) * imw;
        int right = (b.x + b.w / 2.) * imw;
        int top = (b.y - b.h / 2.) * imh;
        int bot = (b.y + b.h / 2.) * imh;
    
        if (left < 0) left = 0;
        if (right > imw - 1) right = imw - 1;
        if (top < 0) top = 0;
        if (bot > imh - 1) bot = imh - 1;
    }

    friend ostream& operator<<(ostream& out, const Box& self){
        out << "[x: " << self.x << ",y: " << self.y << "]"
        << " [w: " << self.w << ",h: " << self.h << "]"
        << " l: " << self.left << " r: " << self.right << " t: " << self.top << " b: " << self.bot;
        return out;
    }
};

/**
 * @brief This class represents an object that is found by an object detection net
 */
class DetectedObject {
public:
    int objectType;
    vector<helper::object::ObjectBox> Boxes;

    DetectedObject(){

    }

    friend ostream& operator<<(ostream& out, const DetectedObject& self){
        for(int i = 0; i < self.Boxes.size(); i++){
            out << "Box[" << i << "] : " << self.Boxes[i];
        }

        return out;
    }

};
}
}