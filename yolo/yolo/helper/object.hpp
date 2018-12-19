#pragma once
#include <vector>
#include <iostream>

using namespace std;

namespace helper{

namespace object{

class Box {
public:
    float x, y, w, h;
    float left; 
    float right;
    float top;
    float bot;
    // Size = C + 1
    // Class + Max_prob
    vector<float> prob;

    Box(float x, float y, float w, float h)
        : x(x), y(y), w(w), h(h) {
        int imw = 608;
        int imh = 608;

        left  = x - w / 2.;
        right = x + w / 2.;
        top   = y - h / 2.;
        bot   = y + h / 2.;
    
        // if (left < 0) left = 0;
        // if (right > imw - 1) right = imw - 1;
        // if (top < 0) top = 0;
        // if (bot > imh - 1) bot = imh - 1;
    }

    friend ostream& operator<<(ostream& out, const Box& self){
        out << "[x: " << self.x << ",y: " << self.y << "]" << endl
        << "[w: " << self.w << ",h: " << self.h << "]" << endl
        << self.left << ":" << self.right << ":" << self.top << ":" << self.bot << endl;

        for(int i = 0; i < self.prob.size(); i ++){
            out << i << " : " << self.prob.at(i) << endl;
        }
        return out;
    }
};

}
}