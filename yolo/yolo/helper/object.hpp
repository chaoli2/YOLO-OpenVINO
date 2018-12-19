#pragma once
#include <vector>
#include <iostream>

using namespace std;

namespace helper{

namespace object{

class Box {
public:
    float x, y, w, h;
    int left; 
    int right;
    int top;
    int bot;
    // Size = C + 1
    // Class + Max_prob
    vector<float> prob;

    int classIdx;

    Box(float x, float y, float w, float h)
        : x(x), y(y), w(w), h(h) {
        int imw = 608;
        int imh = 404;
        classIdx = -1;

        left  = (x - w / 2.) * imw;
        right = (x + w / 2.) * imw;
        top   = (y - h / 2.) * imh;
        bot   = (y + h / 2.) * imh;
    
        if (left < 0) left = 0;
        if (right > imw - 1) right = imw - 1;
        if (top < 0) top = 0;
        if (bot > imh - 1) bot = imh - 1;
    }

    int max_index(const vector<float>& prob, int n) {
        if (n <= 0) return -1;
        int i, max_i = 0;
        float max = prob.at(0);
        for (i = 1; i < n; ++i) {
            if (prob.at(i) > max) {
                max = prob.at(i);
                max_i = i;
            }
        }
        return max_i;
    }

    friend ostream& operator<<(ostream& out, const Box& self){
        out << "coor: (" << self.left << ", " << self.top << ") (" << self.right << "," << self.bot << ")"
        << " Class Id: " << self.classIdx << " Prob: " << self.prob.at(self.classIdx);
        return out;
    }

    bool operator < (const Box& A){
        return prob[classIdx] < A.prob[A.classIdx];
    }

    bool operator > (const Box& A){
        return prob[classIdx] > A.prob[A.classIdx];
    }
};

}
}