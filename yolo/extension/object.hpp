#pragma once
#include <vector>
#include <iostream>

using namespace std;

namespace yolov2{

namespace object{

class Box {
public:
    vector<float> coords;
    vector<float> probs;

    int classIdx;

    friend ostream& operator<<(ostream& out, const Box& self){
        out << "self.coords.size(): " << self.coords.size() << " self.probs.size(): " << self.probs.size();
        return out;
    }

    bool operator < (const Box& A){
        return probs[classIdx] < A.probs[A.classIdx];
    }

    bool operator > (const Box& A){
        return probs[classIdx] > A.probs[A.classIdx];
    }
};

}
}