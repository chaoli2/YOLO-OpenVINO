#pragma once
#include <vector>
#include <iostream>

using namespace std;

namespace helper{

namespace object{

/**
 * @brief This class represents an object that is found by an object detection net
 */
class DetectedObject {
public:
    int objectType;
    float xmin, xmax, ymin, ymax, score;
    vector<float> ClassProb;

    DetectedObject(float xmin, float ymin, float xmax, float ymax, float score)
        : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), score(score) {
    }

    friend ostream& operator<<(ostream& out, const DetectedObject& self){
        out << "xmin: " << self.xmin
        << " xmax: " << self.xmax 
        << " ymin: " << self.ymin 
        << " ymax: " << self.ymax << endl;

        for(int i = 0; i <  self.ClassProb.size(); i ++){
            out << i << " - " << self.ClassProb.at(i) << endl;
        }

        return out;
    }

};
}
}