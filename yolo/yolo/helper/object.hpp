#pragma once
#include <vector>

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

    

};

}
}