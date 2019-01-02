#pragma once
#include <vector>
#include <iostream>

using namespace std;

namespace helper{
namespace object{

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);

        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }

    friend ostream& operator<<(ostream& out, const DetectionObject& obj){
        out << "[" << obj.xmin << ", " << obj.ymin << "] [" 
            << obj.xmax << ", " << obj.ymax << "] id: " << obj.class_id << " confi: " << obj.confidence;
        return out;
    }
};

}
}