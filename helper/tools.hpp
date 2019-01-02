#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "object.hpp"

using namespace InferenceEngine;
using namespace std;

namespace tools{

std::vector<DetectedObject> ParseYOLOV1Output(float *net_out, int class_num) {
    float threshold = 0.2f;         // The confidence threshold
    int C = 20;                     // classes
    int B = 2;                      // bounding boxes
    int S = 7;                      // cell size

    std::vector<DetectedObject> boxes;
    std::vector<DetectedObject> boxes_result;
    int SS = S * S;                 // number of grid cells 7*7 = 49
    // First 980 values correspons to probabilities for each of the 20 classes for each grid cell.
    // These probabilities are conditioned on objects being present in each grid cell.
    int prob_size = SS * C;         // class probabilities 49 * 20 = 980
    // The next 98 values are confidence scores for 2 bounding boxes predicted by each grid cells.
    int conf_size = SS * B;         // 49*2 = 98 confidences for each grid cell

    float *probs = &net_out[0];
    float *confs = &net_out[prob_size];
    float *cords = &net_out[prob_size + conf_size];     // 98*4 = 392 coords x, y, w, h

    for (int grid = 0; grid < SS; grid++) {
        int row = grid / S;
        int col = grid % S;
        for (int b = 0; b < B; b++) {
            int index = grid * B + b;
            int p_index = SS * C + grid * B + b;
            float scale = net_out[p_index];
            int box_index = SS * (C + B) + (grid * B + b) * 4;
            int objectType = class_num;

            float conf = confs[(grid * B + b)];
            float xc = (cords[(grid * B + b) * 4 + 0] + col) / S;
            float yc = (cords[(grid * B + b) * 4 + 1] + row) / S;
            float w = pow(cords[(grid * B + b) * 4 + 2], 2);
            float h = pow(cords[(grid * B + b) * 4 + 3], 2);
            int class_index = grid * C;
            float prob = probs[grid * C + class_num] * conf;

            DetectedObject bx(objectType, xc - w / 2, yc - h / 2, xc + w / 2,
                    yc + h / 2, prob);

            if (prob >= threshold) {
                boxes.push_back(bx);
            }
        }
    }

    // Sorting the higher probablities to the top
    sort(boxes.begin(), boxes.end(),
            [](const DetectedObject & a, const DetectedObject & b) -> bool {
                return a.prob > b.prob;
            });

    // Filtering out overlaping boxes
    std::vector<bool> overlapped(boxes.size(), false);
    for (int i = 0; i < boxes.size(); i++) {
        if (overlapped[i])
            continue;

        DetectedObject box_i = boxes[i];
        for (int j = i + 1; j < boxes.size(); j++) {
            DetectedObject box_j = boxes[j];
            if (DetectedObject::ioU(box_i, box_j) >= 0.4) {
                overlapped[j] = true;
            }
        }
    }

    // for (int i = 0; i < boxes.size(); i++) {
    //     if (boxes[i].prob > 0.0f) {
    //         boxes_result.push_back(boxes[i]);
    //     }
    // }
	if(!boxes.empty()){
		boxes_result.push_back(boxes[0]);
	}
    return boxes_result;
}


void ReadDataNames(const string& ClassesFilePath, std::vector<string>& classes){
    classes.clear();
    ifstream ifs(ClassesFilePath.c_str());
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + ClassesFilePath + " not found");
    string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }

}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

double IntersectionOverUnion(const helper::object::DetectionObject &box_1, 
    const helper::object::DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void embed_image(const cv::Mat& source, cv::Mat& dest, int dx, int dy)
{
    int imh = source.size().height;
    int imw = source.size().width;
    for(int row = 0; row < imh; row ++){
        for(int col = 0; col < imw; col ++){
            for(int ch = 0; ch < 3; ch ++){
                dest.at<cv::Vec3f>(dy + row, dx + col)[ch] = 
                    source.at<cv::Vec3f>(row, col)[ch];
            }
        }
    }
}

cv::Mat ReadImage(const std::string& imageName, int IH, int IW, int* srcw, int* srch, float* rate, int* dx, int* dy){
    cv::Mat image = cv::imread(imageName);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1.0/255.0, 0);
    *srcw = image.size().width;
    *srch = image.size().height;
    int imw = image.size().width;
    int imh = image.size().height;
    float resize_ratio = (float)IH / (float)max(imw, imh);
    *rate = resize_ratio;
    cv::resize(image, image, cv::Size(IH, IW));
    return image;
}


void ParseYOLOV1Output(const Blob::Ptr &blob,
                       const CNNLayerPtr &layer,
                       const unsigned long resized_im_h,
                       const unsigned long resized_im_w, 
                       const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, 
                       std::vector<helper::object::DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    if (layer->type != "RegionYolo")
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    // --------------------------- Extracting layer parameters -------------------------------------
    const int num = layer->GetParamAsInt("num"); // num of bounding box
    const int coords = layer->GetParamAsInt("coords");
    const int classes = layer->GetParamAsInt("classes");
    const int out_blob_h = blob->dims()[0];
    const int out_blob_w = blob->dims()[1];
    const int SS = blob->dims()[0] / (num * 5 + classes);
    const int S = sqrt(SS);
    int prob_size = SS * classes;   // class probabilities 49 * 20 = 980
    int conf_size = SS * num;       // 49*2 = 98 confidences for each grid cell
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    for (int grid = 0; grid < SS; grid++) {
        int row = grid / S;
        int col = grid % S;
        for (int b = 0; b < num; b++) {
            int index = grid * num + b;
            int p_index = SS * classes + grid * num + b;
            float scale = output_blob[p_index];
            int box_index = SS * (classes + num) + (grid * num + b) * 4;
            float x = (output_blob[prob_size + conf_size + (grid * num + b) * 4 + 0] + col) / S;
            float y = (output_blob[prob_size + conf_size + (grid * num + b) * 4 + 1] + row) / S;
            float w = pow(output_blob[prob_size + conf_size + (grid * num + b) * 4 + 2], 2);
            float h = pow(output_blob[prob_size + conf_size + (grid * num + b) * 4 + 3], 2);

            for (int j = 0; j < classes; ++j) {
                int class_index = grid * classes + j;
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                helper::object::DetectionObject obj(x, y, h, w, j, prob,
                        static_cast<float>(resized_im_h),
                        static_cast<float>(resized_im_w));
                objects.push_back(obj);
                // cout << "scale: " << scale 
                //     << " x: " << x << " y: " << y 
                //     << " w: " << w << " h: " << h
                //     << obj
                //     << endl;
            }

        }
    }


}
} // namespace tools