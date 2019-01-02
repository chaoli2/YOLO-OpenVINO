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

cv::Mat ReadImageV2(const std::string& imageName, int IH, int IW, int* srcw, int* srch, float* rate, int* dx, int* dy){
    cv::Mat image = cv::imread(imageName);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1.0/255.0, 0);
    *srcw = image.size().width;
    *srch = image.size().height;
    cv::Mat resizedImg (IH, IW, CV_32FC3);
    resizedImg = cv::Scalar(0.5, 0.5, 0.5);
    int imw = image.size().width;
    int imh = image.size().height;
    float resize_ratio = (float)IH / (float)max(imw, imh);
    *rate = resize_ratio;
    cv::resize(image, image, cv::Size(imw*resize_ratio, imh*resize_ratio));

    int new_w = imw;
    int new_h = imh;
    if (((float)IW/imw) < ((float)IH/imh)) {
        new_w = IW;
        new_h = (imh * IW)/imw;
    } else {
        new_h = IH;
        new_w = (imw * IW)/imh;
    }
    *dx = (IW-new_w)/2;
    *dy = (IH-new_h)/2;
    embed_image(image, resizedImg, (IW-new_w)/2, (IH-new_h)/2);
    return resizedImg;
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

void ParseYOLOV2Output(const Blob::Ptr &blob,
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
    const int num = layer->GetParamAsInt("num");
    const int coords = layer->GetParamAsInt("coords");
    const int classes = layer->GetParamAsInt("classes");
    const int out_blob_h = layer->input()->dims[0];
    const int out_blob_w = layer->input()->dims[1];

    std::vector<float> anchors = {
        0.572730, 0.677385, 
        1.874460, 2.062530, 
        3.338430, 5.474340, 
        7.882820, 3.527780, 
        9.770520, 9.168280
    };
    auto side = out_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            
            float x = (col + output_blob[box_index + 0 * side_square]) / side * original_im_w;
            float y = (row + output_blob[box_index + 1 * side_square]) / side * original_im_h;
            float height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1] / side * original_im_h;
            float width  = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n] / side * original_im_w;
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                helper::object::DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

} // namespace tools