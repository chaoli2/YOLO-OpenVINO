#include <inference_engine.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <gflags/gflags.h>

#include <ext_list.hpp>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <opencv_wraper.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "helper/tools.hpp"
#include "helper/flags.hpp"

using namespace InferenceEngine;
using namespace std;

#define PLUGIN_DIR "/opt/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64"


static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

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
        // printf("[%f, %f] -- [%f, %f] -- id: %i confi: %f\n", x, y, w, h, class_id, confidence);
    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
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

void ParseYOLOV2Output(const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    const int out_blob_h = resized_im_h / 32;
    const int out_blob_w = resized_im_w / 32;
    // --------------------------- Extracting layer parameters -------------------------------------
    // auto num = layer->GetParamAsInt("num");
    auto num = 5;
    auto coords = 4;
    auto classes = 80;
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

    // for(int i = 0; i < 100; i ++){
    //     printf("%i - %f\n", i, output_blob[i]);
    // }

    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            // printf("obj_index: %i \tbox_index: %i \tscale: %f\n", obj_index, box_index, scale);
            if (scale < threshold)
                continue;
            
            float x = (col + output_blob[box_index + 0 * side_square]) / side * original_im_w;
            float y = (row + output_blob[box_index + 1 * side_square]) / side * original_im_h;
            float height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1] / side * original_im_h;
            float width  = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n] / side * original_im_w;
            // cout << "[" << x << ", " << y << "] - [" << width << ", " << height << "]" << endl;
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void embed_image(const cv::Mat& source, cv::Mat& dest, int dx, int dy)
{
    printf("dx: %i dy: %i\n", dx, dy);
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


int main(int argc, char* argv[]){
    gflags::RegisterFlagValidator(&helper::FLAGS_image, helper::ValidateName);
    gflags::RegisterFlagValidator(&helper::FLAGS_m, helper::Validate_m);
    gflags::RegisterFlagValidator(&helper::FLAGS_w, helper::Validate_w);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto version = GetInferenceEngineVersion();
    cout << "InferenceEngine Version: " << version->apiVersion.major << "." << version->apiVersion.minor << endl;
    cout << "build: " << version->buildNumber << endl;

    // 1. Load a Plugin
    vector<string> pluginDirs {PLUGIN_DIR};
    InferenceEnginePluginPtr engine_ptr = PluginDispatcher(pluginDirs).getSuitablePlugin(TargetDevice::eCPU);
    InferencePlugin plugin(engine_ptr);
    cout << "Plugin Version: " << plugin.GetVersion()->apiVersion.major << "." << plugin.GetVersion()->apiVersion.minor << endl;
    cout << "build: " << plugin.GetVersion()->buildNumber << endl;
    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

    // 2. Read the Model Intermediate Representation (IR)
    CNNNetReader network_reader;
    network_reader.ReadNetwork(helper::FLAGS_m);
    network_reader.ReadWeights(helper::FLAGS_w);

    // 3. Configure Input and Output
    CNNNetwork network = network_reader.getNetwork();

    /** Taking information about all topology inputs **/
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    /** Taking information about a`ll topology outputs **/
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());


    auto inputInfoItem = *inputInfo.begin();
    auto inputName = inputInfo.begin()->first;
    int IC = inputInfoItem.second->getTensorDesc().getDims()[1];
    int IH = inputInfoItem.second->getTensorDesc().getDims()[2];
    int IW = inputInfoItem.second->getTensorDesc().getDims()[3];
    inputInfoItem.second->setPrecision(Precision::FP32);
    // inputInfoItem.second->getInputData()->setLayout(Layout::NHWC);
    // inputInfoItem.second->setPrecision(Precision::U8);
    inputInfoItem.second->setLayout(Layout::NCHW);

    float rate = 0;
    int dx = 0;
    int dy = 0;
    int srcw = 0;
    int srch = 0;
    cv::Mat image = ReadImage(helper::FLAGS_image, IH, IW, &srcw, &srch, &rate, &dx, &dy);
    /** Setting batch size using image count **/
    network.setBatchSize(1);
    size_t batchSize = network.getBatchSize();
    // ------------------------------ Prepare output blobs -------------------------------------------------
    OutputsDataMap outputInfo(network.getOutputsInfo());
    // BlobMap outputBlobs;
    std::string firstOutputName;

    for (auto & item : outputInfo) {
        if (firstOutputName.empty()) {
            firstOutputName = item.first;
        }
        DataPtr outputData = item.second;
        if (!outputData) {
            throw std::logic_error("output data pointer is not valid");
        }

        item.second->setPrecision(Precision::FP32);
    }
    // --------------------------- 4. Loading model to the plugin ------------------------------------------
    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Create infer request -------------------------------------------------
    InferRequest infer_request = executable_network.CreateInferRequest();
    // -----------------------------------------------------------------------------------------------------
    // --------------------------- 6. Prepare input --------------------------------------------------------
    /** Iterate over all the input blobs **/

    for (const auto & item : inputInfo) {
        /** Creating input blob **/
        Blob::Ptr input = infer_request.GetBlob(item.first);
        /** Filling input tensor with images. First b channel, then g and r channels **/
        size_t num_channels = input->getTensorDesc().getDims()[1];
        size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];
        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        for(int row = 0; row < IH; row ++){
            for(int col = 0; col < IW; col ++){
                for(int ch = 0; ch < IC; ch ++){
                    // int dst_index = col*IC + row*IW*IC + ch;
                    int dst_index = ch*IW*IH + row*IW + col;
                    data[dst_index] = image.at<cv::Vec3f>(row, col)[ch];
                }
            }
        }
    }
    inputInfo = {};
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Do inference ---------------------------------------------------------
    infer_request.Infer();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Process output -------------------------------------------------------
    std::vector<DetectionObject> objects;
    // Parsing outputs
    for (auto &output : outputInfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request.GetBlob(output_name);
        ParseYOLOV2Output(blob, IH, IW, IH, IW, 0.5, objects);
    }

    // Filtering overlapping boxes
    std::sort(objects.begin(), objects.end());
    for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].confidence == 0)
            continue;
        for (int j = i + 1; j < objects.size(); ++j)
            if (IntersectionOverUnion(objects[i], objects[j]) >= 0.45)
                objects[j].confidence = 0;
    }

    // Drawing boxes
    for (auto &object : objects) {
        if (object.confidence < 0.5)
            continue;
        auto label = object.class_id;
        float confidence = object.confidence;
        if (confidence > 0.5) {
            std::cout << "[" << label << "] element, prob = " << confidence <<
                        "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                        << ((confidence > 0.5) ? " WILL BE RENDERED!" : "") << std::endl;
            /** Drawing only objects when >confidence_threshold probability **/
            std::ostringstream conf;
            conf << ":" << std::fixed << std::setprecision(3) << confidence;
            cv::rectangle(image, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(0, 0, 255));
        }
    }
    cv::Rect ROI(dx, dy, srcw*rate, srch*rate);
    cv::Mat croppedImage = image(ROI);
    cv::resize(croppedImage, croppedImage, cv::Size(srcw, srch));
    cv::cvtColor(croppedImage, croppedImage, cv::COLOR_BGR2RGB);
    cv::imshow("Detection results", croppedImage);
    cv::waitKey(0);


}
