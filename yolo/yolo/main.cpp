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
#include "helper/CPost.hpp"

using namespace InferenceEngine;
using namespace std;

#define PLUGIN_DIR "/opt/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64"

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
    InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    /** Taking information about a`ll topology outputs **/
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

    input_info.begin()->second->setPrecision(Precision::FP32);
    // input_info.begin()->second->setLayout(Layout::NCHW);
    input_info.begin()->second->setLayout(Layout::NHWC);
    cout << "Input: " << input_info.begin()->first << endl
        << "\tPrecision: " << input_info.begin()->second->getPrecision() << endl;
    cout << "\tDim: [ ";
    for(auto x: input_info.begin()->second->getDims()){
        cout << x << " ";
    }
    cout << "]" << endl;

    cout << "Output: " << output_info.begin()->first << endl
        << "\tPrecision: " << output_info.begin()->second->getPrecision() 
        << "\tDim: [ ";
    for(auto x: output_info.begin()->second->dims){
        cout << x << " ";
    }
    cout << "]" << endl;

    // 4. Load the Model
    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    // 5. Create Infer Request
    InferRequest infer_request = executable_network.CreateInferRequest();

    // 6. Prepare Input
    /** Collect images data ptrs **/
    string input_name = (*input_info.begin()).first;
    Blob::Ptr input = infer_request.GetBlob(input_name);
    size_t IC = input->getTensorDesc().getDims()[1];
    size_t IH = input->getTensorDesc().getDims()[2];
    size_t IW = input->getTensorDesc().getDims()[3];
    cout << "IC: " + to_string(IC) 
        + " IH: " + to_string(IH) 
        + " IW: " + to_string(IW) << endl;
    size_t image_size = IH * IW;

    cv::Mat image = cv::imread(helper::FLAGS_image);
    cv::Mat resizedImg (image);
    cv::imshow("Image", image);
    cv::waitKey(0);
    int imw = image.size().width;
    int imh = image.size().height;
    // double resize_ratio = (double)IH / (double)max(imw, imh);
    // cv::resize(image, resizedImg, cv::Size(imw*resize_ratio, imh*resize_ratio));
    
    cv::resize(image, resizedImg, cv::Size(IH, IW));
    // cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);
    resizedImg.convertTo(resizedImg, CV_32F, 1.0/255.0, 0);

    /** Iterating over all input blobs **/
    cout << "Prepare Input: " << input_name << endl;
    /** Getting input blob **/
    float* data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    /** Setting batch size **/
    network.setBatchSize(1);

    /** Iterate over all input images **/
    /** Iterate over all pixel in image (r,g,b) **/
    for(int row = 0; row < IH; row ++){
        for(int col = 0; col < IW; col ++){
            for(int ch = 0; ch < IC; ch ++){
                // int dst_index = i + IW*j + IW*IH*k;
                int dst_index = col*IC + row*IW*IC + ch;
                assert(dst_index >= 0);
                assert(dst_index <= IH * IW *IC);
                data[dst_index] = resizedImg.at<cv::Vec3f>(row, col)[ch];                
            }
        }
    }

    // 7. Perform Inference
    infer_request.Infer();

    // 8. Process Output

    string output_name = (*output_info.begin()).first;
    cout << "Processing output blobs: " << output_name << endl;
    const Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    float* output_data = output_blob->buffer().as<float*>();

    // for(int i = 0; i < 500; i ++){
    //     // if(output_data[i] > 1)
    //     printf("%i %f\n", i,  output_data[i]);
    //     // printf("%i %f\n", i,  data[i]);
    // }
    cout << endl;

    int num = 5;
    int classes = 80;
    float thresh = 0.2;
    vector<tools::detection> dets = 
        tools::yoloNetParseOutput(output_data, IH/32, IW/32, thresh, num);
    vector<tools::detection> do_nms_sort_dets;
    tools::do_nms_sort(dets, (IH/32)*(IW/32)*num, classes, 0.45, do_nms_sort_dets);
    tools::draw_detections(resizedImg, do_nms_sort_dets);

    // for(int index = 0; index < dets.size(); index++){
    //     if(dets[index].objectness)
    //     printf("[%f, %f]-[%f, %f] score:%f \n", 
    //                 dets[index].bbox.x, dets[index].bbox.y,
    //                 dets[index].bbox.w, dets[index].bbox.h,
    //                 dets[index].objectness);
    // }

    // for(int i = 0; i < 10; i++){
    //     cout << output_data[i] << " ";
    // }
    // cout << endl;
    // vector<helper::object::Box> Boxes = tools::yoloNetParseOutput(output_data);

    // for(int i = 0; i < 5; i ++){
    //     cv::Point2d p1 (Boxes[i].left, Boxes[i].top);
    //     cv::Point2d p2 (Boxes[i].right, Boxes[i].bot);
    //     cv::rectangle(resizedImg, p1, p2, cv::Scalar(255, 255, 255));
    // }
    // cv::cvtColor(resizedImg, resizedImg, cv::COLOR_RGB2BGR);
    // cv::resize(resizedImg, resizedImg, cv::Size(imw, imh));
    // cv::imshow("output", resizedImg);
    // cv::waitKey(0);
}
