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

    for(
        map<string, InputInfo::Ptr>::iterator it = input_info.begin(); 
        it != input_info.end();
        it ++){
        it->second->setPrecision(Precision::FP32);
        cout << "Input: " << it->first << endl
            << "\tPrecision: " << it->second->getPrecision() << endl;
        // it->second->setLayout(Layout::NHWC);
        cout << "\tDim: [ ";
        for(auto x: it->second->getDims()){
            cout << x << " ";
        }
        cout << "]" << endl;
    }

    for(
        map<std::string, DataPtr>::iterator it = output_info.begin();
        it != output_info.end();
        it ++){
        it->second->setPrecision(Precision::FP32);
        cout << "Output: " << it->first << endl
            << "\tPrecision: " << it->second->getPrecision() << endl;
        cout << "\tDim: [ ";
        for(auto x: it->second->dims){
            cout << x << " ";
        }
        cout << "]" << endl;
    }

    // 4. Load the Model
    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    // 5. Create Infer Request
    InferRequest infer_request = executable_network.CreateInferRequest();

    // 6. Prepare Input
    /** Collect images data ptrs **/
    cv::Mat image = cv::imread(helper::FLAGS_image);
    image.convertTo(image, CV_32F, 1.0/255.0, 0);
    cv::imshow("Image", image);
    cv::waitKey(0);

    string input_name = (*input_info.begin()).first;
    Blob::Ptr input = infer_request.GetBlob(input_name);
    size_t num_channels = input->getTensorDesc().getDims()[1];
    size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];

    /** Iterating over all input blobs **/
    cout << "Prepare Input: " << input_name << endl;
    /** Getting input blob **/
    auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    /** Setting batch size **/
    network.setBatchSize(1);

    /** Iterate over all input images **/
    /** Iterate over all pixel in image (r,g,b) **/
    for (size_t ch = 0; ch < num_channels; ch++) {
        /** Iterate over all channels **/
        for (size_t pid = 0; pid < image_size; pid++) {
            /** [images stride + channels stride + pixel id ] all in bytes **/
            // data[ch * image_size + pid] = imageData.get()[pid*num_channels + ch] / 255.0;
            data[ch * image_size + pid] = image.at<float>(ch * image_size + pid);
        }
    }

    // 7. Perform Inference
    infer_request.Infer();

    // 8. Process Output

    string output_name = (*output_info.begin()).first;
    cout << "Processing output blobs: " << output_name << endl;
    const Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    float* output_data = output_blob->buffer().as<float*>();

    for(int i = 0; i < 10; i++){
        cout << output_data[i] << " ";
    }
    cout << endl;
    // tools::yoloNetParseOutput(output_data);



}
