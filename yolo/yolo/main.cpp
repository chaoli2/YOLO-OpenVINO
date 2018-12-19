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
        // it->second->setPrecision(Precision::FP32);
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
    FormatReader::ReaderPtr reader(helper::FLAGS_image.c_str());
    if (reader.get() == nullptr) {
        cout << "Image: " << helper::FLAGS_image << " cannot be read!" << endl;
        return -1;
    }

    string input_name = (*input_info.begin()).first;
    Blob::Ptr input = infer_request.GetBlob(input_name);
    size_t num_channels = input->getTensorDesc().getDims()[1];
    size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];


    /** Iterating over all input blobs **/
    cout << "Prepare Input: " << input_name << endl;
    /** Getting input blob **/
    auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    /** Getting image data **/
    std::shared_ptr<unsigned char> imageData(reader->getData(input->getTensorDesc().getDims()[3],
                                                             input->getTensorDesc().getDims()[2]));
    /** Setting batch size **/
    network.setBatchSize(1);

    /** Iterate over all input images **/
    /** Iterate over all pixel in image (r,g,b) **/
    for (size_t ch = 0; ch < num_channels; ch++) {
        /** Iterate over all channels **/
        for (size_t pid = 0; pid < image_size; pid++) {
            /** [images stride + channels stride + pixel id ] all in bytes **/
            data[ch * image_size + pid] = imageData.get()[pid*num_channels + ch] / 255.0;
        }
    }

    // 7. Perform Inference
    infer_request.Infer();

    // 8. Process Output

    string output_name = (*output_info.begin()).first;
    cout << "Processing output blobs: " << output_name << endl;
    const Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    float* output_data = output_blob->buffer().as<float*>();
    tools::yoloNetParseOutput(output_data);



}
