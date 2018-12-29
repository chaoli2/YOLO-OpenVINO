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
#include "helper/object.hpp"
#include "helper/flags.hpp"

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
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    /** Taking information about a`ll topology outputs **/
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());


    auto inputInfoItem = *inputInfo.begin();
    auto inputName = inputInfo.begin()->first;

    // IC: network input channel
    // IH: network input height
    // IW: network input width
    int IC = inputInfoItem.second->getTensorDesc().getDims()[1];
    int IH = inputInfoItem.second->getTensorDesc().getDims()[2];
    int IW = inputInfoItem.second->getTensorDesc().getDims()[3];
    inputInfoItem.second->setPrecision(Precision::FP32);
    inputInfoItem.second->setLayout(Layout::NCHW);

    float rate = 0;
    int dx = 0;
    int dy = 0;
    int srcw = 0;
    int srch = 0;
    cv::Mat image = tools::ReadImage(helper::FLAGS_image, IH, IW, &srcw, &srch, &rate, &dx, &dy);
    
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

        // for(int i = 0; i < 100; i ++){
        //     cout << i << " - " <<  data[i] << endl;
        // }

    }

    inputInfo = {};
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Do inference ---------------------------------------------------------
    infer_request.Infer();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Process output -------------------------------------------------------
    std::vector<helper::object::DetectionObject> objects;
    // Parsing outputs
    for (auto &output : outputInfo) {
        auto output_name = output.first;
        CNNLayerPtr layer = network_reader.getNetwork().getLayerByName(output_name.c_str());
        Blob::Ptr blob = infer_request.GetBlob(output_name);
        tools::ParseYOLOV1Output(blob, layer, IH, IW, IH, IW, objects);
    }

}