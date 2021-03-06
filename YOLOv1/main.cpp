#include <inference_engine.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <gflags/gflags.h>

#include <ext_list.hpp>
#include <samples/common.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <tools.hpp>
#include <object.hpp>
#include <flags.hpp>

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

    vector<string> DataNames;
    string DataNamesPath = "../../common/voc.names";
    tools::ReadDataNames(DataNamesPath, DataNames);
    
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
                    int dst_index = ch*IW*IH + row*IW + col;
                    data[dst_index] = image.at<cv::Vec3f>(row, col)[ch];
                }
            }
        }
    }

    inputInfo = {};
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Do inference ---------------------------------------------------------
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    typedef std::chrono::duration<float> fsec;
    double total = 0.0;

    auto t0 = Time::now();
    infer_request.Infer();
    auto t1 = Time::now();

    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);
    total += d.count();
    std::cout << std::endl << "total inference time: " << total << " ms" << std::endl;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Process output -------------------------------------------------------
    std::vector<helper::object::DetectionObject> objects;
    // Parsing outputs
    for (auto &output : outputInfo) {
        auto output_name = output.first;
        CNNLayerPtr layer = network_reader.getNetwork().getLayerByName(output_name.c_str());
        Blob::Ptr blob = infer_request.GetBlob(output_name);
        tools::ParseYOLOV1Output(blob, layer, IH, IW, 0.2, objects);
    }
    // Filtering overlapping boxes
    std::sort(objects.begin(), objects.end());
    for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].confidence == 0)
            continue;
        for (int j = i + 1; j < objects.size(); ++j)
            if (tools::IntersectionOverUnion(objects[i], objects[j]) >= 0.45)
                objects[j].confidence = 0;
    }

    // Drawing boxes
    for (auto &object : objects) {
        if (object.confidence < 0.2)
            continue;
        auto label = object.class_id;
        float confidence = object.confidence;
        if (confidence > 0.2) {
            std::cout << "[" << std::right << setw(3) << label << "]: " 
                      << std::left << setw(10) << DataNames.at(label) 
                      << " \tprob = " << setprecision(4) << confidence*100 << "\% \t(" 
                      << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                      << std::endl;
            /** Drawing only objects when >confidence_threshold probability **/
            std::ostringstream conf;
            conf << ":" << std::fixed << std::setprecision(3) << confidence;
            cv::Point2f p1 = cv::Point2f(object.xmin, object.ymin);
            cv::Point2f p2 = cv::Point2f(object.xmax, object.ymax);
            cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255));
            cv::putText(image, DataNames.at(label), p1, cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 0, 0), 0.2);
        }
    }
    cv::resize(image, image, cv::Size(srcw, srch));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::imshow("Detection results", image);
    cv::waitKey(0);

}
