/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_yolov1/main.cpp
* \example object_detection_demo_yolov1/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <inference_engine.hpp>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "object_detection_demo_yolov1.h"

#include <ext_list.hpp>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace InferenceEngine;
ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    return true;
}

void FrameToBlob(const cv::Mat &frame, InferRequest &inferRequest, const std::string &inputName) {
    /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
    inferRequest.SetBlob(inputName, wrapMat2Blob(frame));
}

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
        this->ymin = static_cast<int>((y + h / 2) * h_scale);
        // this->xmax = static_cast<int>(this->xmin + w * w_scale);
        // this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->xmax = static_cast<int>((x - w / 2) * w_scale);
        this->ymax = static_cast<int>((y + h / 2) * h_scale);

        // if(this->xmin < 0) this->xmin = 0;
        // if(this->xmax > w_scale-1) this->xmax = w_scale-1;
        // if(this->ymin < 0) this->ymin = 0;
        // if(this->ymax > h_scale-1) this->ymax = h_scale-1;

        this->class_id = class_id;
        this->confidence = confidence;
        // printf("[%f, %f] -- [%f, %f] -- id: %i\n", x, y, w, h, class_id);
    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
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

    for(int i = 0; i < 100; i ++){
        // if(output_blob[i] < 0)
        printf("%i %f\n", i,  output_blob[i]);
    }
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
            
            float x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            float y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            // float height = std::exp(output_blob[box_index + 3 * side_square]);
            // float width  = std::exp(output_blob[box_index + 2 * side_square]);
            // float height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1];
            // float width  = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n];
            float height = std::exp(output_blob[box_index + 3 * side_square]) / side * anchors[2 * n + 1];
            float width  = std::exp(output_blob[box_index + 2 * side_square]) / side * anchors[2 * n];
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

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty()) throw std::logic_error("No suitable images were found");

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64" , "" }).getPluginByDevice(FLAGS_d);
        if (FLAGS_p_msg) {
            static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
        }

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Setting plugin parameter for collecting per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }

        /** Printing plugin version **/
        printPluginVersion(plugin, std::cout);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        slog::info << "Loading network files:"
                "\n\t" << FLAGS_m <<
                "\n\t" << binFileName <<
        slog::endl;

        CNNNetReader networkReader;
        /** Reading network model **/
        networkReader.ReadNetwork(FLAGS_m);

        /** Extracting model name and loading weights **/
        networkReader.ReadWeights(binFileName);
        CNNNetwork network = networkReader.getNetwork();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo = network.getInputsInfo();
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        auto inputInfoItem = *inputInfo.begin();
        auto inputName = inputInfo.begin()->first;

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the plugin **/
        // inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setPrecision(Precision::FP32);
        // inputInfoItem.second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        // inputInfoItem.second->getInputData()->setLayout(Layout::NHWC);
        inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
        cv::Mat image = cv::imread(imageNames.at(0));
        // image.convertTo(image, CV_32F, 1.0/255.0, 0);
        const size_t width  = (size_t) image.size().width;
        const size_t height = (size_t) image.size().height;
        cv::imshow("Image", image);
        cv::waitKey(0);
        cv::resize(image, image, cv::Size(608, 608));
        // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        /** Setting batch size using image count **/
        network.setBatchSize(1);
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // ------------------------------ Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

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
        slog::info << "Loading model to the plugin" << slog::endl;

        ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        // FrameToBlob(image, infer_request, inputName);
        for (const auto & item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);
            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = input->getTensorDesc().getDims()[1];
            size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];
            // auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
            auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            /** Iterate over all input images **/
            for(int k = 0; k < 3; k++){
                for(int j = 0; j < 416; j++){
                    for(int i = 0; i < 416; i++){
                        // int dst_index = i + 608*j + 608*608*k;
                        int dst_index = i*3 + j*416*3 + k;
                        data[dst_index] = image.at<cv::Vec3b>(j, i)[k]/255.0;
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
        /** Start inference & calc performance **/
        for (int iter = 0; iter < FLAGS_ni; ++iter) {
            auto t0 = Time::now();
            infer_request.Infer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;
        unsigned long resized_im_h = 608;
        unsigned long resized_im_w = 608;
        std::vector<DetectionObject> objects;
        // Parsing outputs
        for (auto &output : outputInfo) {
            auto output_name = output.first;
            Blob::Ptr blob = infer_request.GetBlob(output_name);
            ParseYOLOV2Output(blob, resized_im_h, resized_im_w, resized_im_h, resized_im_w, 0.5, objects);
        }

        cout << "objects size: " << objects.size() << endl;

        // Drawing boxes
        for (auto &object : objects) {
            if (object.confidence < 0.5)
                continue;
            auto label = object.class_id;
            float confidence = object.confidence;
            if (confidence > 0.5) {
                // std::cout << "[" << label << "] element, prob = " << confidence <<
                //             "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                //             << ((confidence > 0.5) ? " WILL BE RENDERED!" : "") << std::endl;
                /** Drawing only objects when >confidence_threshold probability **/
                std::ostringstream conf;
                conf << ":" << std::fixed << std::setprecision(3) << confidence;
                cv::rectangle(image, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(0, 0, 255));
            }
        }
        cv::imshow("Detection results", image);
        cv::waitKey(0);

    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
