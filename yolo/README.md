# Deeplab-OpenVINO

<img src=./pics/example.png width=500>

## 1. Model Optimize

1. Download [Deeplab](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) tf model.

2. Converting Deeplab Model to Intermediate Representation (IR)


	- Setup enviroment

    	Use the script bin/setupvars.sh to set the environment variables

    	`source <INSTALL_PATH>/bin/setupvars.sh`

	Use the mo.py script from the <INSTALL_DIR>/deployment_tools/model_optimizer directory to run the Model Optimizer and convert the model to the Intermediate Representation (IR).

	`python3 mo.py --input_model <frozen_inference_graph.pb MODEL_PATH> --output ArgMax --input 1:mul_1 --input_shape "(1,513,513,3)" --output_dir <OUTPUT_DIR>`

```bash
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	<frozen_inference_graph.pb MODLE_PATH>
	- Path for generated IR: 	<OUTPUT_DIR>
	- IR output name: 	frozen_inference_graph
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	1:mul_1
	- Output layers: 	ArgMax
	- Input shapes: 	(1,513,513,3)
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP32
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Offload unsupported operations: 	False
	- Path to model dump for TensorBoard: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	None
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	None
Model Optimizer version: 	1.2.185.5335e231
/home/sfy/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/mo/front/common/partial_infer/slice.py:90: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  value = value[slice_idx]

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: <OUPUT_DIR>/frozen_inference_graph.xml
[ SUCCESS ] BIN file: <OUPUT_DIR>/frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 8.65 seconds. 

```

*References*:

[*Model Optimizer Developer Guide*](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer)

## 2. Inferring Deeplab Model with the Inference Engine

1. Setup enviroment

    Use the script bin/setupvars.sh to set the environment variables

    `source <INSTALL_PATH>/bin/setupvars.sh`

2. Build

    - mkdir build
    - cd build
    - cmake ..
	- set `PLUGIN_DIR` in deeplabv3/main.cpp
    - make

3. run

    - cd deeplabv3

    - `./deeplabv3 --image <IMG_PATH> --m <.xml_PATH> --w <.bin_PATH>`

*References*:

[*Inference Engine Developer Guide*](https://software.intel.com/en-us/articles/OpenVINO-InferEngine)