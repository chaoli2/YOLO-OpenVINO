## Env

- OpenVINO R4


[YOLO V2](https://pjreddie.com/darknet/yolov2/)

## 1. Darkflow to protobuf(.pb)

convert `.cfg` and `.weights` to `.pb`.

1. install [darkflow](https://github.com/thtrieu/darkflow)



---

V3:



## 1. Convert YOLO* TensorFlow Model to the IR

- generate `V3` IR

```bash
mo_tf.py
--input_model /path/to/yolo_v3.pb
--tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json
--batch 1
```

- generate `V2` IR

yolo_v1_v2.json

```json
 [
   {
     "id": "TFYOLO",
     "match_kind": "general",
     "custom_attributes": {
       "classes": 80,
       "coords": 4,
       "num": 3,
       "do_softmax": 1
     }
   }
 ]
```

```bash
./mo_tf.py
--input_model <path_to_model>/<model_name>.pb       \
--batch 1                                       \
--tensorflow_use_custom_operations_config <yolo_v1_v2.json PATH>

Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/sfy/ws/ir/yolo/v1/yolov2.pb
	- Path for generated IR: 	/home/sfy/ws/ir/yolo/v1/
	- IR output name: 	yolov2
	- Log level: 	ERROR
	- Batch: 	1
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
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
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	None
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/yolo_v1_v2.json
Model Optimizer version: 	1.4.292.6ef7232d

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/sfy/ws/ir/yolo/v1/yolov2.xml
[ SUCCESS ] BIN file: /home/sfy/ws/ir/yolo/v1/yolov2.bin
[ SUCCESS ] Total execution time: 9.39 seconds. 

```


## 2. Inferring 

