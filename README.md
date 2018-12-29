# Status

- **Only** Support YOLO V2 currently.

# Env

- OpenVINO R4

# Darkflow to protobuf(.pb)

convert `.cfg` and `.weights` to `.pb`.

1. git clone [darkflow](https://github.com/thtrieu/darkflow)

2. `python3 setup.py build_ext --inplace`

	```bash
	AssertionError: expect 44948596 bytes, found 44948600
	```

3. modify the line self.offset = 16 in the ./darkflow/utils/loader.py file and replace with self.offset = 20
	[[error solution ref](https://sites.google.com/view/tensorflow-example-java-api/complete-guide-to-train-yolo/convert-darknet-weights-to-pb-file)]


4. Copy `coco.names` in `darknet/data` to `labels.txt` in `darkflow`.

5. `flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb`

---

# YOLO V1

1. Create `yolo_v1.json`

```json
 [
   {
     "id": "TFYOLO",
     "match_kind": "general",
     "custom_attributes": {
       "classes": 20,
       "coords": 4,
       "num": 2,
       "do_softmax": 1
     }
   }
 ]
```

2. Convert `.pb` to IR

    *PS: Can **only** convert yolo-tiny version currently.*

```bash
./mo_tf.py
--input_model <path_to_model>/<model_name>.pb \
--batch 1 \
--tensorflow_use_custom_operations_config <yolo_v1.json PATH> \
--output_dir <IR_PATH>
```

---

# YOLO V2


## 1. Convert pb to IR

1. Create `yolo_v2.json`

```json
 [
   {
     "id": "TFYOLO",
     "match_kind": "general",
     "custom_attributes": {
       "classes": 80,
       "coords": 4,
       "num": 5,
       "do_softmax": 1
     }
   }
 ]
```

2. Convert `.pb` to IR

```bash
./mo_tf.py
--input_model <path_to_model>/<model_name>.pb \
--batch 1 \
--tensorflow_use_custom_operations_config <yolo_v2.json PATH> \
--output_dir <IR_PATH>
```

## 2. Build&Run OpenVINO

1. mkdir build && cd build

2. cmake .. && make

3. `./yolo -m <IR.XML_PATH> -w <IR.BIN_PATH> -image <IMG_PATH>`

---

TODO: YOLO V1

TODO: YOLO V3


## 1. Convert YOLOv3 TensorFlow Model to the IR

- generate `V3` IR

	```bash
	mo_tf.py
	--input_model /path/to/yolo_v3.pb
	--tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json
	--batch 1
	```
