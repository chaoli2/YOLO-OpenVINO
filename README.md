# Status

- YOLO V1 :smile:
- YOLO V2 :smile:
- YOLO V3 :smile:

# Env

- OpenVINO R4

# Darkflow to protobuf(.pb)

convert `YOLOv1` and `YOLOv2` `.cfg` and `.weights` to `.pb`.

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

## 1. Convert pb to IR

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

## 2. Build&Run OpenVINO

1. mkdir build && cd build

2. cmake .. && make

3. `./yolov1 -m <IR.XML_PATH> -w <IR.BIN_PATH> -image <IMG_PATH>`

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

3. `./yolov2 -m <IR.XML_PATH> -w <IR.BIN_PATH> -image <IMG_PATH>`

---

# YOLO V3

## 1. Convert pb to IR

1. Using [`v3ConvertTool`](./v3ConvertTool/) Dump YOLOv3 TenorFlow* Model.

  ```bash
    python3 dump.py                        \
      --class_names ../common/coco.names   \
      --weights_file <yolov3.weights_paht> \
      --size <302 or 416 or 608>
  ```

2. Convert `.pb` to IR

	```bash
    mo_tf.py
      --input_model /path/to/yolo_v3.pb
      --output_dir <OUTPUT_PATH>
      --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json
      --batch 1
	```

## 2. Build&Run OpenVINO

1. mkdir build && cd build

2. cmake .. && make

3. `./yolov3 -m <IR.XML_PATH> -w <IR.BIN_PATH> -image <IMG_PATH>`