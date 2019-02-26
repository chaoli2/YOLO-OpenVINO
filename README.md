# Status

- YOLO V1 :smile:
- YOLO V2 :smile:
- YOLO V3 :smile:

# Env

- OpenVINO R4

# Darkflow to protobuf(.pb)

How to convert `YOLOv1` and `YOLOv2` `.cfg` and `.weights` to `.pb`.

1.Install [darkflow](https://github.com/thtrieu/darkflow)

    1.1 Install prerequsites
    ```bash
    pip3 install tensorflow opencv-python numpy
    ```

    1.2 Get darkflow and YOLO-OpenVINO
    ```bash
    git clone https://github.com/thtrieu/darkflow
    git clone https://github.com/chaoli2/YOLO-OpenVINO
    ```

    1.3 modify the line self.offset = 16 in the ./darkflow/utils/loader.py file and replace with self.offset = 20

    1.4 Install darkflow
    ```bash
    pip install .
    ```

2. Copy `voc.names` in `YOLO-OpenVINO/common` to `labels.txt` in `darkflow`.
```bash
cp YOLO-OpenVINO/common/voc.names darkflow/labels.txt
```

3. Get yolov2 weights and cfg 
```bash
cd darkflow
mkdir -p models
wget -c https://pjreddie.com/media/files/yolov2-voc.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-voc.cfg

```

4. Run convert script
```bash
flow --model models/yolov2-voc.cfg --load models/yolov2-voc.weights --savepb
```

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
