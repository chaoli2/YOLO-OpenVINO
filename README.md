# Status

- YOLO V1 :smile:
- YOLO V2 :smile:
- YOLO V2 :smile:

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

1. Dump YOLOv3 TenorFlow* Model

  - git clone https://github.com/mystic123/tensorflow-yolo-v3.git
  - git checkout fb9f543
  - Open demo.py file in a text editor and make the following changes:
    - Replace NCHW with NHWC on the line 57.
    - Insert the following lines after the line 64:
      ```python
      from tensorflow.python.framework import graph_io
      frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['concat_1'])
      graph_io.write_graph(frozen, './', 'yolo_v3.pb', as_text=False)
      ```
  - run the following command:
    ```bash
    python3 demo.py                                         \
    --weights_file <path_to_weights_file>/yolov3.weights    \
    --class_names <path_to_labels_file>/coco.names.txt      \
    --size <network_input_size>                             \
    --input_img <path_to_image>/<image>                     \
    --output_img ./out.jpg
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