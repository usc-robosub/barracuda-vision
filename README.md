# barracuda-vision

## Running with Docker

### Step 1: Prepare the Model

Place your YOLO model file (e.g., `yolov8n.pt`) in a directory on your host, such as `/localmodels`. Only one model file should be present in this directory.

### Step 2: Build and Run the Docker Container

Use the provided Dockerfile and docker-compose.yaml to build and run the Docker container. Mount your model directory using the `MODEL_PATH` environment variable (defaults to `/localmodels`):

```bash
MODEL_PATH=/localmodels docker-compose up --build
```

### Step 3: Publish an Image

Run the `test_pub_image.py` script to publish an image to the `yolo_input_image` topic. This script works with PNG images:

```bash
rosrun barracuda_vision test_pub_image.py
```

This will publish an image to the topic, which will be processed by the `yolo_image_listener.py` script using the local model.

### Step 4: View the Result Image

To view the result image, you can use `rqt` (optionally in a conda environment):

```bash
conda activate ros_env
rqt
```

---

**Note:**
- The system no longer supports Roboflow server inference. Only local model inference is used.
- The container will shut down if no model or more than one model file is found in the mounted model directory.
- Make sure your model directory is mounted to `/opt/barracuda-vision/catkin_ws/src/barracuda_vision/localModels` inside the container (handled by the default docker-compose setup).