# barracuda-vision

## Running with Docker

### Step 1: Start the Inference Server

Run the following command to start the inference server:

```bash
inference server start
```

### Step 2: Build and Run the Docker Container

Use the provided Dockerfile and docker-compose.yaml to build and run the Docker container:

```bash
docker-compose up --build
```

### Step 3: Publish an Image

Run the `test_pub_image.py` script to publish an image to the `yolo_input_image` topic. Note that this script only works with PNG images:

```bash
rosrun barracuda_vision test_pub_image.py
```

This will publish an image to the topic, which will be processed by the `yolo_image_listener.py` script.

### Step 4: View the Result Image

To run `rqt` to see the result image, activate the `ros_env` conda environment:

```bash
conda activate ros_env
rqt
```