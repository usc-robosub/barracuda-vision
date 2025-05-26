#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from inference_sdk import InferenceHTTPClient
from cv_bridge import CvBridge
import supervision as sv
import torch
from torchvision import transforms
import os
from ultralytics import YOLO
import sys
import base64

# Initialize inference client for remote server
client = InferenceHTTPClient(
    api_url="http://localhost:9001",  # Use local inference server
)

def infer(data, use_local_model):
    """Run inference on incoming image data using either local or remote model."""
    image = cvBridge.imgmsg_to_cv2(data, "bgr8")

    if use_local_model:
        with torch.no_grad():
            outputs = model(image)
        rospy.loginfo(f"Local model inference: {len(outputs[0])} detections")
        process_result(data, image, outputs[0], is_local=True)
    else:
        result = client.run_workflow(
            workspace_name="roboflow-docs",
            workflow_id="model-comparison",
            images={"image": image},
            parameters={"model1": "yolov11n-640"}
        )
        n_detections = len(result[0]["model1_predictions"]["predictions"])
        rospy.loginfo(f"Remote model inference: {n_detections} detections")
        process_result(data, image, result, is_local=False)

def process_result(data, image, result_or_outputs, is_local=False):
    """Process inference results and publish detection outputs."""
    if is_local:
        detections = sv.Detections.from_ultralytics(result_or_outputs)
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_id = detections.class_id
    else:
        predictions = result_or_outputs[0]["model1_predictions"]["predictions"]
        detections = sv.Detections.from_inference(result_or_outputs[0]["model1_predictions"])
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_id = [pred["class"] for pred in predictions]

    boundingBoxes = BoundingBoxes()
    boundingBoxes.header = data.header
    boundingBoxes.image_header = data.header
    boundingBoxes.bounding_boxes = []

    for i in range(len(detections)):
        boundingBox = BoundingBox()
        boundingBox.xmin = int(xyxy[i][0])
        boundingBox.ymin = int(xyxy[i][1])
        boundingBox.xmax = int(xyxy[i][2])
        boundingBox.ymax = int(xyxy[i][3])
        boundingBox.probability = confidence[i]
        boundingBox.Class = class_id[i]
        boundingBoxes.bounding_boxes.append(boundingBox)

    # Annotate and publish detection results
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )

    detectionImage = Image()
    detectionImage.header = data.header
    detectionImage.encoding = "bgr8"
    detectionImage.height = image.shape[0]
    detectionImage.width = image.shape[1]
    detectionImage.step = image.shape[1] * 3

    detectionImage = cvBridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")

    pub_object_detector.publish(len(detections))
    pub_bounding_boxes.publish(boundingBoxes)
    pub_detection_image.publish(detectionImage)

def listener():
    """Initialize ROS node and publishers, and subscribe to image topic."""
    rospy.init_node('yolo_image_listener', anonymous=True)
    use_local_model = rospy.get_param('~use_local_model', True)  # Default to local model
    rospy.Subscriber("yolo_input_image", Image, lambda data: infer(data, use_local_model))

    global pub_object_detector, pub_bounding_boxes, pub_detection_image
    pub_object_detector = rospy.Publisher('object_detector', Int8, queue_size=10)
    pub_bounding_boxes = rospy.Publisher('bounding_boxes', BoundingBoxes, queue_size=10)
    pub_detection_image = rospy.Publisher('detection_image', Image, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    # Find the localModels directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'localModels')

    # Ensure exactly one model file exists in localModels
    model_files = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
    if not model_files:
        rospy.logerr(f"No model file found in {models_dir}. Shutting down container.")
        sys.exit(1)
    if len(model_files) > 1:
        rospy.logerr(f"Multiple model files found in {models_dir}. Only one model should be present. Shutting down container.")
        sys.exit(1)

    # Load the only model file found
    model_path = os.path.join(models_dir, model_files[0])
    rospy.loginfo(f"Using model file: {model_path}")

    model = YOLO(model_path)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    cvBridge = CvBridge()
    listener()