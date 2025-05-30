#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge
from torchvision import transforms
from ultralytics import YOLO
import supervision as sv
import torch
import os
import sys

def infer(data):
    """Run inference on incoming image data using the local model only."""
    image = cvBridge.imgmsg_to_cv2(data, "bgr8")
    with torch.no_grad():
        outputs = model(image)
    rospy.loginfo(f"Local model inference: {len(outputs[0])} detections")
    process_result(data, image, outputs[0])


def process_result(data, image, result_or_outputs):
    """Process inference results and publish detection outputs."""
    detections = sv.Detections.from_ultralytics(result_or_outputs)
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id

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

    detectionImage = cvBridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
    detectionImage.header = data.header
    
    pub_object_detector.publish(len(detections))
    pub_bounding_boxes.publish(boundingBoxes)
    pub_detection_image.publish(detectionImage)


def listener():
    """Initialize ROS node and publishers, and subscribe to image topic."""
    rospy.init_node('yolo_image_listener', anonymous=True)
    rospy.Subscriber("yolo_input_image", Image, infer)

    global pub_object_detector, pub_bounding_boxes, pub_detection_image
    pub_object_detector = rospy.Publisher('object_detector', Int32, queue_size=10)
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
    cvBridge = CvBridge()
    listener()