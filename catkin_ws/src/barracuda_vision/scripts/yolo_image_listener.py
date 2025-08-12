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
import cv2

def letterbox_image(image, size=640):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded

def center_crop_image(image, size=640):
    h, w = image.shape[:2]
    min_dim = min(h, w, size)
    # Calculate cropping coordinates for center crop
    start_x = max((w - min_dim) // 2, 0)
    start_y = max((h - min_dim) // 2, 0)
    cropped = image[start_y:start_y+min_dim, start_x:start_x+min_dim]
    # If crop is not exactly size, resize
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return cropped

def infer(data):
    """Run inference on incoming image data using the local model only."""
    image = cvBridge.imgmsg_to_cv2(data, "bgr8")
    # image = letterbox_image(image, size=640)
    # image = center_crop_image(image, size=640)
    with torch.no_grad():
        outputs = model(image, conf = 0.4, iou= 0.4, agnostic_nms=True, max_det=300)
    rospy.loginfo(f"Local model inference: {len(outputs[0])} detections")
    process_result(data, image, outputs[0])


def process_result(data, image, result_or_outputs):
    """Process inference results and publish detection outputs."""
    detections = sv.Detections.from_ultralytics(result_or_outputs)
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id

    boundingBoxes = BoundingBoxes()
    # Use current time for bounding box message to avoid transform extrapolation issues
    boundingBoxes.header = data.header
    boundingBoxes.header.stamp = rospy.Time.now()
    boundingBoxes.image_header = data.header  # Keep original image header for reference
    boundingBoxes.bounding_boxes = []

    for i in range(len(detections)):
        boundingBox = BoundingBox()
        boundingBox.xmin = int(xyxy[i][0])
        boundingBox.ymin = int(xyxy[i][1])
        boundingBox.xmax = int(xyxy[i][2])
        boundingBox.ymax = int(xyxy[i][3])
        boundingBox.probability = confidence[i]
        # Convert class ID to class name if model has names, otherwise use string of ID
        if hasattr(model, 'names') and class_id[i] in model.names:
            boundingBox.Class = model.names[class_id[i]]
        else:
            boundingBox.Class = str(class_id[i])  # Convert numpy.int64 to string
        boundingBoxes.bounding_boxes.append(boundingBox)

    # Annotate and publish detection results
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )

    detectionImage = cvBridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
    # Use current time for detection image to match bounding box timestamp
    detectionImage.header = data.header
    detectionImage.header.stamp = rospy.Time.now()
    
    pub_object_detector.publish(len(detections))
    pub_bounding_boxes.publish(boundingBoxes)
    pub_detection_image.publish(detectionImage)


def listener():
    """Initialize ROS node and publishers, and subscribe to image topic."""
    rospy.init_node('yolo_image_listener', anonymous=True)
    rospy.Subscriber("/barracuda/right_camera/right_camera_image", Image, infer)

    global pub_object_detector, pub_bounding_boxes, pub_detection_image
    pub_object_detector = rospy.Publisher('object_detector', Int32, queue_size=10)
    pub_bounding_boxes = rospy.Publisher('bounding_boxes', BoundingBoxes, queue_size=10)
    pub_detection_image = rospy.Publisher('detection_image', Image, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    # Find the localModels directory relative to this script1.5708
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