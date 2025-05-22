#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from inference_sdk import InferenceHTTPClient
from cv_bridge import CvBridge
import supervision as sv
import torch 
import torch.nn as nn
from torchvision import transforms
import os
from ultralytics import YOLO

# from base64 import b64encode
import base64

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    # api_key="<YOUR API KEY>" # optional to access your private data and models
)

def infer(data, use_local_model):
    rospy.loginfo(rospy.get_caller_id() + " I heard an image")
    image = cvBridge.imgmsg_to_cv2(data, "bgr8")

    if use_local_model:
        with torch.no_grad():
            outputs = model(image)
        print("model output", outputs)
        print("model output type", type(outputs))
        process_result(data, image, outputs[0], is_local=True)
    else:
        result = client.run_workflow(
            workspace_name="roboflow-docs",
            workflow_id="model-comparison",
            images={
                "image": image
            },
            parameters={
                "model1": "yolov11n-640"
            }
        )
        print(result[0]["model1_predictions"]["predictions"])
        process_result(data, image, result, is_local=False)

def process_result(data, image, result_or_outputs, is_local=False):
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
    rospy.init_node('yolo_image_listener', anonymous=True)
    use_local_model = rospy.get_param('~use_local_model', True)  # Default to local model
    rospy.Subscriber("yolo_input_image", Image, lambda data: infer(data, use_local_model))

    global pub_object_detector, pub_bounding_boxes, pub_detection_image
    pub_object_detector = rospy.Publisher('object_detector', Int8, queue_size=10)
    pub_bounding_boxes = rospy.Publisher('bounding_boxes', BoundingBoxes, queue_size=10)
    pub_detection_image = rospy.Publisher('detection_image', Image, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'localModels', 'yolo11n.pt')
    # model_path = os.path.join(script_dir, '..', 'localModels')
    # model = torch.hub.load(model_path, 'yolov11n', pretrained=True)

    ## this load custom model , but require internet 
    # model = torch.hub.load("Ultralytics/yolov11n","custom",f"{model_path}",trust_repo=True)
    
    ## this does not work
    # model = torch.load(model_path)

    model = YOLO(model_path)
    model.eval()
    # Define preprocessing (adjust to your model)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # match your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # update based on training
    ])


    cvBridge = CvBridge()

    listener()