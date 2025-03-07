#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from darknet_ros_msgs.msg import BoundingBoxes
from inference_sdk import InferenceHTTPClient
from cv_bridge import CvBridge
from base64 import b64encode

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    # api_key="<YOUR API KEY>" # optional to access your private data and models
)

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + " I heard an image")
    # Process the data and publish the result
    # Assuming data contains the image data
    image = CvBridge().imgmsg_to_cv2(data, "bgr8")
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
    # Print the result
    print(result)
    
    boundingBoxes = BoundingBoxes()
    boundingBoxes.header = data.header
    boundingBoxes.image_header = data.header

    detectionImage = Image()
    detectionImage.header = data.header
    detectionImage.header = data.header
    detectionImage.encoding = "bgr8"
    detectionImage.height = result[0]["model1_predictions"]['image']['height']
    detectionImage.width = result[0]["model1_predictions"]['image']['width']
    detectionImage.step = data.step
    detectionImage.data = b64encode(result[0]["model_comparison_visualization"]).decode('utf-8')

    
    # Publish the result
    pub_object_detector.publish(len(result[0]["model1_predictions"]['predictions']))
    pub_bounding_boxes.publish(boundingBoxes)
    pub_detection_image.publish(detectionImage)

def listener():
    rospy.init_node('barracuda_vision_listener', anonymous=True)
    rospy.Subscriber("yolo_input_image", Image, callback)
    global pub_object_detector, pub_bounding_boxes, pub_detection_image
    pub_object_detector = rospy.Publisher('object_detector', Int8, queue_size=10)
    pub_bounding_boxes = rospy.Publisher('bounding_boxes', BoundingBoxes, queue_size=10)
    pub_detection_image = rospy.Publisher('detection_image', Image, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    listener()