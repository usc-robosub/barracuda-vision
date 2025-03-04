#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from darknet_ros_msgs.msg import BoundingBoxes
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    # api_key="<YOUR API KEY>" # optional to access your private data and models
)

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + " I heard an image")
    # Process the data and publish the result
    # Assuming data contains the image data
    result = client.run_workflow(
        workspace_name="roboflow-docs",
        workflow_id="model-comparison",
        images={
            "image": data
        },
        parameters={
            "model1": "yolov8n-640",
            "model2": "yolov11n-640"
        }
    )
    # Print the result
    print(result)
    
    # Publish the result
    object_detector.publish(result["object_detector"])
    bounding_boxes.publish(result["bounding_boxes"])
    detection_image.publish(result["detection_image"])

def listener():
    rospy.init_node('barracuda_vision_listener', anonymous=True)
    rospy.Subscriber("input_topic", Image, callback)
    global object_detector, bounding_boxes, detection_image
    object_detector = rospy.Publisher('object_detector', Int8, queue_size=10)
    bounding_boxes = rospy.Publisher('bounding_boxes', BoundingBoxes, queue_size=10)
    detection_image = rospy.Publisher('detection_image', Image, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    listener()