#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def publish_image():
    rospy.init_node('image_test_publisher', anonymous=True)
    pub = rospy.Publisher('yolo_input_image', Image, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    bridge = CvBridge()

    while not rospy.is_shutdown():
        # Read an image using OpenCV

        cv_image = cv2.imread('../images/sea.jpg')
        if cv_image is None:
            rospy.logerr("Failed to read image")
            continue

        # Convert OpenCV image to ROS Image message
        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Publish the image
        print("Publishing image")
        pub.publish(ros_image)
        rate.sleep()

if __name__ == '__main__':
    try:
        print("Publishing image main")
        publish_image()
    except rospy.ROSInterruptException:
        print("An error occurred.")
        pass
