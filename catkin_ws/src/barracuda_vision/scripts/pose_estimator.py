#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Pose, PoseArray, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import math
import tf2_ros
import tf2_geometry_msgs

# Import custom messages
from barracuda_vision.msg import ObjectPose

# Import our custom modules
from particle_filter import ParticleFilter3D, TrackState
from camera_model import CameraModel
from geometry_utils import GeometryUtils

class PoseEstimator:
    """
    3D pose estimator using bounding boxes and depth information
    
    Coordinate Frame Architecture:
    - Raw object poses are calculated in camera_frame (sensor coordinates)
    - Poses are transformed to map frame for particle filter tracking
    - Individual ObjectPose messages published in camera_frame
    - Visualization (markers, pose arrays) published in map frame
    - Particle filter operates entirely in map frame for consistent tracking
    """
    
    def __init__(self):
        rospy.init_node('pose_estimator', anonymous=True)
        
        # Single publisher for individual object poses (in camera frame)
        self.pose_pub = rospy.Publisher('/object_pose', ObjectPose, queue_size=10)
        
        # Publishers for RViz visualization (in map frame)
        self.pose_stamped_pub = rospy.Publisher('/object_poses_stamped', PoseStamped, queue_size=10)
        self.pose_array_pub = rospy.Publisher('/object_poses_array', PoseArray, queue_size=10)
        self.marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
        
        # Subscribers
        self.bbox_sub = rospy.Subscriber('/bounding_boxes', BoundingBoxes, self.bbox_callback)
        self.depth_sub = rospy.Subscriber('/barracuda/left_camera/zed_image_depth', Image, self.depth_callback)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # TF2 buffer and listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Frame names
        self.camera_frame = "barracuda/zed_right_camera_frame"  # or "base_camera" depending on your setup
        self.map_frame = "map"
        # "barracuda/zed_right_camera_frame"
        
        # Initialize camera model
        self.camera_model = CameraModel()
        
        # Initialize geometry utilities
        self.geometry_utils = GeometryUtils()
        
        self.depth_image = None
        self.last_depth_time = None
        
        # Particle filters for different object classes
        self.particle_filters = {}
        
        # Store all current poses for array publishing
        self.current_poses = {}
        
        # Track which classes were seen in the current frame
        self.classes_seen_this_frame = set()
        
        rospy.loginfo("Pose estimator initialized")
        
        # Verify TF setup after a short delay
        rospy.Timer(rospy.Duration(2.0), self.verify_tf_setup, oneshot=True)
    
    def verify_tf_setup(self, event):
        """Verify that TF transforms are available"""
        try:
            # Check if transform from camera to map is available
            self.tf_buffer.lookup_transform(self.map_frame, self.camera_frame, rospy.Time(0), timeout=rospy.Duration(1.0))
            rospy.loginfo(f"✓ TF transform from {self.camera_frame} to {self.map_frame} is available")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"⚠ TF transform from {self.camera_frame} to {self.map_frame} not available: {e}")
            rospy.logwarn("Object poses will be published in camera frame, but filtering may not work correctly without proper transforms")
    
    def depth_callback(self, msg):
        """Store latest depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.last_depth_time = msg.header.stamp
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
    
    def transform_pose_to_map(self, position_3d_camera, timestamp):
        """Transform pose from camera frame to map frame"""
        try:
            # Create a PointStamped in camera frame
            point_camera = PointStamped()
            point_camera.header.frame_id = self.camera_frame
            point_camera.header.stamp = timestamp
            point_camera.point.x = position_3d_camera[0]
            point_camera.point.y = position_3d_camera[1] 
            point_camera.point.z = position_3d_camera[2]
            
            # Transform to map frame
            point_map = self.tf_buffer.transform(point_camera, self.map_frame, timeout=rospy.Duration(1.0))
            
            return np.array([point_map.point.x, point_map.point.y, point_map.point.z])
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not transform pose from {self.camera_frame} to {self.map_frame}: {e}")
            # If transformation fails, return the original position (fallback)
            # This assumes camera and map frames are aligned, which may not be true
            return position_3d_camera
    
    def bbox_callback(self, msg):
        """Process bounding boxes and estimate 3D poses"""
        if self.depth_image is None:
            rospy.logwarn("Waiting for depth image")
            return
        
        # Reset the set of classes seen this frame
        self.classes_seen_this_frame.clear()
        
        for bbox in msg.bounding_boxes:
            # Track that we've seen this class in this frame
            self.classes_seen_this_frame.add(bbox.Class)
            
            # Calculate 3D position from bounding box and depth (in camera frame)
            position_3d_camera = self.calculate_3d_position(bbox)
            print(f"Bounding box: {bbox.Class}, Position 3D (camera): {position_3d_camera}")
            
            if position_3d_camera is not None:
                # Transform position to map frame for particle filter
                position_3d_map = self.transform_pose_to_map(position_3d_camera, msg.header.stamp)
                print(f"Position 3D (map): {position_3d_map}")
                
                # Update particle filter for this object class (working in map frame)
                class_id = bbox.Class
                if class_id not in self.particle_filters:
                    # Create new optimized particle filter with decay-based loss detection
                    self.particle_filters[class_id] = ParticleFilter3D(
                        min_particles=5,    # Fewer particles during HOLD for efficiency
                        max_particles=25,    # Reduced from 100 for better performance
                        max_hold_frames=10,  # Still used for CONFIRMED->HOLD transition
                        survival_factor=0.97,  # 97% survival per frame
                        max_covariance_threshold=2.0,
                        min_confidence_threshold=0.1  # Track is LOST when confidence drops below 10%
                    )
                
                pf = self.particle_filters[class_id]
                
                # Update particle filter with new measurement in map frame
                pf.predict()
                pf.update(position_3d_map)
                
                # Get filtered estimate in map frame
                estimated_position_map = pf.get_estimate()
                
                # Check if we got a valid estimate (not None)
                if estimated_position_map is not None:
                    # Create ObjectPose message for this detection (in camera frame)
                    object_pose = ObjectPose()
                    object_pose.header = msg.header
                    object_pose.header.frame_id = self.camera_frame  # Pose is in camera frame
                    object_pose.object_class = bbox.Class
                    
                    # Set pose in camera frame (raw detection)
                    object_pose.pose.position.x = position_3d_camera[0]
                    object_pose.pose.position.y = position_3d_camera[1]
                    object_pose.pose.position.z = position_3d_camera[2]
                    
                    # For now, assume no rotation (identity quaternion)
                    object_pose.pose.orientation.x = 0.0
                    object_pose.pose.orientation.y = 0.0
                    object_pose.pose.orientation.z = 0.0
                    object_pose.pose.orientation.w = 1.0
                    
                    # Set covariance based on track confidence
                    confidence = pf.get_confidence()
                    base_uncertainty = 0.1  # 10cm std dev baseline
                    uncertainty = base_uncertainty / confidence if confidence > 0 else 1.0
                    covariance = np.eye(6) * uncertainty
                    object_pose.covariance = covariance.flatten().tolist()
                    
                    # Publish individual pose (in camera frame)
                    self.pose_pub.publish(object_pose)
                    
                    # Store filtered pose in map frame for visualization
                    map_pose = Pose()
                    map_pose.position.x = estimated_position_map[0]
                    map_pose.position.y = estimated_position_map[1]
                    map_pose.position.z = estimated_position_map[2]
                    map_pose.orientation.x = 0.0
                    map_pose.orientation.y = 0.0
                    map_pose.orientation.z = 0.0
                    map_pose.orientation.w = 1.0
                    
                    self.current_poses[bbox.Class] = map_pose
                    
                    # Publish PoseStamped for RViz visualization (in map frame)
                    pose_stamped = PoseStamped()
                    pose_stamped.header = msg.header
                    pose_stamped.header.frame_id = self.map_frame  # Visualization in map frame
                    pose_stamped.pose = map_pose
                    self.pose_stamped_pub.publish(pose_stamped)
                    
                    state_str = pf.get_state().value.upper()
                    frames_since = pf.get_frames_since_update()
                    confidence_pct = int(confidence * 100)
                    num_particles = pf.get_num_particles()
                    
                    rospy.loginfo(f"Published pose for {bbox.Class}: camera=({position_3d_camera[0]:.2f}, {position_3d_camera[1]:.2f}, {position_3d_camera[2]:.2f}), map=({estimated_position_map[0]:.2f}, {estimated_position_map[1]:.2f}, {estimated_position_map[2]:.2f}), state={state_str}, confidence={confidence_pct}%, particles={num_particles}")
                else:
                    rospy.logwarn(f"Particle filter for {bbox.Class} not properly initialized or returned invalid estimate")
        
        # Handle missed detections for existing tracks
        self.handle_missed_detections(msg.header)
        
        # Publish pose array and markers for all current objects (in map frame)
        self.publish_pose_array(msg.header)
        self.publish_markers(msg.header)
    
    def calculate_3d_position(self, bbox):
        """Calculate 3D position from bounding box and depth data"""
        try:
            # Get bounding box center using geometry utils
            center_x, center_y = self.geometry_utils.bbox_center(
                bbox.xmin, bbox.ymin, 
                bbox.xmax - bbox.xmin, 
                bbox.ymax - bbox.ymin
            )
            
            # Calculate depth at the center point
            depth = self.geometry_utils.calculate_depth_at_point(
                self.depth_image, center_x, center_y, window_size=5
            )
            
            if depth is None:
                rospy.logwarn("No valid depth found at bounding box center")
                return None
            
            # Convert mm to meters if needed
            depth_meters = depth / 1000.0 if depth > 10 else depth
            
            # Convert to 3D coordinates using camera model
            position_3d = self.camera_model.project_to_3d(
                center_x, center_y, depth_meters
            )
            
            return position_3d
            
        except Exception as e:
            rospy.logerr(f"Error calculating 3D position: {e}")
            return None
    
    def handle_missed_detections(self, header):
        """Handle tracks that didn't receive measurements this frame"""
        classes_to_remove = []
        
        for class_id, pf in self.particle_filters.items():
            if class_id not in self.classes_seen_this_frame:
                # This track didn't get a measurement this frame
                pf.handle_missed_detection()
                pf.predict()  # Still predict for held tracks
                
                if pf.is_lost():
                    # Mark for removal
                    classes_to_remove.append(class_id)
                    confidence_pct = int(pf.get_confidence() * 100)
                    frames_missed = pf.get_frames_since_update()
                    rospy.loginfo(f"Track for {class_id} lost due to low confidence ({confidence_pct}%) after {frames_missed} frames")
                else:
                    # Track is in HOLD state, still publish estimated pose
                    estimated_position_map = pf.get_estimate()
                    if estimated_position_map is not None:
                        # Store filtered pose in map frame for visualization
                        map_pose = Pose()
                        map_pose.position.x = estimated_position_map[0]
                        map_pose.position.y = estimated_position_map[1]
                        map_pose.position.z = estimated_position_map[2]
                        map_pose.orientation.x = 0.0
                        map_pose.orientation.y = 0.0
                        map_pose.orientation.z = 0.0
                        map_pose.orientation.w = 1.0
                        
                        self.current_poses[class_id] = map_pose
                        
                        # Publish PoseStamped for RViz visualization (in map frame)
                        pose_stamped = PoseStamped()
                        pose_stamped.header = header
                        pose_stamped.header.frame_id = self.map_frame
                        pose_stamped.pose = map_pose
                        self.pose_stamped_pub.publish(pose_stamped)
                        
                        state_str = pf.get_state().value.upper()
                        frames_since = pf.get_frames_since_update()
                        confidence_pct = int(pf.get_confidence() * 100)
                        num_particles = pf.get_num_particles()
                        
                        rospy.loginfo(f"Holding track for {class_id}: map=({estimated_position_map[0]:.2f}, {estimated_position_map[1]:.2f}, {estimated_position_map[2]:.2f}), state={state_str}, frames_missed={frames_since}, confidence={confidence_pct}%, particles={num_particles}")
        
        # Remove lost tracks
        for class_id in classes_to_remove:
            del self.particle_filters[class_id]
            if class_id in self.current_poses:
                del self.current_poses[class_id]
    
    def publish_pose_array(self, header):
        """Publish all current poses as a PoseArray for RViz visualization (in map frame)"""
        if self.current_poses:
            pose_array = PoseArray()
            pose_array.header = header
            pose_array.header.frame_id = self.map_frame  # Poses are in map frame
            pose_array.poses = list(self.current_poses.values())
            self.pose_array_pub.publish(pose_array)
    
    def publish_markers(self, header):
        """Publish markers for each detected object for RViz visualization (in map frame)"""
        marker_array = MarkerArray()
        
        for i, (object_class, pose) in enumerate(self.current_poses.items()):
            # Get particle filter info for this class
            pf = self.particle_filters.get(object_class)
            confidence = pf.get_confidence() if pf else 1.0
            state = pf.get_state() if pf else None
            
            # Create a marker for each object
            marker = Marker()
            marker.header = header
            marker.header.frame_id = self.map_frame  # Markers are in map frame
            marker.ns = "object_poses"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Set pose
            marker.pose = pose
            
            # Set scale based on confidence (smaller = less confident)
            base_scale = 0.2
            scale = base_scale * (0.5 + 0.5 * confidence)  # Scale between 50-100% based on confidence
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            
            # Set color based on object class and modify alpha based on confidence
            marker.color = self.get_color_for_class(object_class)
            marker.color.a = 0.4 + 0.6 * confidence  # Alpha between 40-100% based on confidence
            
            # Set lifetime
            marker.lifetime = rospy.Duration(1.0)  # 1 second
            
            marker_array.markers.append(marker)
            
            # Create text marker for object class and state
            text_marker = Marker()
            text_marker.header = header
            text_marker.header.frame_id = self.map_frame  # Text markers are in map frame
            text_marker.ns = "object_labels"
            text_marker.id = i + 1000  # Offset to avoid ID conflicts
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position text above the sphere
            text_marker.pose = pose
            text_marker.pose.position.z += 0.3
            
            # Set text with state and confidence info
            state_str = state.value.upper() if state else "UNKNOWN"
            confidence_pct = int(confidence * 100)
            text_marker.text = f"{object_class}\n{state_str} ({confidence_pct}%)"
            
            # Set scale (text size)
            text_marker.scale.z = 0.1
            
            # Set color based on state
            if state and state.value == "confirmed":
                text_marker.color.r = 0.0
                text_marker.color.g = 1.0
                text_marker.color.b = 0.0
            elif state and state.value == "hold":
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 0.0
            else:
                text_marker.color.r = 1.0
                text_marker.color.g = 0.0
                text_marker.color.b = 0.0
            text_marker.color.a = 1.0
            
            # Set lifetime
            text_marker.lifetime = rospy.Duration(1.0)
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)
    
    def get_color_for_class(self, object_class):
        """Get a color for a specific object class"""
        colors = {
            'gate': ColorRGBA(1.0, 0.0, 0.0, 1.0),      # Red
            'buoy': ColorRGBA(0.0, 1.0, 0.0, 1.0),      # Green
            'torpedo': ColorRGBA(0.0, 0.0, 1.0, 1.0),   # Blue
            'path': ColorRGBA(1.0, 1.0, 0.0, 1.0),      # Yellow
            'bin': ColorRGBA(1.0, 0.0, 1.0, 1.0),       # Magenta
        }
        
        # Return color for class or default to cyan
        return colors.get(object_class, ColorRGBA(0.0, 1.0, 1.0, 1.0))
    
    def run(self):
        """Main loop"""
        rospy.loginfo("Pose estimator running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        estimator = PoseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pose estimator interrupted")