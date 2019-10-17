#!/usr/bin/env python
from cv_bridge import CvBridge
from duckietown_msgs.msg import SegmentList, LanePose, BoolStamped, Twist2DStamped, FSMState
from duckietown_utils.instantiate_utils import instantiate
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
import json
import numpy as np
import collections
import copy
import time


class SensorFusionNode(object):

    def __init__(self):
        self.node_name = "Sensor Fusion"

        # My variables
        self.colors = {0, 1}
        self.lookahead_distance = self.setup_parameter("~lookahead_distance", 0.10)
        self.reference_points = self.setup_parameter("~reference_points", 15)
        self.lane_size = self.setup_parameter("~lane_size", 0.585)
        self.v = self.setup_parameter("~linear_speed", 0.4)
        self.alpha = self.setup_parameter("~alpha", 1.5)
        self.init_time = time.time_ns()

        # Log file
        self.file = open("/data/logs/log.txt", "w")

        # Subscribers
        self.sub = rospy.Subscriber("~segment_list_filtered", SegmentList, self.calculate_speed, queue_size=1)
        self.sub_pose = rospy.Subscriber("~lane_pose", LanePose, self.log_to_file, queue_size=1)

        # Publication
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

    def setup_parameter(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)
        rospy.loginfo("[%s] %s = %s " % (self.node_name, param_name, value))
        return value

    def average_minus_lookahead(self, segment):
        # Average the segment endpoints on x and subtract the lookahead
        average = (segment.points[1].x + segment.points[0].x) / 2

        return abs(average - self.lookahead_distance)

    def closest_segments_to_lookahead(self, segment_list):
        # Get the closest point to the lookahead distance on the ground projection by averaging the two endpoints of
        # the segments, then sort the points in the segment in ascending distance.
        segment_list = [segment for segment in segment_list if segment.color in self.colors and
                        abs(segment.points[0].y + segment.points[1].y) / 2 - self.lane_size < 0]
        segment_list = sorted(segment_list, key=self.average_minus_lookahead)[:self.reference_points]

        return segment_list

    def segments_to_points(self, segment_list):
        white_points = []
        yellow_points = []
        for segment in segment_list:
            points = segment.points
            if segment.color == 0:
                white_points.append(((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2 + self.lane_size / 2))
            else:
                yellow_points.append(((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2 - self.lane_size / 2))
        return white_points, yellow_points

    def estimate_lookahead_point(self, point_list):
        numerator_x, denominator_x = 0, 0
        numerator_y, denominator_y = 0, 0
        for idx, points in enumerate(point_list):
            if len(points) > 2:
                data = np.array(points)
                x, y = data[:, :1], data[:, 1:2]
                numerator_x += len(points) * (1 / x.std()) * x.mean()
                denominator_x += len(points) * (1 / x.std())
                numerator_y += len(points) * (1 / y.std()) * y.mean()
                denominator_y += len(points) * (1 / y.std())
        if denominator_x == 0 or denominator_y == 0:
            x, y = self.lookahead_distance, 0
        else:
            x, y = numerator_x / denominator_x, numerator_y / denominator_y

        return x, y

    def calculate_speed(self, segment_list_msg):
        # Get the segments closest to the lookahead
        closest_segment_list = self.closest_segments_to_lookahead(segment_list_msg.segments)

        # Turn the segments to points
        point_list = self.segments_to_points(closest_segment_list)

        # Get the angle and number of
        x, y = self.estimate_lookahead_point(point_list)

        # Calculate the angular speed
        omega = self.alpha * 2 * self.v * y / (x ** 2 + y ** 2)

        # Log the speed
        self.file.write("[CMD]: {}, {}, {}\n".format(self.v, omega, time.time_ns() - self.init_time))
        # Publish the speed
        self.publish_cmd(self.v, omega)

    def publish_cmd(self, v, omega):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v
        car_control_msg.omega = omega
        self.pub_car_cmd.publish(car_control_msg)

    def log_to_file(self, lane_pose_msg):
        d = lane_pose_msg.d
        phi = lane_pose_msg.phi
        msg = "[ERR]: {}, {}, {}\n".format(d, phi, time.time_ns() - self.init_time)
        self.file.write(msg)

    def on_shutdown(self):
        rospy.loginfo("[{}] Shutdown.".format(self.node_name))
        self.file.close()

    def log_info(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    rospy.init_node('sensor_fusion_node', anonymous=False)
    sensor_fusion_node = SensorFusionNode()
    rospy.on_shutdown(sensor_fusion_node.on_shutdown)
    rospy.spin()
