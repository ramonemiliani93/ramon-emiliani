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
from sklearn import linear_model

class RansacNode(object):

    def __init__(self):
        self.node_name = "Ransac"

        # My variables
        self.colors = {0, 1}
        self.lookahead_distance = self.setup_parameter("~lookahead_distance", 0.10)
        self.reference_points = self.setup_parameter("~reference_points", 20)
        self.lane_size = self.setup_parameter("~lane_size", 0.585)
        self.v = self.setup_parameter("~linear_speed", 0.5)

        # Model
        self.ransac = linear_model.RANSACRegressor()
        self.past_angle = 0
        self.alpha = self.setup_parameter("~alpha", 0.4)

        # Subscribers
        self.sub = rospy.Subscriber("~segment_list_filtered", SegmentList, self.calculate_speed, queue_size=1)

        # Publication
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

    def setup_parameter(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)   # Write to parameter server for transparancy
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

    @staticmethod
    def segments_to_points(segment_list):
        white_points = []
        yellow_points = []
        for segment in segment_list:
            points = segment.points
            if segment.color == 0:
                white_points.append(((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2))
            else:
                yellow_points.append(((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2))
        return white_points, yellow_points

    def estimate_angle(self, points_per_color):
        angle, count = 0, 0
        for color in points_per_color:
            if len(color) > 2:
                data = np.array(color)
                self.ransac.fit(data[:, :1], data[:, 1:2])
                angle += np.arctan(self.ransac.estimator_.coef_).item()
                count += 1
        return angle, count

    def calculate_speed(self, segment_list_msg):
        # Get the segments closest to the lookahead
        closest_segment_list = self.closest_segments_to_lookahead(segment_list_msg.segments)
        self.log_info("{}")

        # Turn the segments to points
        point_list = self.segments_to_points(closest_segment_list)
        self.log_info("{}".format(point_list[0]))

        # Get the angle and number of
        angle, count = self.estimate_angle(point_list)

        # Weighted average of the current angle and the past ones
        if count > 0:
            angle = self.alpha * self.past_angle + (1 - self.alpha) * (angle / count)
            self.past_angle = angle
        else:
            angle = self.past_angle

        # Calculate the angular speed
        omega = 2 * self.v * np.sin(angle) / (self.lookahead_distance)

        self.log_info("{} - {}".format(self.v, omega))
        # self.log_info('------------------')
        # Publish the speed
        self.publish_cmd(self.v, omega)

    def publish_cmd(self, v, omega):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v
        car_control_msg.omega = omega
        self.pub_car_cmd.publish(car_control_msg)

    def on_shutdown(self):
        rospy.loginfo("[{}] Shutdown.".format(self.node_name))

    def log_info(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    rospy.init_node('ransac_node', anonymous=False)
    ransac_node = RansacNode()
    rospy.on_shutdown(ransac_node.on_shutdown)
    rospy.spin()
