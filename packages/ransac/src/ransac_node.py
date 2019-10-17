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


class PurePursuitNode(object):

    def __init__(self):
        self.node_name = "Pure Pursuit"

        # My variables
        self.colors = {0, 1}
        self.lookahead_distance = self.setup_parameter("~lookahead_distance", 0.10)
        self.reference_points = self.setup_parameter("~reference_points", 2)
        self.lane_size = self.setup_parameter("~lane_size", 0.585)
        self.v = self.setup_parameter("~linear_speed", 0.5)

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
        s = [(segment.color, np.mean([segment.points[0].x, segment.points[1].x]), np.mean([segment.points[0].y, segment.points[1].y])) for segment in segment_list if segment.color in self.colors]
        self.log_info("{}".format(s))
        self.log_info("-----------------")
        segment_list = sorted(segment_list, key=self.average_minus_lookahead)[:self.reference_points]
        if segment_list:
            for segment in segment_list:
                segment.points = sorted(segment.points, key=lambda point: point.x)

        return segment_list

    def lane_centers(self, segment_list_msg):
        # Get the closest lane centers
        closest_segment_list = self.closest_segments_to_lookahead(segment_list_msg.segments)
        lane_center_list = [self.calculate_lane_center(segment) for segment in closest_segment_list]

        return lane_center_list

    def calculate_lane_center(self, segment):
        # Calculate the normal vector to the segment list given assuming the list is ordered
        point_1 = segment.points[0]
        point_2 = segment.points[1]
        dy, dx = abs(point_2.y - point_1.y), point_2.x - point_1.x
        norm = (dx ** 2 + dy ** 2) ** 0.5

        # Find direction of normal vector depending on position and direction of segment
        # if not ((dy > 0) ^ (point_1.y > 0)):
        #     x = dy / norm
        # else:
        #     x = -dy / norm
        # if point_1.y > 0:
        #     y = - dx / norm
        # else:
        #     y = dx / norm
        ###
        # if dy > 0:
        #     x, y = dy / norm, -dx / norm
        # else:
        #     x, y = -dy / norm, dx / norm
        if segment.color == 0:
            x, y = -dy / norm, dx / norm
        elif segment.color == 1:
            x, y = dy / norm, -dx / norm
        else:
            x, y = 0, 0

        # Calculate the lane center using the normal vector, first point, and the lane size
        lane_center_x = point_1.x + x * (self.lane_size / 2)
        lane_center_y = point_1.y + y * (self.lane_size / 2)

        return lane_center_x, lane_center_y

    @staticmethod
    def filter_lane_centers(lane_center_list):
        # Filter valid lane centers
        lane_center_list = list(filter(lambda lc: (lc[0] is not None) and (lc[1] is not None), lane_center_list))

        return lane_center_list

    def calculate_speed(self, segment_list_msg):
        # self.log_info("{}".format(segment_list_msg))

        # Get lane centers
        lane_center_list = self.lane_centers(segment_list_msg)
        # self.log_info(str(lane_center_list))

        # Filter lane center
        valid_lane_centers = self.filter_lane_centers(lane_center_list)
        # self.log_info(str(valid_lane_centers))

        # Average valid lane centers, if none is valid set straight as goal
        if valid_lane_centers:
            # x = sum([lc[0] for lc in valid_lane_centers]) / len(valid_lane_centers)
            # y = sum([lc[1] for lc in valid_lane_centers]) / len(valid_lane_centers)
            x = np.median([lc[0] for lc in valid_lane_centers])
            y = np.median([lc[1] for lc in valid_lane_centers])
        else:
            x, y = 10, 0

        # self.log_info("{} - {}".format(x, y))
        # Calculate the angular speed
        omega = 2 * self.v * y / (x ** 2 + y ** 2)

        # self.log_info("{} - {}".format(self.v, omega))
        # self.log_info('------------------')
        # Publish the speed
        self.publish_cmd(self.v, omega)

    def publish_cmd(self, v, omega):
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v
        car_control_msg.omega = omega
        # self.pub_car_cmd.publish(car_control_msg)

    def on_shutdown(self):
        rospy.loginfo("[{}] Shutdown.".format(self.node_name))

    def log_info(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    rospy.init_node('pure_pursuit_node', anonymous=False)
    pure_pursuit_node = PurePursuitNode()
    rospy.on_shutdown(pure_pursuit_node.on_shutdown)
    rospy.spin()
