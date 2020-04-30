# import time
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray

from dr_spaam.detector import Detector


def read_subscriber_param(name):
    """
    @brief      Convenience function to read subscriber parameter.
    """
    topic = rospy.get_param("~subscriber/" + name + "/topic")
    queue_size = rospy.get_param("~subscriber/" + name + "/queue_size")
    return topic, queue_size


def read_publisher_param(name):
    """
    @brief      Convenience function to read publisher parameter.
    """
    topic = rospy.get_param("~publisher/" + name + "/topic")
    queue_size = rospy.get_param("~publisher/" + name + "/queue_size")
    latch = rospy.get_param("~publisher/" + name + "/latch")
    return topic, queue_size, latch


class DrSpaamROS():
    """ROS node to detect pedestrian using DR-SPAAM."""
    def __init__(self):
        self._read_params()
        self._detector = Detector(
            self.weight_file, original_drow=False,
            gpu=True, stride=self.stride)
        self._init()

    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        self.weight_file = rospy.get_param("~weight_file")
        self.conf_thresh = rospy.get_param("~conf_thresh")
        self.stride = rospy.get_param("~stride")

    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = read_publisher_param("detections")
        self._dets_pub = rospy.Publisher(
            topic, PoseArray, queue_size=queue_size, latch=latch)

        # Subscriber
        topic, queue_size = read_subscriber_param("scan")
        self._scan_sub = rospy.Subscriber(
            topic, LaserScan, self._scan_callback, queue_size=queue_size)

    def _scan_callback(self, msg):
        if self._dets_pub.get_num_connections() == 0:
            return

        if not self._detector.laser_spec_set():
            self._detector.set_laser_spec(angle_inc=msg.angle_increment,
                                          num_pts=len(msg.ranges))

        scan = np.array(msg.ranges)
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        # t = time.time()
        dets_xy, dets_cls, _ = self._detector(scan)
        # print("[DrSpaamROS] End-to-end inference time: %f" % (t - time.time()))

        # confidence threshold
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        # if not np.sum(conf_mask) > 0:
        #     return
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]

        # convert and publish ros msg
        dets_msg = self._detections_to_ros_msg(dets_xy, dets_cls)
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

    def _detections_to_ros_msg(self, dets_xy, dets_cls):
        pose_array = PoseArray()
        for d_xy, d_cls in zip(dets_xy, dets_cls):
            # If laser is facing front, DR-SPAAM's y-axis aligns with the laser
            # center ray, x-axis points to right, z-axis points upward
            p = Pose()
            p.position.x = d_xy[1]
            p.position.y = d_xy[0]
            p.position.z = 0.0
            pose_array.poses.append(p)

        return pose_array
