import rospy
from geometry_msgs.msg import PoseStamped
import threading

class PosePublisher(threading.Thread):
    def __init__(self, pose_source_fn, rate_hz=60):
        super(PosePublisher, self).__init__()
        self.pose_source_fn = pose_source_fn
        self.rate_hz = rate_hz
        self.shutdown = False
        self.pub = rospy.Publisher('/rs007l/eef_pose', PoseStamped, queue_size=10)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and not self.shutdown:
            pos, quat = self.pose_source_fn()
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.pose.position.x = pos[0]
            msg.pose.position.y = pos[1]
            msg.pose.position.z = pos[2]
            msg.pose.orientation.x = quat[0]
            msg.pose.orientation.y = quat[1]
            msg.pose.orientation.z = quat[2]
            msg.pose.orientation.w = quat[3]
            self.pub.publish(msg)
            rate.sleep()