#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize
import geometry_msgs.msg

class RealTimePoseFollower:
    def __init__(self):
        roscpp_initialize([])
        rospy.init_node('rs007l_pose_stream_follower', anonymous=True)

        self.robot = RobotCommander()
        self.group = MoveGroupCommander("manipulator")
        self.group.set_max_velocity_scaling_factor(0.3)
        self.group.set_max_acceleration_scaling_factor(0.3)

        self.joint_pub = rospy.Publisher(
            "/rs007l_arm_controller/command", JointTrajectory, queue_size=1
        )

        rospy.Subscriber("/rs007l_real_pose", Pose, self.pose_callback)
        self.last_q = None

    def pose_callback(self, msg):
        # Use IK to compute joint angles (no planning)
        self.group.set_pose_target(msg)
        plan = self.group.plan()

        if plan[0] and plan[1].joint_trajectory.points:
            pt = plan[1].joint_trajectory.points[-1]
            traj = JointTrajectory()
            traj.header.stamp = rospy.Time.now()
            traj.joint_names = plan[1].joint_trajectory.joint_names
            pt.time_from_start = rospy.Duration(1.0 / 60.0)
            traj.points = [pt]
            self.joint_pub.publish(traj)
        else:
            rospy.logwarn("Failed IK or empty trajectory point")

if __name__ == '__main__':
    try:
        RealTimePoseFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
