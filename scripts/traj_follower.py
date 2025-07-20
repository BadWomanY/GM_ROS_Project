#!/usr/bin/env python
import rospy
import actionlib
from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

class RS007LJointTrajectoryFollower:
    def __init__(self):
        rospy.init_node("rs007l_trajectory_follower")

        # Connect to RS007L's MoveIt controller
        self.client = actionlib.SimpleActionClient(
            '/rs007l_arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )

        rospy.loginfo("Waiting for /rs007l_arm_controller/follow_joint_trajectory server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to RS007L trajectory action server.")

        # Subscribe to the trajectory topic
        rospy.Subscriber(
            '/rs007l_joint_trajectory',
            JointTrajectory,
            self.trajectory_callback
        )

    def trajectory_callback(self, msg):
        rospy.loginfo(f"Received trajectory with {len(msg.points)} points.")
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = msg
        self.client.send_goal(goal)
        self.client.wait_for_result()
        rospy.loginfo("Trajectory execution complete.")

if __name__ == '__main__':
    try:
        RS007LJointTrajectoryFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
