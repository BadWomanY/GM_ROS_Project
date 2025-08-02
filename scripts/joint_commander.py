#!/usr/bin/env python
import rospy
import moveit_commander
from sensor_msgs.msg import JointState
import sys

class RS007MoveItCommander:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('rs_moveit_live_controller', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        self.group.set_max_velocity_scaling_factor(0.1)
        self.group.set_max_acceleration_scaling_factor(0.1)

    def go_to_joint_state(self, target):
        self.group.go(target, wait=True)
        self.group.stop()


if __name__ == "__main__":
    controller = RS007MoveItCommander()

    # test_joint_target = [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_joint_target = [-1.5700, -0.1500, -0.9383, -0.1605, -2.1798, -1.3879]
    rospy.sleep(1.0)
    controller.go_to_joint_state(test_joint_target)

    moveit_commander.roscpp_shutdown()