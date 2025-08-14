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
        self.robot = moveit_commander.MoveGroupCommander("manipulator")
        self.robot.set_max_velocity_scaling_factor(0.1)
        self.robot.set_max_acceleration_scaling_factor(0.1)
        self.tool = moveit_commander.MoveGroupCommander("tool")
        print(self.tool.get_active_joints())
        pose = self.robot.get_current_pose(end_effector_link="weld_tip").pose
        print("Position:", pose.position)
        print("Orientation:", pose.orientation)

    def go_to_joint_state(self, target):
        self.robot.go(target[:6], wait=True)
        self.tool.go(target[6:], wait=True)
        self.robot.stop()


if __name__ == "__main__":
    controller = RS007MoveItCommander()

    # test_joint_target = [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0]
    # test_joint_target = [-1.5700, -0.1500, -0.9383, -0.1605, -2.1798, -1.3879]
    test_joint_target = [-1.5700, -0.1500, -0.9383, -0.1605, -1.57, -1.3879, 0.04, 0.04]  # Adjusted for RS007L with gripper
    # sample solution from quat = [0, 0, -1, 0] (180 deg around Z)
    # test_joint_target = [-1.5708292284685044, -0.3445853532424338, -1.9916518926182303, -3.141241790894978, -1.6465793947660872, -1.5702051247833892, 0.03979074279199716, 0.03979243401909016]
    rospy.sleep(1.0)
    controller.go_to_joint_state(test_joint_target)

    moveit_commander.roscpp_shutdown()