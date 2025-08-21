#!/usr/bin/env python
import rospy
import moveit_commander
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class RS007MoveItCommander:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('rs_moveit_live_controller', anonymous=True)

        self.robot1 = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot1 = moveit_commander.MoveGroupCommander("a1_manipulator")
        # self.robot1.set_planner_id("PTP")
        self.robot1.set_max_velocity_scaling_factor(0.2)
        self.robot1.set_max_acceleration_scaling_factor(0.2)
        
        self.tool1_pub = rospy.Publisher("/a1/franka_gripper_controller/command", JointTrajectory, queue_size=1)
        pose1 = self.robot1.get_current_pose(end_effector_link="a1_weld_tip").pose
        print("Position:", pose1.position)
        print("Orientation:", pose1.orientation)

        self.robot2 = moveit_commander.MoveGroupCommander("a2_manipulator")
        # self.robot2.set_planner_id("PTP")
        self.robot2.set_max_velocity_scaling_factor(0.2)
        self.robot2.set_max_acceleration_scaling_factor(0.2)
        
        self.tool2_pub = rospy.Publisher("/a2/franka_gripper_controller/command", JointTrajectory, queue_size=1)
        pose2 = self.robot2.get_current_pose(end_effector_link="a2_weld_tip").pose
        print("Position:", pose2.position)
        print("Orientation:", pose2.orientation)

        self.robot3 = moveit_commander.MoveGroupCommander("a3_manipulator")
        # self.robot3.set_planner_id("PTP")
        self.robot3.set_max_velocity_scaling_factor(0.2)
        self.robot3.set_max_acceleration_scaling_factor(0.2)
        
        self.tool3_pub = rospy.Publisher("/a3/franka_gripper_controller/command", JointTrajectory, queue_size=1)
        pose3 = self.robot3.get_current_pose(end_effector_link="a3_weld_tip").pose
        print("Position:", pose3.position)
        print("Orientation:", pose3.orientation)

    

    def _send_gripper(self, prefix, width: float, duration_s: float = 0.1):
        """
        width: finger separation per joint (you use two independent prismatic joints).
            0.04 -> open, 0.00 -> close
        """
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = [f"{prefix}panda_finger_joint1", f"{prefix}panda_finger_joint2"]

        p0 = JointTrajectoryPoint()
        # read current finger pos from /joint_states if you store it; fallback to current command:
        # here we just start at current q_real end segments if you mirror them; safe to omit
        p0.positions = [None, None]  # controller will treat as start-at-current if omitted
        p0.time_from_start = rospy.Duration(0.0)

        p1 = JointTrajectoryPoint()
        p1.positions = [width, width]
        p1.time_from_start = rospy.Duration(duration_s)

        jt.points = [p1]  # send single timed point is fine; controller ramps to it
        if prefix == "a1_":
            self.tool1_pub.publish(jt)
        elif prefix == "a2_":
            self.tool2_pub.publish(jt)
        elif prefix == "a3_":
            self.tool3_pub.publish(jt)

    def go_to_targets(self, targets, mode="joint"):
        """
        Move each robot either in joint space or Cartesian LIN.

        Args:
            targets: list of 3 targets
                - if mode == "joint": each target is a list of 6+ DOF joint values
                - if mode == "cartesian": each target is a geometry_msgs/Pose
            mode: "joint" or "cartesian"
        """
        robots = [self.robot1, self.robot2, self.robot3]
        grippers = ["a1_", "a2_", "a3_"]

        for idx, robot in enumerate(robots):
            # --- Joint-space move ---
            self._send_gripper(grippers[idx], 0.04)
            robot.go(targets[idx][:6], wait=True)
            robot.stop()


if __name__ == "__main__":
    controller = RS007MoveItCommander()

    # test_joint_target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # test_joint_target = [-1.5700, -0.1500, -0.9383, -0.1605, -2.1798, -1.3879]
    a1_joint_target = [0, -0.0325, -1.5809, 0, -1.8860, -1.5678, 0.04, 0.04]
    a2_joint_target = [0, -0.0325, -1.5809, 0, -1.8860, -1.5678, 0.04, 0.04]
    a3_joint_target = [-1.5700, -0.1500, -0.9383, -0.1605, -1.57, -1.3879, 0.04, 0.04]  # Adjusted for RS007L with gripper
    # sample solution from quat = [0, 0, -1, 0] (180 deg around Z)
    # test_joint_target = [-1.5708292284685044, -0.3445853532424338, -1.9916518926182303, -3.141241790894978, -1.6465793947660872, -1.5702051247833892, 0.03979074279199716, 0.03979243401909016]
    rospy.sleep(1.0)
    controller.go_to_targets([a1_joint_target, a2_joint_target, a3_joint_target], mode="joint")

    moveit_commander.roscpp_shutdown()