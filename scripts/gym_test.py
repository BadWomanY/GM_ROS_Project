#!/usr/bin/env python3

import os
import sys
isaac_path = "/opt/isaacgym/python"
if isaac_path not in sys.path:
    sys.path.insert(0, isaac_path)
# === Ensure ROS + IsaacGym coexist ===
import rospy
from isaac_ros_bridge.robot_controller import ArmController1, ArmController2, ArmController3
from isaac_ros_bridge.planner.task_planner import TaskPlanner

# === Ensure src/ is in PYTHONPATH ===
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(pkg_root)

from isaac_ros_bridge.sim_manager import RobotCellSim
# from isaac_ros_bridge.robot_controller import RobotController
import torch
import numpy as np

from IPython import embed

if __name__ == "__main__":
    np.random.seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    sim = RobotCellSim(num_envs=1)

    task_planner = TaskPlanner(sim)
    task_planner.create_task_lib_r1()
    task_planner.create_task_lib_r2()
    task_planner.create_task_lib_r3()
    task_lib = task_planner.task_lib

    def get_eef_pose():
        pos = sim.hand1_pos[0].detach().cpu().numpy()
        quat = sim.hand1_rot[0].detach().cpu().numpy()
        return pos, quat

    # Robot controller
    rospy.init_node("sim_to_real_bridge", anonymous=True)
    arm1_controller = ArmController1(sim, task_lib, arm_idx=0)
    arm2_controller = ArmController2(sim, task_lib, arm_idx=1)
    arm3_controller = ArmController3(sim, task_lib, arm_idx=2)

    # Main simulation loop
    timer = 0
    real_timer = 0
    while not sim.is_viewer_closed():
        sim.step()

        # Evaluate/Assign tasks to each robot.
        cur_task = task_planner.task_assignment(timer, real_timer)
        arm1_task_name = cur_task[0][0]
        arm2_task_name = cur_task[1][0]

        arm1_pose = torch.cat([sim.hand1_pos, sim.hand1_rot], dim=1)
        arm2_pose = torch.cat([sim.hand2_pos, sim.hand2_rot], dim=1)
        arm3_pose = torch.cat([sim.hand3_pos, sim.hand3_rot], dim=1)
        # Arm goals.
        arm1_goal_pose = cur_task[0][1]
        arm2_goal_pose = cur_task[1][1]
        

        # Gripper Mode for each arm.
        gripper1_mode = cur_task[0][-2]
        gripper2_mode = cur_task[1][-2]
        

        # Planning required
        arm1_plan = cur_task[0][-3]
        arm2_plan = cur_task[1][-3]
        
        
        # Each robot plans/follows a path to reach goal.
        arm1_controller.step(arm1_task_name, arm1_pose, arm1_goal_pose, arm1_plan, gripper1_mode)
        arm2_controller.step(arm2_task_name, arm2_pose, arm2_goal_pose, arm2_plan, gripper2_mode)
        if cur_task[2] is not None:
            arm3_task_name = cur_task[2][0]
            arm3_goal_pose = cur_task[2][1]
            gripper3_mode = cur_task[2][-2]
            arm3_plan = cur_task[2][-3]
            timer, real_timer = arm3_controller.step(arm3_task_name, arm3_pose, arm3_goal_pose, arm3_plan, gripper3_mode)
        sim.update_viewer()
    sim.cleanup()