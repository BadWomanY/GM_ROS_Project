import torch
from isaacgym.torch_utils import *
from isaac_ros_bridge.utils.franka_utils import *
from IPython import embed


def gripper_logic(hand_pos, part_pos, dof_pos, arm_idx, robot_dof, grasp_offset, mode="auto", threshold=0.01):
    """
    mode: 
        - "auto" (default): close when near part
        - "open": always open
        - "close": always close
    """
    device = hand_pos.device

    gripper_sep = dof_pos[:, arm_idx, robot_dof] + dof_pos[:, arm_idx, robot_dof + 1]
    part_dist = torch.norm(part_pos - hand_pos, dim=-1).unsqueeze(-1)

    if mode == "close":
        grip_acts = torch.tensor([[0.0, 0.0]] * hand_pos.shape[0], device=device)
    elif mode == "open":
        grip_acts = torch.tensor([[0.04, 0.04]] * hand_pos.shape[0], device=device)
    else:  # auto mode
        gripped = (gripper_sep < 0.045) & (part_dist < grasp_offset + 0.5 * 0.03)
        close_gripper = (part_dist < grasp_offset + threshold) | gripped
        grip_acts = torch.where(
            close_gripper,
            torch.tensor([[0.0, 0.0]] * hand_pos.shape[0], device=device),
            torch.tensor([[0.04, 0.04]] * hand_pos.shape[0], device=device)
        )

    return grip_acts

def simple_controller(hand_pos, hand_rot, waypoints, goal_quat, gripp_quat, waypoint_idx,
                      j_eef, dof_pos, pos_action, mids, arm_idx, num_envs, robot_dof,
                      plan_mode, part_pos=None, grasp_offset=None, gripper_mode="auto"):
    device = hand_pos.device

    # Waypoint logic
    current_targets = torch.stack([
        waypoints[waypoint_idx[i]][i] for i in range(num_envs)
    ]).to(device)

    dist = torch.norm(hand_pos - current_targets, dim=-1)
    reached = dist < 0.03
    waypoint_idx[reached] += 1
    waypoint_idx = torch.clamp(waypoint_idx, max=len(waypoints) - 1)

    current_targets = torch.stack([
        waypoints[waypoint_idx[i]][i] for i in range(num_envs)
    ]).to(device)

    goal_pos = current_targets
    if plan_mode == "plan":
        goal_rot = torch.where(
            (waypoint_idx > 2 * len(waypoints) // 3).unsqueeze(-1),
            goal_quat,
            gripp_quat
        )
    else:
        goal_rot = goal_quat

    pos_err = torch.clamp(goal_pos - hand_pos, -0.03, 0.03)
    orn_err = torch.clamp(orientation_error(goal_rot, hand_rot), -0.2, 0.2)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    delta_q = control_ik_nullspace(dpose, j_eef, num_envs, dof_pos, mids, arm_idx, robot_dof)
    pos_action[:, arm_idx, :robot_dof] = dof_pos.squeeze(-1)[:, arm_idx, :robot_dof] + delta_q

    # Optional gripper logic
    if gripper_mode != "none" and part_pos is not None and grasp_offset is not None:
        grip_acts = gripper_logic(hand_pos, part_pos, dof_pos, arm_idx, robot_dof, grasp_offset, mode=gripper_mode)
        pos_action[:, arm_idx, robot_dof:robot_dof + 2] = grip_acts

    return waypoint_idx, pos_action


def weld_controller(hand_pos, hand_rot, waypoints, goal_quat,
                    waypoint_idx, j_eef, dof_pos, dt,
                    pos_action, mids, arm_idx, num_envs, 
                    robot_dof, timer, gripper_mode="auto"):
    device = hand_pos.device

    # Waypoint logic
    current_targets = torch.stack([
        waypoints[waypoint_idx[i]][i] for i in range(num_envs)
    ]).to(device)

    dist = torch.norm(hand_pos - current_targets, dim=-1)
    reached = dist < 0.03

    for i in range(num_envs):
        if reached[i] and waypoint_idx[i] == (len(waypoints) - 1):
            if gripper_mode == "auto" and timer < 2.0:
                grip_acts = torch.ones((num_envs, 2), device=device) * 0.02
                timer += dt
            else:
                grip_acts = torch.ones((num_envs, 2), device=device) * 0.04
        else:
            grip_acts = torch.ones((num_envs, 2), device=device) * 0.04
            if reached[i]:
                waypoint_idx[i] += 1

    waypoint_idx = torch.clamp(waypoint_idx, max=len(waypoints) - 1)

    current_targets = torch.stack([
        waypoints[waypoint_idx[i]][i] for i in range(num_envs)
    ]).to(device)

    goal_pos = current_targets
    # pos_err = torch.clamp(goal_pos - hand_pos, -0.02, 0.02)
    pos_err = torch.clamp(goal_pos - hand_pos, -0.015, 0.015)
    orn_err = torch.clamp(orientation_error(goal_quat, hand_rot), -0.2, 0.2)

    dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)

    delta_q = control_ik_nullspace(dpose, j_eef, num_envs, dof_pos, mids, 2, robot_dof)
    pos_action[:, 2, :robot_dof] = dof_pos.squeeze(-1)[:, 2, :robot_dof] + delta_q
    pos_action[:, arm_idx, robot_dof:robot_dof + 2] = grip_acts

    return waypoint_idx, pos_action, timer