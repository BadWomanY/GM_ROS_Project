from isaacgym.torch_utils import *
from isaac_ros_bridge.planner.motion_planner import rrt_plan
from IPython import embed

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

# def orientation_error(desired, current):
#     cc = quat_conjugate(current)
#     q_r = quat_mul(desired, cc)
#     q_r = torch.where(q_r[:, 3:4] < 0, -q_r, q_r)  # ensure shortest path
#     return 2.0 * q_r[:, :3]



def box_grasping_yaw(q, axis_vec):
    """
    Returns yaw-only quaternion that aligns the gripper with the box's long side.
    axis_vec: a unit vector (e.g. [1, 0, 0]) pointing along the box's long side in local frame.
    """
    rc = quat_rotate(q, axis_vec)  # rotate long axis into world frame
    yaw = torch.atan2(rc[:, 1], rc[:, 0])  # angle in XY-plane (world)
    
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats

def control_ik_nullspace(dpose, j_eef, num_envs, dof_pos, mids, arm_idx, robot_dof, desired_q=None, damping=0.05, max_delta=0.15, nullspace_gain=0.1):

    # Get current joint positions
    q = dof_pos[:, arm_idx, :robot_dof].squeeze(-1)

    j_eef_T = torch.transpose(j_eef, 1, 2)
    identity6 = torch.eye(6, device=j_eef.device).unsqueeze(0).repeat(num_envs, 1, 1)
    lambda_mat = identity6 * (damping ** 2)

    A = j_eef @ j_eef_T + lambda_mat
    pinv = j_eef_T @ torch.linalg.solve(A, identity6)  # J^T (JJ^T + λI)^-1

    # Primary task solution
    dq_task = (pinv @ dpose).squeeze(-1)  # (num_envs, 7)

    # Null-space projection
    identity_r = torch.eye(robot_dof, device=j_eef.device).unsqueeze(0)
    null_proj = identity_r - pinv @ j_eef  # (num_envs, 7, 7)

    # Secondary objective: stay near mid-range joint angles
    if desired_q is None:
        desired_q = torch.tensor(mids[:robot_dof], device=q.device).unsqueeze(0).repeat(num_envs, 1)

    dq_null = nullspace_gain * (desired_q - q)  # (num_envs, 7)

    dq_total = dq_task + (null_proj @ dq_null.unsqueeze(-1)).squeeze(-1)

    # Clamp final update
    dq_total = torch.clamp(dq_total, -max_delta, max_delta)
    return dq_total

def matrix_to_quaternion(matrix):
    """
    Convert rotation matrix (batch) to quaternion.
    Input: (N, 3, 3) rotation matrices
    Output: (N, 4) quaternions in (x, y, z, w) format
    """
    import torch
    m = matrix
    r = torch.zeros((m.shape[0], 4), device=m.device)

    r[:, 0] = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=0.0)) / 2.0
    r[:, 1] = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], min=0.0)) / 2.0
    r[:, 2] = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], min=0.0)) / 2.0
    r[:, 3] = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], min=0.0)) / 2.0

    r[:, 0] = torch.copysign(r[:, 0], m[:, 2, 1] - m[:, 1, 2])
    r[:, 1] = torch.copysign(r[:, 1], m[:, 0, 2] - m[:, 2, 0])
    r[:, 2] = torch.copysign(r[:, 2], m[:, 1, 0] - m[:, 0, 1])

    return r


def look_at_quat(forward, up):
    """
    Generate a quaternion such that the local Z-axis aligns with `forward`,
    and the Y-axis (or X depending on convention) is oriented using `up`.
    """
    forward = torch.nn.functional.normalize(forward, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(up, forward, dim=-1), dim=-1)
    new_up = torch.cross(forward, right, dim=-1)

    rot_mat = torch.stack([right, new_up, forward], dim=-1)  # (B, 3, 3)
    return matrix_to_quaternion(rot_mat)

def elevated_midpoint(start: torch.Tensor, goal: torch.Tensor, z_offset=0.15):
    mid = (start + goal) / 2
    mid[2] = max(start[2], goal[2]) + z_offset
    return mid

def interpolate_waypoints(path, step=0.01):
    """
    Densify a path by interpolating between points.
    Each segment is divided into N = ceil(||p2 - p1|| / step) waypoints.
    Input:
        path: list of (3,) torch.Tensor
        step: float, max L2 distance between interpolated points
    Returns:
        smoothed_path: list of (3,) torch.Tensor
    """
    smoothed = []
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        dist = torch.norm(end - start).item()
        n_pts = max(2, int(dist / step))
        for t in torch.linspace(0, 1, n_pts, device=start.device):
            interp = (1 - t) * start + t * end
            smoothed.append(interp)
    smoothed.append(path[-1])
    return smoothed

def part_mate_hand_pos(anchor, anchor_rot, hand_part_offset, part_offset, big_part_welding_pos, small_part_welding_pos, device):
    centroid_big = big_part_welding_pos.mean(dim=1)
    centroid_small = small_part_welding_pos.mean(dim=1)

    Q_big = big_part_welding_pos[0] - centroid_big
    Q_part = small_part_welding_pos[0] - centroid_small

    H = Q_part.T @ Q_big
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    part_goal_quat = matrix_to_quaternion(R.unsqueeze(0))

    big_part_thickness = small_part_thickness = 0.01
    part_z_offset = torch.tensor([0, 0, -(big_part_thickness + small_part_thickness) / 2], device=device)
    part2_align = anchor + quat_rotate(anchor_rot, part_offset + part_z_offset)
    hand2_align = part2_align + quat_rotate(part_goal_quat, hand_part_offset)
    return hand2_align, part_goal_quat

def waypoint_generation(grasp_target, big_part_welding_pos, small_part_welding_pos, reachable, part_offset, obstacles, num_envs, device):
    # big_part_anchor = torch.tensor([0.0, -0.0, 0.7], device=device)
    # big_part_goal_quat = torch.tensor([[0.0, -0.707, 0.707, 0.0]] * num_envs).to(device)
    big_part_anchor = torch.tensor([0.0, -0.0, 0.74], device=device)
    big_part_goal_quat = torch.tensor([[0.7071068, 0, 0, -0.7071068]] * num_envs).to(device)
    hand_2_part = torch.tensor([0, 0, 0.132], device=device).unsqueeze(0)

    # Compute the desired mating orientation based on welding point cloud in world frame.
    # Algorithm used: Kabsch Algorithm.
    # big_part_welding_pos is predefined using the known anchor pos and quat.
    # TODO: The shape need further work if a new part is added.
    centroid_big = big_part_welding_pos.mean(dim=1)
    centroid_small = small_part_welding_pos.mean(dim=1)

    Q_big = big_part_welding_pos[0] - centroid_big
    Q_part = small_part_welding_pos[0] - centroid_small

    H = Q_part.T @ Q_big
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    part_goal_quat = matrix_to_quaternion(R.unsqueeze(0))

    big_part_thickness = small_part_thickness = 0.01
    part_z_offset = torch.tensor([0, 0, -(big_part_thickness + small_part_thickness) / 2], device=device)
    part2_align = big_part_anchor + quat_rotate(big_part_goal_quat, part_offset + part_z_offset)
    hand2_align = part2_align + quat_rotate(part_goal_quat, hand_2_part)

    start2 = grasp_target[0]
    goal2 = hand2_align[0]

    mid = elevated_midpoint(start2, goal2, z_offset=0.12)
    path1a = rrt_plan(start2, mid, reachable, obstacles, device=device, step_size=0.07, goal_thresh=0.01, safety_radius=0.12)
    path1b = rrt_plan(mid, goal2, reachable, obstacles, device=device, step_size=0.07, goal_thresh=0.01, safety_radius=0.12)
    path1 = path1a + path1b[1:]
    
    # Remove the jerky actions of the arms.
    dense_path = interpolate_waypoints(path1, step=0.01)
    hand_waypoints = [pt.unsqueeze(0).repeat(num_envs, 1) for pt in dense_path]

    hand_waypoints.append(hand2_align)
    waypoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    return waypoint_idx, hand_waypoints, part_goal_quat