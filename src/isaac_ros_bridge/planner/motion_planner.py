import torch
import random
from IPython import embed

# class RRTNode:
#     def __init__(self, pos: torch.Tensor, parent=None):
#         self.pos = pos  # shape: (3,)
#         self.parent = parent

# def is_in_collision(point, obstacles, safety_radius=0.0):
#     """
#     Point: (3,)
#     obstacles: list of (low, high)
#     safety_radius: float, how far to keep away from obstacles
#     """
#     for low, high in obstacles:
#         inflated_low = low - safety_radius
#         inflated_high = high + safety_radius
#         if torch.all(point >= inflated_low) and torch.all(point <= inflated_high):
#             return True
#     return False


# def steer(from_point, to_point, step_size):
#     direction = to_point - from_point
#     dist = torch.norm(direction)
#     if dist < step_size:
#         return to_point
#     return from_point + direction / dist * step_size


def rrt_plan(start: torch.Tensor, goal: torch.Tensor, reachable: torch.Tensor, obstacles: list,
             step_size=0.05, max_iter=1000, goal_thresh=0.05, device="cuda:0", safety_radius = 0.15):
    """
    start, goal: (3,) torch.Tensor
    obstacles: list of (low: (3,), high: (3,))
    bounds: list of (low, high) tuples for each dim
    returns: list of (num_envs, 3) tensors
    """
    torch.manual_seed(42)
    random.seed(42)

    tree = [RRTNode(start)]

    # voxel_resolution, voxel_origin, voxel_keys = build_voxel_grid(reachable)
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.4, 1.0)]
    for _ in range(max_iter):
        # Sample from reachable set
        if random.random() < 0.1:
            sample = goal
        else:
            sample_idx = random.randint(0, reachable.shape[0] - 1)
            sample = reachable[sample_idx]

        # Find nearest node
        dists = [torch.norm(n.pos - sample) for n in tree]
        nearest = tree[torch.argmin(torch.tensor(dists))]

        # Steer
        new_pos = steer(nearest.pos, sample, step_size)

        # Sample point
        # if random.random() < 0.1:
        #     sample = goal
        # else:
        #     sample = torch.tensor([random.uniform(lo, hi) for lo, hi in bounds], device=device)
        # # sample = goal if random.random() < 0.1 else sample_point(bounds, device)  


        # # Find nearest node
        # dists = [torch.norm(n.pos - sample) for n in tree]
        # nearest = tree[torch.argmin(torch.tensor(dists))]

        # # Steer
        # new_pos = steer(nearest.pos, sample, step_size)

        if is_in_collision(new_pos, obstacles, safety_radius):
            continue

        new_node = RRTNode(new_pos, parent=nearest)
        tree.append(new_node)

        if torch.norm(new_node.pos - goal) < goal_thresh:
            # Trace back
            path = []
            cur = new_node
            while cur is not None:
                path.append(cur.pos)
                cur = cur.parent
            path.reverse()
            return path


    print("RRT failed to find a path.")
    return []
import torch
import random
from collections import deque
# from urdfpy import URDF
# from scipy.spatial.transform import Rotation
# import numpy as np

# robot = URDF.load("/home/gymuser/catkin_ws/src/isaac_ros_bridge/scripts/assets/urdf/RS007/rs007l_panda.urdf")

# def fwd_kin_urdfpy(robot, q):
#     # q: (dof,) list or np.array of joint values
#     joint_names = [j.name for j in robot.joints if j.joint_type != 'fixed'][:6]
#     joint_map = dict(zip(joint_names, q.tolist()))
    
#     # Get pose of the end-effector (assume link6 is the end-effector)
#     ee_tf = robot.link_fk(joint_map)[robot.links[-3]]  # (4x4)

#     # --- Apply offset transform here ---
#     # Define fixed transform from URDF frame to Isaac frame
#     offset_rot = Rotation.from_euler('z', -180, degrees=True).as_matrix()  # 180° Z
#     offset_trans = np.array([-0.45, 0.0, 0.1])
    
#     T_offset = np.eye(4)
#     T_offset[:3, :3] = offset_rot
#     T_offset[:3, 3] = offset_trans

#     # Apply transform: T_final = T_offset @ T_urdf
#     ee_tf_transformed = T_offset @ ee_tf

#     # Extract transformed position and orientation
#     pos = ee_tf_transformed[:3, 3]
#     quat_xyzw = rotation_matrix_to_quaternion(ee_tf_transformed[:3, :3])  # (x, y, z, w)

#     return torch.tensor(pos, dtype=torch.float32), torch.tensor(quat_xyzw, dtype=torch.float32)


# def rotation_matrix_to_quaternion(R):
#     # Convert 3x3 rotation matrix to (x, y, z, w) quaternion
#     return Rotation.from_matrix(R).as_quat()

class RRTNode:
    def __init__(self, pos: torch.Tensor, parent=None, cost=0.0):
        self.pos = pos  # (3,)
        self.parent = parent
        self.cost = cost  # total cost to reach this node

def is_in_collision(point, obstacles, safety_radius=0.0):
    for low, high in obstacles:
        inflated_low = low - safety_radius
        inflated_high = high + safety_radius
        if torch.all(point >= inflated_low) and torch.all(point <= inflated_high):
            return True
    return False

def steer(from_point, to_point, step_size):
    direction = to_point - from_point
    dist = torch.norm(direction)
    if dist < step_size:
        return to_point
    return from_point + direction / dist * step_size

def get_nearest(tree, sample):
    dists = [torch.norm(n.pos - sample) for n in tree]
    return tree[torch.argmin(torch.tensor(dists))]

def get_nearby(tree, new_pos, radius):
    return [n for n in tree if torch.norm(n.pos - new_pos) <= radius]

def rrt_star_plan(start, goal, reachable, obstacles, step_size=0.05,
                  max_iter=1000, goal_thresh=0.05, rewire_radius=0.2,
                  device="cuda:0", safety_radius=0.15):
    torch.manual_seed(42)
    random.seed(42)

    tree = [RRTNode(start, cost=0.0)]
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.4, 1.0)]

    for _ in range(max_iter):
        # Sample
        if random.random() < 0.1:
            sample = goal
        else:
            sample = reachable[random.randint(0, reachable.shape[0] - 1)]

        # Nearest node
        nearest = get_nearest(tree, sample)
        new_pos = steer(nearest.pos, sample, step_size)

        if is_in_collision(new_pos, obstacles, safety_radius):
            continue

        # Create new node
        new_cost = nearest.cost + torch.norm(new_pos - nearest.pos)
        new_node = RRTNode(new_pos, parent=nearest, cost=new_cost)

        # Find nearby nodes to possibly rewire
        nearby_nodes = get_nearby(tree, new_pos, radius=rewire_radius)

        # Choose best parent among nearby nodes
        for node in nearby_nodes:
            potential_cost = node.cost + torch.norm(new_pos - node.pos)
            if not is_in_collision((node.pos + new_pos) / 2, obstacles, safety_radius) and potential_cost < new_node.cost:
                new_node.parent = node
                new_node.cost = potential_cost

        tree.append(new_node)

        # Rewire nearby nodes to new node if it's better
        for node in nearby_nodes:
            potential_cost = new_node.cost + torch.norm(node.pos - new_node.pos)
            if not is_in_collision((node.pos + new_node.pos) / 2, obstacles, safety_radius) and potential_cost < node.cost:
                node.parent = new_node
                node.cost = potential_cost

        # Check for goal reach
        if torch.norm(new_node.pos - goal) < goal_thresh:
            # Reconstruct path
            path = []
            cur = new_node
            while cur is not None:
                path.append(cur.pos)
                cur = cur.parent
            path.reverse()
            return path

    print("RRT* failed to find a path.")
    return []
