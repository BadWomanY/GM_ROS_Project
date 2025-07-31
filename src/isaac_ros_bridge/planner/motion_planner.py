import torch
import random
from IPython import embed

class RRTNode:
    def __init__(self, pos: torch.Tensor, parent=None):
        self.pos = pos  # shape: (3,)
        self.parent = parent

def is_in_collision(point, obstacles, safety_radius=0.0):
    """
    Point: (3,)
    obstacles: list of (low, high)
    safety_radius: float, how far to keep away from obstacles
    """
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
