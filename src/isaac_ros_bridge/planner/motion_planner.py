import torch
import random
import time
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

def line_collision_check(start, end, obstacles, safety_radius=0.0, num_checks=10):
    """
    Check if the line segment from start to end collides with obstacles
    """
    for i in range(num_checks + 1):
        t = i / num_checks
        point = (1 - t) * start + t * end
        if is_in_collision(point, obstacles, safety_radius):
            return True
    return False

def steer(from_point, to_point, step_size):
    """Steer from one point toward another with given step size"""
    direction = to_point - from_point
    dist = torch.norm(direction)
    if dist < step_size:
        return to_point
    return from_point + direction / dist * step_size

def adaptive_goal_bias(iteration, max_iter, base_bias=0.1):
    """Increase goal bias as iterations progress"""
    progress = iteration / max_iter
    return min(base_bias + 0.3 * progress, 0.5)

def sample_reachable_space(reachable_points, goal, goal_bias, device):
    """Smart sampling that balances exploration and exploitation"""
    if random.random() < goal_bias:
        return goal
    else:
        # Sample from reachable space
        sample_idx = random.randint(0, reachable_points.shape[0] - 1)
        return reachable_points[sample_idx]

def find_nearest_node(tree, sample_point):
    """Find the nearest node in the tree to the sample point"""
    if not tree:
        return None
    
    distances = torch.stack([torch.norm(node.pos - sample_point) for node in tree])
    nearest_idx = torch.argmin(distances)
    return tree[nearest_idx]

def rrt_plan(start: torch.Tensor, goal: torch.Tensor, reachable: torch.Tensor, obstacles: list,
             step_size=0.05, max_iter=1000, goal_thresh=0.05, device="cuda:0", safety_radius=0.15):
    """
    Improved RRT planner with adaptive parameters and better sampling
    
    Args:
        start, goal: (3,) torch.Tensor
        reachable: (N, 3) tensor of reachable positions
        obstacles: list of (low: (3,), high: (3,)) tuples
        step_size: maximum distance for each RRT extension
        max_iter: maximum number of iterations
        goal_thresh: distance threshold to consider goal reached
        device: torch device
        safety_radius: safety margin around obstacles
    
    Returns:
        list of (3,) tensors representing the path, or empty list if failed
    """
    
    # Remove fixed seeds for better exploration
    # Use time-based seed for reproducibility in debugging
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Validate inputs
    if torch.norm(start - goal) < goal_thresh:
        return [start, goal]
    
    # Check if start or goal are in collision
    if is_in_collision(start, obstacles, safety_radius):
        print(f"Start position {start} is in collision!")
        return []
    
    if is_in_collision(goal, obstacles, safety_radius):
        print(f"Goal position {goal} is in collision!")
        return []
    
    tree = [RRTNode(start)]
    
    # Adaptive parameters based on step size
    base_goal_bias = 0.05 if step_size < 0.03 else 0.1 if step_size < 0.06 else 0.15
    
    # Track progress for adaptive behavior
    last_progress_iter = 0
    best_distance_to_goal = torch.norm(start - goal)
    
    for iteration in range(max_iter):
        # Adaptive goal bias - increase over time and when stuck
        if iteration - last_progress_iter > 100:
            current_goal_bias = min(base_goal_bias * 2, 0.4)
        else:
            current_goal_bias = adaptive_goal_bias(iteration, max_iter, base_goal_bias)
        
        # Sample point
        sample = sample_reachable_space(reachable, goal, current_goal_bias, device)
        
        # Find nearest node
        nearest = find_nearest_node(tree, sample)
        if nearest is None:
            continue
        
        # Steer toward sample
        new_pos = steer(nearest.pos, sample, step_size)
        
        # Check for collision along the path
        if line_collision_check(nearest.pos, new_pos, obstacles, safety_radius):
            continue
        
        # Create new node
        new_node = RRTNode(new_pos, parent=nearest)
        tree.append(new_node)
        
        # Check if we're closer to goal
        dist_to_goal = torch.norm(new_node.pos - goal)
        if dist_to_goal < best_distance_to_goal:
            best_distance_to_goal = dist_to_goal
            last_progress_iter = iteration
        
        # Check if goal is reached
        if dist_to_goal < goal_thresh:
            # Try to connect directly to goal
            if not line_collision_check(new_node.pos, goal, obstacles, safety_radius):
                # Add goal node
                goal_node = RRTNode(goal, parent=new_node)
                
                # Trace back path
                path = []
                current = goal_node
                while current is not None:
                    path.append(current.pos)
                    current = current.parent
                path.reverse()
                
                print(f"RRT found path in {iteration + 1} iterations with step_size={step_size}")
                return path
        
        # Every 200 iterations, try different step sizes if stuck
        if iteration > 0 and iteration % 200 == 0:
            if iteration - last_progress_iter > 150:
                # Temporarily use different step size
                alt_step_size = step_size * (0.5 if step_size > 0.03 else 1.5)
                alt_new_pos = steer(nearest.pos, sample, alt_step_size)
                
                if not line_collision_check(nearest.pos, alt_new_pos, obstacles, safety_radius):
                    new_node = RRTNode(alt_new_pos, parent=nearest)
                    tree.append(new_node)
    
    print(f"RRT failed to find path after {max_iter} iterations with step_size={step_size}")
    print(f"Best distance to goal achieved: {best_distance_to_goal:.3f}")
    
    # Return partial path to closest point if no full path found
    if len(tree) > 1:
        closest_node = min(tree, key=lambda n: torch.norm(n.pos - goal))
        if torch.norm(closest_node.pos - goal) < best_distance_to_goal * 1.5:
            path = []
            current = closest_node
            while current is not None:
                path.append(current.pos)
                current = current.parent
            path.reverse()
            print(f"Returning partial path to closest point (dist: {torch.norm(closest_node.pos - goal):.3f})")
            return path
    
    return []

def rrt_plan_with_retry(start: torch.Tensor, goal: torch.Tensor, reachable: torch.Tensor, 
                       obstacles: list, step_size=0.05, max_retries=3, **kwargs):
    """
    RRT planner with automatic retry using different parameters
    """
    base_step_size = step_size
    
    for attempt in range(max_retries):
        if attempt == 0:
            # First attempt with original parameters
            current_step_size = base_step_size
            current_max_iter = kwargs.get('max_iter', 1000)
        elif attempt == 1:
            # Second attempt with smaller step size and more iterations
            current_step_size = base_step_size * 0.7
            current_max_iter = kwargs.get('max_iter', 1000) * 1.5
        else:
            # Final attempt with larger step size
            current_step_size = base_step_size * 1.4
            current_max_iter = kwargs.get('max_iter', 1000) * 2
        
        print(f"RRT attempt {attempt + 1}/{max_retries} with step_size={current_step_size:.3f}")
        
        # Update kwargs
        retry_kwargs = kwargs.copy()
        retry_kwargs['max_iter'] = int(current_max_iter)
        
        path = rrt_plan(start, goal, reachable, obstacles, 
                       step_size=current_step_size, **retry_kwargs)
        
        if path:
            return path
    
    print("All RRT retry attempts failed")
    return []