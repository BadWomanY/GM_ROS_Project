# Enhanced Collision Management for Grasping and Mating Operations

## Overview

This document describes the enhanced collision management system implemented for ArmController1 and ArmController2, which now use full ROS MoveIt planning like ArmController3, but with additional collision avoidance features for grasping and mating operations.

## Key Features

### 1. Dynamic Collision Object Management
- **Tables**: Static collision objects representing the two tables in front of arms 1 and 2
- **Parts**: Dynamic collision objects that are removed from the planning scene when grasped
- **Gripper Management**: Automatic gripper control based on task state

### 2. Grasp State Management
Each arm controller tracks the current grasp state:
- `approach`: Moving toward the part with open gripper
- `grasp`: Closing gripper to grasp the part
- `lift`: Lifting the part from the table
- `transport`: Moving the part to mating position
- `mate`: Performing the mating operation

### 3. Part Attachment Tracking
- **Arm 1**: Tracks `part_attached` for the big part
- **Arm 2**: Tracks `l_part_attached` and `r_part_attached` for L and R parts respectively

## Implementation Details

### ArmController1 (Big Part Grasping)

```python
class ArmController1:
    def __init__(self, sim, task_lib, arm_idx=0):
        # ... initialization ...
        self.grasp_state = "approach"
        self.part_attached = False
        
    def _set_moveit_boxes_param(self):
        # Always include tables
        # Include big_part only if not grasped
        if not self.part_attached:
            # Add big_part collision object
```

**Key Features:**
- Grasps the big part from table 0
- Removes big_part from collision scene when grasped
- Uses MoveIt planning with dynamic collision management

### ArmController2 (L and R Part Grasping)

```python
class ArmController2:
    def __init__(self, sim, task_lib, arm_idx=1):
        # ... initialization ...
        self.grasp_state = "approach"
        self.l_part_attached = False
        self.r_part_attached = False
        
    def _determine_part_type(self, task_name):
        # Determines whether task is for L or R part
        # Updates appropriate attachment state
```

**Key Features:**
- Grasps both L and R parts from table 1
- Tracks attachment state for each part separately
- Removes parts from collision scene when grasped
- Supports different gripper commands for each part

### Collision Avoidance Requirements

#### 1. Table Collision Avoidance
- Arms must avoid colliding with tables during approach and transport
- Tables are always present as static collision objects

#### 2. Part Collision Avoidance
- Arms must avoid colliding with parts during approach
- Parts are removed from collision scene once grasped
- Prevents self-collision during transport

#### 3. Gripper Collision Management
- Gripper fingers are excluded from collision detection during grasping
- Implemented through MoveIt configuration
- Prevents false collision detection during part manipulation

#### 4. Mating Collision Avoidance
- During mating operations, the two parts should not collide
- Parts are considered attached to their respective arms
- Collision avoidance ensures proper mating alignment

## Task State Detection

The system automatically detects task states based on task names:

```python
def _determine_grasp_state(self, task_name):
    if "grasp" in task_name.lower():
        if "approach" in task_name.lower():
            return "approach"
        elif "grasp" in task_name.lower():
            return "grasp"
        elif "lift" in task_name.lower():
            return "lift"
    elif "transport" in task_name.lower():
        return "transport"
    elif "mate" in task_name.lower():
        return "mate"
    return "approach"
```

## Gripper Control

Automatic gripper control based on grasp state:

```python
if self.grasp_state == "grasp":
    self._send_gripper(0.0)  # Close gripper
    self.part_attached = True
elif self.grasp_state == "approach":
    self._send_gripper(0.04)  # Open gripper
elif self.grasp_state == "mate":
    self._send_gripper(0.0)  # Keep closed during mating
```

## MoveIt Integration

### Planning Groups
- **Arm 1**: `a1_manipulator`
- **Arm 2**: `a2_manipulator`
- **Arm 3**: `a3_manipulator`

### Collision Object Updates
The system automatically updates collision objects when grasp states change:

```python
if new_grasp_state != self.grasp_state:
    self.grasp_state = new_grasp_state
    self._push_boxes_to_moveit()  # Update collision scene
```

## Testing

### Basic Table Test
```bash
cd catkin_ws/src/isaac_ros_bridge/scripts
python3 test_tables.py
```

### Enhanced Collision Test
```bash
cd catkin_ws/src/isaac_ros_bridge/scripts
python3 test_enhanced_collision.py
```

## Configuration Requirements

### MoveIt Configuration
Ensure your MoveIt configuration includes:
- Proper gripper link exclusions for collision detection
- Appropriate planning groups for each arm
- Collision matrix configuration for mating operations

### ROS Topics
Required topics for each arm:
- `/a1/rs007l_arm_controller/command` (Arm 1)
- `/a2/rs007l_arm_controller/command` (Arm 2)
- `/a3/rs007l_arm_controller/command` (Arm 3)
- `/a1/franka_gripper_controller/command` (Arm 1 gripper)
- `/a2/franka_gripper_controller/command` (Arm 2 gripper)
- `/a3/franka_gripper_controller/command` (Arm 3 gripper)

## Usage Example

```python
# Initialize controllers
arm1_controller = ArmController1(sim, task_lib, arm_idx=0)
arm2_controller = ArmController2(sim, task_lib, arm_idx=1)
arm3_controller = ArmController3(sim, task_lib, arm_idx=2)

# Execute grasping tasks
# Arm 1 grasps big part
arm1_controller.step("grasp_big_part_approach", cur_pose, approach_pose, "plan", "auto")
arm1_controller.step("grasp_big_part_grasp", cur_pose, grasp_pose, "plan", "auto")

# Arm 2 grasps L part
arm2_controller.step("grasp_l_part_approach", cur_pose, approach_pose, "plan", "auto")
arm2_controller.step("grasp_l_part_grasp", cur_pose, grasp_pose, "plan", "auto")

# Mating operation
arm1_controller.step("mate_big_part", cur_pose, mate_pose, "plan", "auto")
arm2_controller.step("mate_l_part", cur_pose, mate_pose, "plan", "auto")
```

## Benefits

1. **Safety**: Prevents collisions between arms, tables, and parts
2. **Efficiency**: Dynamic collision management reduces planning complexity
3. **Reliability**: Automatic state detection and gripper control
4. **Flexibility**: Supports different part types and grasping strategies
5. **Integration**: Seamless integration with existing MoveIt infrastructure

## Troubleshooting

### Common Issues

1. **Parts not removed from collision scene**
   - Check attachment state tracking
   - Verify task name parsing
   - Ensure `_push_boxes_to_moveit()` is called

2. **Gripper collision detection**
   - Verify MoveIt gripper link exclusions
   - Check gripper state management
   - Ensure proper gripper commands

3. **Planning failures**
   - Check collision object updates
   - Verify planning group configuration
   - Ensure proper pose transformations

### Debug Commands

```bash
# Check collision objects
rosservice call /get_planning_scene

# Update collision scene
rosservice call /update_boxes

# Check MoveIt status
rostopic echo /move_group/status
```
