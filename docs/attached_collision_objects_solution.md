# Attached Collision Objects Solution for Grasped Parts

## Problem Statement

When planning paths for robot arms that are holding parts, the current system only considers collisions with the robot itself but not with the parts being carried. This can lead to:

1. **Collisions with obstacles**: The carried part may collide with tables, walls, or other objects
2. **Inter-arm collisions**: When multiple arms are working, one arm's carried part may collide with another arm
3. **Unsafe paths**: The planned trajectory may be valid for the robot but unsafe when considering the carried part

## Solution Overview

The solution implements **attached collision objects** in MoveIt, which dynamically attach grasped parts to the robot's end-effector during path planning. This ensures that:

1. **Complete collision detection**: MoveIt considers both the robot and any carried parts when planning paths
2. **Dynamic updates**: Collision objects are automatically attached when parts are grasped and detached when released
3. **Accurate representation**: The collision geometry matches the actual part dimensions and orientation
4. **Import conflict resolution**: Uses ROS services to avoid naming conflicts between IsaacGym and MoveIt
5. **Complete part geometry**: Full L_part geometry including main segment, short segment, and cube

## Implementation Details

### 1. Service-Based Architecture

To avoid import conflicts between IsaacGym and MoveIt, the solution uses a **service-based architecture**:

- **MoveIt Attached Objects Server**: A separate ROS node that handles MoveIt operations
- **ROS Services**: Communication between the robot controller and MoveIt server
- **No Direct Imports**: The robot controller never imports MoveIt directly

#### Architecture Diagram

```
┌─────────────────┐    ROS Services    ┌─────────────────────────┐
│ Robot Controller│◄─────────────────►│ MoveIt Attached Objects │
│ (IsaacGym)      │                   │ Server                  │
│                 │                   │ (MoveIt)                │
└─────────────────┘                   └─────────────────────────┘
```

### 2. Service Definitions

#### AttachPart Service
```yaml
# Request
string part_type
string gripper_frame
geometry_msgs/PoseStamped part_pose

# Response
bool success
string message
```

#### DetachPart Service
```yaml
# Request
string part_type

# Response
bool success
string message
```

### 3. Robot Controller Integration

#### ArmController2 (L and R Parts)

```python
class ArmController2:
    def __init__(self, sim, task_lib, arm_idx=1):
        # ... existing initialization ...
        
        # Service clients for attached collision objects
        self.attach_srv = None  # ROS service for attaching parts
        self.detach_srv = None  # ROS service for detaching parts
```

#### Key Methods

- `ensure_services_connected()`: Establishes connection to ROS services
- `_attach_part_to_gripper(part_type)`: Calls service to attach collision objects
- `_detach_part_from_gripper(part_type)`: Calls service to remove collision objects

### 4. MoveIt Server Implementation

The `moveit_attached_objects_server.py` handles all MoveIt operations:

```python
class MoveItAttachedObjectsServer:
    def __init__(self):
        # Initialize MoveIt planning scene interface
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        
        # ROS services
        rospy.Service('/attach_part', AttachPart, self.handle_attach_part)
        rospy.Service('/detach_part', DetachPart, self.handle_detach_part)
```

### 5. Complete Part Geometry Definition

#### L Part (Complex Geometry)
The L part consists of **three collision objects** in a single attached collision object:
- **Main segment**: 0.04 × 0.20 × 0.01 m (long rectangular part)
- **Short segment**: 0.04 × 0.10 × 0.01 m (perpendicular to main, rotated 90° around Z)
- **Small cube**: 0.04 × 0.04 × 0.04 m (attached to the part)

#### R Part (Simple Geometry)
The R part consists of **two collision objects**:
- **Main segment**: 0.04 × 0.20 × 0.01 m (long rectangular part)
- **Small cube**: 0.04 × 0.04 × 0.04 m (attached to the part)

### 6. Technical Fixes

#### Serialization Error Fix
The original implementation had a serialization error where `link_name` was not properly converted to a string:

```python
# Fixed implementation
attached_object.link_name = str(gripper_frame)  # Ensure it's a string
attached_object.touch_links = [str(gripper_frame)]  # Ensure it's a string
```

#### Complete Geometry Implementation
The server now creates complete part geometries with multiple primitives:

```python
def _create_part_collision_object(self, part_type, part_pose):
    if part_type == "L_part":
        collision_object.id = f"attached_{part_type}_complete"
        
        # Main segment (long part)
        collision_object.primitives.append(self._create_box_primitive(0.04, 0.20, 0.01))
        collision_object.primitive_poses.append(part_pose.pose)
        
        # Short segment (perpendicular to main, rotated 90° around Z)
        short_pose = self._create_short_segment_pose(part_pose.pose)
        collision_object.primitives.append(self._create_box_primitive(0.04, 0.10, 0.01))
        collision_object.primitive_poses.append(short_pose)
        
        # Small cube (attached to the part)
        cube_pose = self._create_cube_pose(part_pose.pose)
        collision_object.primitives.append(self._create_box_primitive(0.04, 0.04, 0.04))
        collision_object.primitive_poses.append(cube_pose)
```

### 7. Grasp State Detection

The system automatically detects when parts are grasped or released:

```python
# Check grasp state changes and handle attached collision objects
was_L_grasped = self.is_L_grasped
was_R_grasped = self.is_R_grasped

if hand_L_part_dist < 0.05 and gripper_width < 0.03:
    self.is_L_grasped = True
else:
    self.is_L_grasped = False

# Handle L part attachment/detachment
if self.is_L_grasped and not was_L_grasped:
    # Part was just grasped - attach collision object
    self._attach_part_to_gripper("L_part")
elif not self.is_L_grasped and was_L_grasped:
    # Part was just released - detach collision object
    self._detach_part_from_gripper("L_part")
```

## Usage

### 1. Start the System

```bash
# Terminal 1: Start the MoveIt attached objects server
roslaunch isaac_ros_bridge attached_objects_server.launch

# Terminal 2: Start your robot simulation
roslaunch isaac_ros_bridge robot_simulation.launch
```

### 2. Automatic Operation

The system works automatically - no manual intervention required:

```python
# Initialize your robot controllers as usual
arm2_controller = ArmController2(sim, task_lib, arm_idx=1)

# The system automatically handles collision objects during grasping
arm2_controller.step("grasp_l_part", cur_pose, grasp_pose, "plan", "auto")
arm2_controller.step("move_with_part", cur_pose, goal_pose, "plan", "auto")
arm2_controller.step("release_part", cur_pose, release_pose, "plan", "auto")
```

### 3. Manual Control (For Testing)

```python
# Manually attach a part (for testing purposes)
arm2_controller._attach_part_to_gripper("L_part")

# Manually detach a part (for testing purposes)
arm2_controller._detach_part_from_gripper("L_part")

# Check current attached objects
print(arm2_controller.attached_objects)
```

### 4. Direct Service Calls

You can also call the services directly:

```bash
# Attach L part
rosservice call /attach_part "part_type: 'L_part'
gripper_frame: 'a2_panda_hand'
part_pose:
  header:
    frame_id: 'a2_panda_hand'
  pose:
    position: {x: 0.0, y: 0.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"

# Detach L part
rosservice call /detach_part "part_type: 'L_part'"

# List attached objects
rosservice call /list_attached_objects
```

## Testing

### 1. Basic Functionality Test

```bash
# Start the attached objects server
roslaunch isaac_ros_bridge attached_objects_server.launch

# In another terminal, run the test
cd catkin_ws/src/isaac_ros_bridge/scripts
python3 test_attached_objects.py
```

### 2. Service Testing

```bash
# Test service availability
rosservice list | grep attach
rosservice list | grep detach

# Test service calls
rosservice call /attach_part "part_type: 'L_part'"
rosservice call /detach_part "part_type: 'L_part'"
```

### 3. Verify in RViz

```bash
# Start RViz to visualize attached collision objects
rosrun rviz rviz

# Add the following displays:
# - RobotModel (to see the robot)
# - PlanningScene (to see collision objects)
# - TF (to see coordinate frames)
```

## Configuration Requirements

### 1. MoveIt Configuration

Ensure your MoveIt configuration includes:

```yaml
# In your robot's moveit_config/urdf/robot.urdf.xacro
<gripper_link name="a2_panda_hand">
    <!-- Gripper geometry -->
</gripper_link>

# In your moveit_config/config/robot.srdf.xacro
<end_effector name="a2_ee" parent_link="a2_weld_tip" group="a2_tool" />
```

### 2. Planning Groups

The system uses the following planning groups:
- `a2_manipulator`: Main arm planning group
- `a2_panda_hand`: End-effector frame for attached objects

### 3. Collision Matrix

Configure collision detection to exclude gripper links from self-collision:

```yaml
# In your moveit_config/config/robot.srdf.xacro
<disable_collisions link1="a2_panda_hand" link2="a2_link6" reason="adjacent" />
<disable_collisions link1="a2_panda_hand" link2="a2_link5" reason="adjacent" />
```

## Benefits

### 1. Safety
- **Complete collision avoidance**: Prevents collisions between carried parts and obstacles
- **Inter-arm safety**: Prevents collisions between different arms' carried parts
- **Dynamic safety**: Automatically adapts to changing grasp states

### 2. Efficiency
- **Better path quality**: MoveIt can find more optimal paths when considering carried parts
- **Reduced replanning**: Fewer failed plans due to unexpected collisions
- **Faster execution**: More direct paths that avoid obstacles

### 3. Reliability
- **Automatic operation**: No manual intervention required
- **Robust detection**: Reliable grasp state detection
- **Error handling**: Graceful degradation if services are unavailable
- **Import conflict resolution**: No conflicts between IsaacGym and MoveIt

### 4. Compatibility
- **Service-based architecture**: Clean separation of concerns
- **Modular design**: Easy to extend and modify
- **ROS standard**: Uses standard ROS services and messages

## Troubleshooting

### Common Issues

1. **Services not available**
   - Check if the attached objects server is running: `rosnode list | grep attached`
   - Verify service availability: `rosservice list | grep attach`
   - Start the server: `roslaunch isaac_ros_bridge attached_objects_server.launch`

2. **Planning failures with attached objects**
   - Check that your MoveIt configuration has proper end-effector definitions
   - Verify collision matrix settings
   - Ensure gripper links are properly excluded from self-collision

3. **Import conflicts**
   - The service-based approach eliminates MoveIt import conflicts
   - Ensure IsaacGym and MoveIt run in separate processes

4. **Serialization errors**
   - Ensure all string fields are properly converted to strings
   - Check that ROS message types are correctly specified

### Debug Commands

```bash
# Check service status
rosservice call /attach_part "part_type: 'test'"
rosservice call /detach_part "part_type: 'test'"

# Check planning scene
rosservice call /get_planning_scene

# List attached objects
rosservice call /list_attached_objects

# Check TF transforms
rosrun tf tf_echo /world /a2_panda_hand

# Monitor collision detection
rostopic echo /move_group/display_planned_path
```

## Future Enhancements

### 1. Dynamic Geometry
- **Adaptive collision objects**: Adjust collision geometry based on part orientation
- **Deformable parts**: Handle flexible or deformable parts
- **Multi-part assemblies**: Handle complex assemblies with multiple components

### 2. Advanced Detection
- **Vision-based grasp detection**: Use camera feedback for more reliable grasp detection
- **Force feedback**: Use force sensors for grasp confirmation
- **Machine learning**: Learn optimal collision geometries from demonstrations

### 3. Performance Optimization
- **Lazy collision checking**: Only check collisions when necessary
- **Hierarchical collision detection**: Use different detail levels for different planning phases
- **Parallel planning**: Plan multiple arms simultaneously with collision awareness

### 4. Service Enhancements
- **Batch operations**: Attach/detach multiple parts at once
- **Pose updates**: Update attached object poses dynamically
- **Geometry customization**: Support for custom part geometries

## Conclusion

The service-based attached collision objects solution provides a robust, automatic, and efficient way to handle collision detection for grasped parts during path planning. It ensures safety, improves path quality, and integrates seamlessly with the existing MoveIt infrastructure while avoiding import conflicts with IsaacGym.

The implementation is designed to be:
- **Automatic**: No manual intervention required
- **Robust**: Handles errors gracefully
- **Efficient**: Minimal performance impact
- **Extensible**: Easy to add new part types
- **Compatible**: No import conflicts with IsaacGym
- **Complete**: Full part geometry representation

This solution addresses the core issue of collision detection for carried parts while maintaining compatibility with existing systems and providing a foundation for future enhancements.

## Files Modified/Created

### New Files
- `scripts/moveit_attached_objects_server.py`: MoveIt server for handling attached objects
- `srv/AttachPart.srv`: Service definition for attaching parts
- `srv/DetachPart.srv`: Service definition for detaching parts
- `launch/attached_objects_server.launch`: Launch file for the server
- `scripts/test_attached_objects.py`: Simple test script
- `docs/attached_collision_objects_solution.md`: Updated documentation

### Modified Files
- `robot_controller.py`: Updated to use ROS services instead of direct MoveIt imports
- `package.xml`: Added service dependencies
- `CMakeLists.txt`: Added service file definitions
