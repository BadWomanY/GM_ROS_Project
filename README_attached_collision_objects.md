# Attached Collision Objects for Grasped Parts

## Quick Start

This solution automatically handles collision detection for parts grasped by robot arms during path planning. When a part is grasped, it's automatically attached to the robot's end-effector as a collision object, ensuring safe path planning.

## How It Works

1. **Automatic Detection**: The system automatically detects when parts are grasped or released
2. **Service-Based Architecture**: Uses ROS services to avoid import conflicts between IsaacGym and MoveIt
3. **Dynamic Attachment**: Grasped parts are attached as collision objects to the robot's end-effector
4. **Safe Planning**: MoveIt considers both the robot and carried parts when planning paths
5. **Automatic Cleanup**: When parts are released, collision objects are automatically removed

## Usage

### Automatic Operation (Default)

The system works automatically - no manual intervention required:

```python
# Initialize your robot controllers as usual
arm2_controller = ArmController2(sim, task_lib, arm_idx=1)

# The system automatically handles collision objects during grasping
arm2_controller.step("grasp_l_part", cur_pose, grasp_pose, "plan", "auto")
arm2_controller.step("move_with_part", cur_pose, goal_pose, "plan", "auto")
arm2_controller.step("release_part", cur_pose, release_pose, "plan", "auto")
```

### Manual Control (For Testing)

```python
# Manually attach a part (for testing purposes)
arm2_controller._attach_part_to_gripper("L_part")

# Manually detach a part (for testing purposes)
arm2_controller._detach_part_from_gripper("L_part")

# Check current attached objects
print(arm2_controller.attached_objects)
```

## Testing

### Run the Test Script

```bash
# Terminal 1: Start the attached objects server
roslaunch isaac_ros_bridge attached_objects_server.launch

# Terminal 2: Start your robot simulation
roslaunch isaac_ros_bridge robot_simulation.launch

# Terminal 3: Run the test
cd catkin_ws/src/isaac_ros_bridge/scripts
python3 test_attached_collision_objects.py
```

### Verify in RViz

```bash
# Start RViz to visualize attached collision objects
rosrun rviz rviz

# Add the following displays:
# - RobotModel (to see the robot)
# - PlanningScene (to see collision objects)
# - TF (to see coordinate frames)
```

## Configuration

### Required MoveIt Setup

Ensure your MoveIt configuration includes:

1. **End-effector definition** in your SRDF:
```xml
<end_effector name="a2_ee" parent_link="a2_weld_tip" group="a2_tool" />
```

2. **Proper collision matrix** to exclude gripper self-collisions:
```xml
<disable_collisions link1="a2_panda_hand" link2="a2_link6" reason="adjacent" />
```

### Part Geometry

The system supports two part types:

- **L Part**: Complex geometry with main segment, short segment, and cube
- **R Part**: Simple geometry with main segment and cube

To add new part types, modify the `_attach_part_to_gripper()` method in `ArmController2`.

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

## Benefits

- **Safety**: Prevents collisions between carried parts and obstacles
- **Efficiency**: Better path planning when considering carried parts
- **Reliability**: Automatic operation with robust error handling
- **Compatibility**: Works with existing MoveIt infrastructure

## Files Modified/Created

### New Files
- `scripts/moveit_attached_objects_server.py`: MoveIt server for handling attached objects
- `srv/AttachPart.srv`: Service definition for attaching parts
- `srv/DetachPart.srv`: Service definition for detaching parts
- `launch/attached_objects_server.launch`: Launch file for the server

### Modified Files
- `robot_controller.py`: Updated to use ROS services instead of direct MoveIt imports
- `package.xml`: Added service dependencies
- `CMakeLists.txt`: Added service file definitions
- `test_attached_collision_objects.py`: Updated to test service-based approach
- `docs/attached_collision_objects_solution.md`: Updated documentation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed documentation in `docs/attached_collision_objects_solution.md`
3. Run the test script to verify functionality
4. Check MoveIt logs for detailed error messages
