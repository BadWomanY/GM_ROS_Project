#!/usr/bin/env python3

"""
MoveIt Attached Objects Server

This server handles attached collision objects for grasped parts via ROS services,
avoiding import conflicts with IsaacGym.
"""

import rospy
import sys
import tf
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from std_srvs.srv import Trigger, TriggerResponse
from isaac_ros_bridge.srv import AttachPart, DetachPart, AttachPartResponse, DetachPartResponse
from IPython import embed

class MoveItAttachedObjectsServer:
    def __init__(self):
        rospy.init_node('moveit_attached_objects_server')
        
        # Initialize MoveIt planning scene interface
        try:
            import moveit_commander
            self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
            rospy.sleep(0.5)  # Allow time for connection
            rospy.loginfo("Connected to MoveIt planning scene interface.")
            self.moveit_available = True
        except ImportError:
            rospy.logwarn("moveit_commander not available, attached collision objects disabled")
            self.scene = None
            self.moveit_available = False
        
        # TF listener for coordinate transformations
        self.listener = tf.TransformListener()
        
        # Track attached objects
        self.attached_objects = {}
        
        # ROS services
        rospy.Service('/attach_part', AttachPart, self.handle_attach_part)
        rospy.Service('/detach_part', DetachPart, self.handle_detach_part)
        rospy.Service('/list_attached_objects', Trigger, self.handle_list_objects)
        
        rospy.loginfo("MoveIt Attached Objects Server ready!")

    def handle_attach_part(self, req):
        """Handle part attachment request."""
        response = AttachPartResponse()
        
        if not self.moveit_available:
            response.success = False
            response.message = "MoveIt not available"
            return response
        
        try:
            part_type = req.part_type
            gripper_frame = req.gripper_frame
            part_pose = req.part_pose
            
            # Create attached collision object with proper link_name
            attached_object = self._create_attached_collision_object(part_type, gripper_frame, part_pose)
            
            # Get current planning scene first with retry
            get_scene_srv = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
            max_retries = 3
            current_scene = None
            
            for attempt in range(max_retries):
                try:
                    current_scene = get_scene_srv()
                    if current_scene.scene.robot_state.joint_state.name:
                        break
                    else:
                        rospy.logwarn(f"Attempt {attempt + 1}: Robot state not ready, retrying...")
                        rospy.sleep(0.5)
                except Exception as e:
                    rospy.logwarn(f"Attempt {attempt + 1}: Failed to get planning scene: {e}")
                    rospy.sleep(0.5)
            
            # Check if robot state has joint states after retries
            if not current_scene or not current_scene.scene.robot_state.joint_state.name:
                response.success = False
                response.message = "Robot state not available after retries - no joint states found"
                rospy.logerr("Found empty JointState message after retries")
                return response
            
            # Apply to planning scene
            apply_scene_srv = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
            
            scene_request = PlanningScene()
            scene_request.is_diff = True
            scene_request.robot_state = current_scene.scene.robot_state
            scene_request.robot_state.attached_collision_objects.append(attached_object)
            
            apply_response = apply_scene_srv(scene_request)
            
            if apply_response.success:
                # Track attached object
                self.attached_objects[part_type] = {
                    'id': attached_object.object.id,
                    'gripper_frame': gripper_frame,
                    'pose': part_pose
                }
                
                response.success = True
                response.message = f"Successfully attached {part_type} to {gripper_frame}"
                rospy.loginfo(f"Attached {part_type} to {gripper_frame}")
            else:
                response.success = False
                response.message = f"Failed to apply planning scene: {apply_response.error_message}"
                rospy.logerr(f"Failed to apply planning scene: {apply_response.error_message}")
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to attach part: {str(e)}"
            rospy.logerr(f"Failed to attach part: {e}")
        
        return response

    def handle_detach_part(self, req):
        """Handle part detachment request."""
        response = DetachPartResponse()
        
        if not self.moveit_available:
            response.success = False
            response.message = "MoveIt not available"
            return response
        
        try:
            part_type = req.part_type
            
            if part_type in self.attached_objects:
                obj_id = self.attached_objects[part_type]['id']
                
                # Get current planning scene first with retry
                get_scene_srv = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
                max_retries = 3
                current_scene = None
                
                for attempt in range(max_retries):
                    try:
                        current_scene = get_scene_srv()
                        if current_scene.scene.robot_state.joint_state.name:
                            break
                        else:
                            rospy.logwarn(f"Attempt {attempt + 1}: Robot state not ready, retrying...")
                            rospy.sleep(0.5)
                    except Exception as e:
                        rospy.logwarn(f"Attempt {attempt + 1}: Failed to get planning scene: {e}")
                        rospy.sleep(0.5)
                
                # Check if robot state has joint states after retries
                if not current_scene or not current_scene.scene.robot_state.joint_state.name:
                    response.success = False
                    response.message = "Robot state not available after retries - no joint states found"
                    rospy.logerr("Found empty JointState message after retries")
                    return response
                
                
                scene_request = PlanningScene()
                scene_request.is_diff = True
                scene_request.robot_state = current_scene.scene.robot_state
                scene_request.robot_state.is_diff = True

                # Remove from planning scene
                apply_scene_srv = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)
                
                
                # Create an empty attached collision object to remove it
                remove_object = AttachedCollisionObject()
                remove_object.object.id = obj_id
                remove_object.object.operation = CollisionObject.REMOVE
                remove_object.link_name = str(self.attached_objects[part_type]['gripper_frame'])
                
                scene_request.robot_state.attached_collision_objects.append(remove_object)

                # Also remove the object from the world
                world_remove_object = CollisionObject()
                world_remove_object.id = obj_id
                world_remove_object.operation = CollisionObject.REMOVE

                # Add to world
                scene_request.world.collision_objects.append(world_remove_object)
                
                apply_response = apply_scene_srv(scene_request)
                
                if apply_response.success:
                    del self.attached_objects[part_type]
                    response.success = True
                    response.message = f"Successfully detached {part_type}"
                    rospy.loginfo(f"Detached {part_type}")
                else:
                    response.success = False
                    response.message = f"Failed to apply planning scene: {apply_response.error_message}"
                    rospy.logerr(f"Failed to apply planning scene: {apply_response.error_message}")
            else:
                response.success = False
                response.message = f"Part {part_type} not found in attached objects"
                rospy.logwarn(f"Part {part_type} not found in attached objects")
                
        except Exception as e:
            response.success = False
            response.message = f"Failed to detach part: {str(e)}"
            rospy.logerr(f"Failed to detach part: {e}")
        
        return response

    def handle_list_objects(self, req):
        """List all currently attached objects."""
        response = TriggerResponse()
        
        if self.attached_objects:
            obj_list = ", ".join(self.attached_objects.keys())
            response.success = True
            response.message = f"Attached objects: {obj_list}"
            rospy.loginfo(f"Attached objects: {obj_list}")
        else:
            response.success = True
            response.message = "No attached objects"
            rospy.loginfo("No attached objects")
        
        return response

    def _create_part_collision_object(self, part_type, part_pose):
        """Create a collision object for the specified part type."""
        # Ensure pose has proper orientation
        if (part_pose.pose.orientation.x == 0.0 and 
            part_pose.pose.orientation.y == 0.0 and 
            part_pose.pose.orientation.z == 0.0 and 
            part_pose.pose.orientation.w == 0.0):
            part_pose.pose.orientation.w = 1.0
            rospy.logwarn("Empty quaternion found in pose message. Setting to neutral orientation.")
        
        collision_object = CollisionObject()
        collision_object.header.frame_id = part_pose.header.frame_id
        collision_object.header.stamp = rospy.Time.now()
        
        if part_type == "L_part":
            # L part consists of multiple components
            collision_object.id = f"attached_{part_type}_complete"

            # Small cube (attached to the part)
            collision_object.primitives.append(self._create_box_primitive(0.04, 0.04, 0.04))
            collision_object.primitive_poses.append(part_pose.pose)
            
            # Main segment (long part)
            long_pose = self._create_long_segment_pose(part_pose.pose)
            collision_object.primitives.append(self._create_box_primitive(0.04, 0.20, 0.01))
            collision_object.primitive_poses.append(long_pose)
            
            # Short segment (perpendicular to main, rotated 90° around Z)
            short_pose = self._create_short_segment_pose(part_pose.pose)
            collision_object.primitives.append(self._create_box_primitive(0.04, 0.10, 0.01))
            collision_object.primitive_poses.append(short_pose)
            
            
            
        elif part_type == "R_part":
            # R part is simpler - just main segment and cube
            collision_object.id = f"attached_{part_type}_complete"
            
            # Main segment
            long_pose = self._create_long_segment_pose(part_pose.pose)
            collision_object.primitives.append(self._create_box_primitive(0.04, 0.20, 0.01))
            collision_object.primitive_poses.append(long_pose)
            
            # Small cube
            collision_object.primitives.append(self._create_box_primitive(0.04, 0.04, 0.04))
            collision_object.primitive_poses.append(part_pose.pose)
            
        elif part_type == "big_part":
            # Big part (base plate)
            # Small cube
            collision_object.primitives.append(self._create_box_primitive(0.04, 0.04, 0.04))
            collision_object.primitive_poses.append(part_pose.pose)


            collision_object.id = f"attached_{part_type}"
            plate_pose = self._create_long_segment_pose(part_pose.pose)
            collision_object.primitives.append(self._create_box_primitive(0.30, 0.30, 0.01))
            collision_object.primitive_poses.append(plate_pose)
            
        else:
            # Default box for unknown part types
            collision_object.id = f"attached_{part_type}"
            collision_object.primitives.append(self._create_box_primitive(0.05, 0.05, 0.05))
            collision_object.primitive_poses.append(part_pose.pose)
        
        return collision_object

    def _create_box_primitive(self, x, y, z):
        """Create a box primitive with the specified dimensions."""
        from shape_msgs.msg import SolidPrimitive
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [x, y, z]
        return box

    def _create_short_segment_pose(self, base_pose):
        """Create pose for the short segment of L part (perpendicular to main)."""
        import tf.transformations as tf_trans
        
        # Create a copy of the base pose
        short_pose = Pose()
        short_pose.position.x = base_pose.position.x - 0.07  # Offset in X
        short_pose.position.y = base_pose.position.y + 0.08  # Offset in Y
        short_pose.position.z = base_pose.position.z + 0.025
        
        # Rotate 90 degrees around Z axis relative to base pose
        base_quat = [base_pose.orientation.x, base_pose.orientation.y, 
                    base_pose.orientation.z, base_pose.orientation.w]
        z_rot_90 = tf_trans.quaternion_from_euler(0, 0, -1.5708)  # 90 degrees in radians
        short_quat = tf_trans.quaternion_multiply(base_quat, z_rot_90)
        
        short_pose.orientation.x = short_quat[0]
        short_pose.orientation.y = short_quat[1]
        short_pose.orientation.z = short_quat[2]
        short_pose.orientation.w = short_quat[3]
        
        return short_pose

    def _create_long_segment_pose(self, base_pose):
        """Create pose for the long segment attached to the part."""
        long_pose = Pose()
        long_pose.position.x = base_pose.position.x
        long_pose.position.y = base_pose.position.y
        long_pose.position.z = base_pose.position.z + 0.025  # Offset in Z
        long_pose.orientation.x = base_pose.orientation.x
        long_pose.orientation.y = base_pose.orientation.y
        long_pose.orientation.z = base_pose.orientation.z
        long_pose.orientation.w = base_pose.orientation.w
        
        return long_pose

    def _create_attached_collision_object(self, part_type, gripper_frame, part_pose):
        """Create an attached collision object message."""
        attached_object = AttachedCollisionObject()
        attached_object.link_name = str(gripper_frame)  # Ensure it's a string
        attached_object.object = self._create_part_collision_object(part_type, part_pose)
        attached_object.touch_links = [str(gripper_frame)]  # Allow collision with gripper
        return attached_object

def main():
    try:
        server = MoveItAttachedObjectsServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Server interrupted by user")
    except Exception as e:
        rospy.logerr(f"Server failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
