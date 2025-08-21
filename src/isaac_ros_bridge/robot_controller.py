from isaacgym.torch_utils import *
from isaac_ros_bridge.utils.franka_utils import *
from isaac_ros_bridge.arm_control import simple_controller, weld_controller
from isaac_ros_bridge.planner.motion_planner import rrt_star_plan, rrt_plan
from isaac_ros_bridge.models.spot_weld_offsets import L_part_offset, R_part_offset
from geometry_msgs.msg import PoseStamped
from isaac_ros_bridge.srv import PlanPose
import torch

# ROS service imports for attached collision objects
from isaac_ros_bridge.srv import AttachPart, DetachPart

# === MoveIt IK Interface ===
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import rospy
import tf
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

class ArmController1:
    def __init__(self, sim, task_lib, arm_idx=0):
        # ---------- sim / bookkeeping ----------
        self.sim = sim
        self.arm_idx = arm_idx
        self.device = sim.device
        self.arm_dof = sim.robot_dof
        self.task_name = None
        self.task_lib = task_lib
        self.plan_srv = None
        self.traj = None
        self.is_big_part_grasped = False

        # ---------- ROS comms ----------
        self.joint_names = [f"a1_joint{i+1}" for i in range(self.arm_dof)]
        self.ros_pub = rospy.Publisher("/a1/rs007l_arm_controller/command", JointTrajectory, queue_size=1)
        self.tool_pub = rospy.Publisher("/a1/franka_gripper_controller/command", JointTrajectory, queue_size=1)
        self.listener = tf.TransformListener()

        # live buffers for real robot state
        self.q_real = torch.zeros(self.arm_dof, dtype=torch.float32, device=self.device)

        # Attached collision objects for grasped parts
        self.attached_objects = {}  # Track attached collision objects
        self.attach_srv = None  # ROS service for attaching parts
        self.detach_srv = None  # ROS service for detaching parts

        rospy.Subscriber("/joint_states", JointState, self._joint_state_cb, queue_size=1)

    # ---------- callbacks ----------
    def _joint_state_cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        indices = [name_to_idx[jn] for jn in self.joint_names if jn in name_to_idx]
        if len(indices) == self.arm_dof:
            self.q_real = torch.tensor([msg.position[i] for i in indices], dtype=torch.float32, device=self.device)

    # ---------- util ----------
    def ensure_moveit_connected(self):
        """Connect to the MoveIt planning service only when needed."""
        if self.plan_srv is None:
            rospy.loginfo("Connecting to /plan_pose service...")
            rospy.wait_for_service("/plan_pose")
            self.plan_srv = rospy.ServiceProxy("/plan_pose", PlanPose)
            rospy.loginfo("Connected to MoveIt planner service.")

    def ensure_services_connected(self):
        """Connect to the attached collision objects services."""
        if self.attach_srv is None:
            try:
                rospy.wait_for_service('/attach_part', timeout=2.0)
                self.attach_srv = rospy.ServiceProxy('/attach_part', AttachPart)
                rospy.loginfo("Connected to /attach_part service.")
            except rospy.ROSException:
                rospy.logwarn("Timeout waiting for /attach_part service; attached collision objects disabled")
                self.attach_srv = None
        
        if self.detach_srv is None:
            try:
                rospy.wait_for_service('/detach_part', timeout=2.0)
                self.detach_srv = rospy.ServiceProxy('/detach_part', DetachPart)
                rospy.loginfo("Connected to /detach_part service.")
            except rospy.ROSException:
                rospy.logwarn("Timeout waiting for /detach_part service; attached collision objects disabled")
                self.detach_srv = None

    def _attach_part_to_gripper(self, part_type):
        """Attach a grasped part to the robot's end-effector for collision detection."""
        if self.attach_srv is None:
            rospy.logwarn("Attach service not available")
            return
            
        try:
            # Get current part and hand1_tip poses from simulation
            if part_type == "big_part":
                part_pos = self.sim.big_part_pos[0]  # Current part position
                part_rot = self.sim.big_part_rot[0]  # Current part rotation
            else:
                rospy.logwarn(f"Unknown part type: {part_type}")
                return
            
            # Get current hand1_tip pose (this is where the part is grasped)
            hand1_tip_pos = self.sim.hand1_tip_pos  # Hand tip position
            
            # Get current hand1 orientation (this is the actual gripper orientation)
            hand1_rot = self.sim.hand1_rot[0]  # Current hand1 rotation
            
            # Convert part position to hand1_tip frame
            part_pos_hand1_tip = part_pos - hand1_tip_pos
            part_pos_hand1_tip[2] += 0.019  # Small offset for attachment
            part_rot_hand1_tip = quat_mul(part_rot, quat_conjugate(hand1_rot))
            
            # Create pose for the part in hand1_tip frame
            part_pose = PoseStamped()
            part_pose.header.frame_id = "a1_weld_tip"
            part_pose.header.stamp = rospy.Time.now()
            
            # Set the calculated relative pose
            part_pose.pose.position.x = part_pos_hand1_tip[0].item()
            part_pose.pose.position.y = part_pos_hand1_tip[1].item()
            part_pose.pose.position.z = part_pos_hand1_tip[2].item()
            part_pose.pose.orientation.x = part_rot_hand1_tip[0].item()
            part_pose.pose.orientation.y = part_rot_hand1_tip[1].item()
            part_pose.pose.orientation.z = part_rot_hand1_tip[2].item()
            part_pose.pose.orientation.w = part_rot_hand1_tip[3].item()
            
            rospy.loginfo(f"Attaching {part_type} with relative pose in hand1_tip frame: pos=({part_pos_hand1_tip[0]:.3f}, {part_pos_hand1_tip[1]:.3f}, {part_pos_hand1_tip[2]:.3f}), rot=({part_rot_hand1_tip[0]:.3f}, {part_rot_hand1_tip[1]:.3f}, {part_rot_hand1_tip[2]:.3f}, {part_rot_hand1_tip[3]:.3f})")
            
            # Call the attach service
            response = self.attach_srv(part_type, "a1_weld_tip", part_pose)
            
            if response.success:
                rospy.loginfo(f"Successfully attached {part_type} to hand1_tip")
                self.attached_objects[part_type] = True
            else:
                rospy.logwarn(f"Failed to attach {part_type}: {response.message}")
                
        except Exception as e:
            rospy.logwarn(f"Failed to attach {part_type} to hand1_tip: {e}")

    def _detach_part_from_gripper(self, part_type):
        """Detach a part from the robot's end-effector."""
        if self.detach_srv is None:
            rospy.logwarn("Detach service not available")
            return
            
        try:
            # Call the detach service
            response = self.detach_srv(part_type)
            
            if response.success:
                rospy.loginfo(f"Successfully detached {part_type} from gripper")
                if part_type in self.attached_objects:
                    del self.attached_objects[part_type]
            else:
                rospy.logwarn(f"Failed to detach {part_type}: {response.message}")
                
        except Exception as e:
            rospy.logwarn(f"Failed to detach {part_type} from gripper: {e}")

    def plan_with_moveit(self, goal_pos_xyz, goal_quat_xyzw, frame="world", group="a1_manipulator"):
        self.ensure_moveit_connected()

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = frame
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = goal_pos_xyz
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = goal_quat_xyzw

        resp = self.plan_srv(goal, group)
        if not resp.success:
            raise RuntimeError(f"MoveIt planning failed: {resp.message}")
        return resp.traj  # trajectory_msgs/JointTrajectory

    def execute_traj(self, traj):
        self.ros_pub.publish(traj)

    def _set_moveit_boxes_param(self):
        # Build boxes in world frame using dictionary for deduplication
        boxes_dict = {}

        # Table for Arm 1 (table 0)
        table0_center = torch.tensor([0.5, 0.7, 0.15], device=self.device)  # From sim_manager.py
        table0_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # Identity quaternion
        table0_size = [0.6, 1.0, 0.3]  # From sim_manager.py table_dims

        boxes_dict["table0"] = {
            "id": "table0",
            "size": table0_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": table0_center[0].item(), "y": table0_center[1].item(), "z": table0_center[2].item()},
                "orientation": {"x": table0_quat[0].item(), "y": table0_quat[1].item(), "z": table0_quat[2].item(), "w": table0_quat[3].item()},
            },
        }

        # Big part (base plate) - only add if not grasped
        big_center = self.sim.big_part_pos[0]
        big_quat = self.sim.big_part_rot[0]
        big_size = [0.30, 0.30, 0.01]
        if not self.is_big_part_grasped:
            boxes_dict["big_part"] = {
                "id": "big_part",
                "size": big_size,
                "pose": {
                    "frame_id": "world",
                    "position": {"x": big_center.tolist()[0], "y": big_center.tolist()[1], "z": big_center.tolist()[2]},
                    "orientation": {"x": big_quat.tolist()[0], "y": big_quat.tolist()[1], "z": big_quat.tolist()[2], "w": big_quat.tolist()[3]},
                },
            }

            # Small cube near big part
            cube_center_sim = (self.sim.big_part_pos + quat_rotate(
                self.sim.big_part_rot, torch.tensor([[0,0,0.025]], device=self.device)
            ))[0]
            cube_center = cube_center_sim
            cube_quat = self.sim.big_part_rot[0]
            cube_size = [0.04, 0.04, 0.04]

            boxes_dict["big_part_cube"] = {
                "id": "big_part_cube",
                "size": cube_size,
                "pose": {
                    "frame_id": "world",
                    "position": {"x": cube_center.tolist()[0], "y": cube_center.tolist()[1], "z": cube_center.tolist()[2]},
                    "orientation": {"x": cube_quat.tolist()[0], "y": cube_quat.tolist()[1], "z": cube_quat.tolist()[2], "w": cube_quat.tolist()[3]},
                },
            }

        L_center = self.sim.L_part_pos[0]
        L_quat = self.sim.L_part_rot[0]
        L_long = [0.04, 0.20, 0.01]

        boxes_dict["l_part"] = {
            "id": "l_part",
            "size": L_long,
            "pose": {
                "frame_id": "world",
                "position": {"x": L_center.tolist()[0], "y": L_center.tolist()[1], "z": L_center.tolist()[2]},
                "orientation": {"x": L_quat.tolist()[0], "y": L_quat.tolist()[1], "z": L_quat.tolist()[2], "w": L_quat.tolist()[3]},
            },
        }

        # L part short segment
        L_short = self.sim.L_part_pos[0] + quat_rotate(self.sim.L_part_rot, torch.tensor([[0.07,-0.08,0.0]], device=self.device))[0]
        z_rot_90 = torch.tensor([0.0, 0.0, 0.7071, 0.7071], device=self.device)  # 90 deg around z
        L_short_quat = quat_mul(self.sim.L_part_rot[0], z_rot_90)
        L_short_size = [0.04, 0.1, 0.01]

        boxes_dict["l_short_part"] = {
            "id": "l_short_part",
            "size": L_short_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": L_short.tolist()[0], "y": L_short.tolist()[1], "z": L_short.tolist()[2]},
                "orientation": {"x": L_short_quat.tolist()[0], "y": L_short_quat.tolist()[1], "z": L_short_quat.tolist()[2], "w": L_short_quat.tolist()[3]},
            },
        }

        # Small cube near L part
        cube_center_sim = (self.sim.L_part_pos + quat_rotate(
            self.sim.L_part_rot, torch.tensor([[0,0,0.025]], device=self.device)
        ))[0]
        cube_center = cube_center_sim
        cube_quat = self.sim.L_part_rot[0]
        cube_size = [0.04, 0.04, 0.04]

        boxes_dict["small_cube"] = {
            "id": "small_cube",
            "size": cube_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": cube_center.tolist()[0], "y": cube_center.tolist()[1], "z": cube_center.tolist()[2]},
                "orientation": {"x": cube_quat.tolist()[0], "y": cube_quat.tolist()[1], "z": cube_quat.tolist()[2], "w": cube_quat.tolist()[3]},
            },
        }

        # R part
        R_center = self.sim.R_part_pos[0]
        R_quat = self.sim.R_part_rot[0]
        R_size = [0.04, 0.20, 0.01]

        boxes_dict["r_part"] = {
            "id": "r_part",
            "size": R_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": R_center.tolist()[0], "y": R_center.tolist()[1], "z": R_center.tolist()[2]},
                "orientation": {"x": R_quat.tolist()[0], "y": R_quat.tolist()[1], "z": R_quat.tolist()[2], "w": R_quat.tolist()[3]},
            },
        }

        # Small cube near R part
        R_cube_center_sim = (self.sim.R_part_pos + quat_rotate(
            self.sim.R_part_rot, torch.tensor([[0,0,0.02]], device=self.device)
        ))[0]
        R_cube_center = R_cube_center_sim
        R_cube_quat = self.sim.R_part_rot[0]
        cube_size = [0.04, 0.04, 0.04]

        boxes_dict["r_small_cube"] = {
            "id": "r_small_cube",
            "size": cube_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": R_cube_center.tolist()[0], "y": R_cube_center.tolist()[1], "z": R_cube_center.tolist()[2]},
                "orientation": {"x": R_cube_quat.tolist()[0], "y": R_cube_quat.tolist()[1], "z": R_cube_quat.tolist()[2], "w": R_cube_quat.tolist()[3]},
            },
        }

        # Convert dictionary back to list for ROS parameter
        boxes = list(boxes_dict.values())
        rospy.set_param("/isaac/moveit_boxes", boxes)

    def _push_boxes_to_moveit(self):
        self._set_moveit_boxes_param()
        try:
            rospy.wait_for_service("/update_boxes", timeout=1.0)
            upd = rospy.ServiceProxy("/update_boxes", Trigger)
            resp = upd()
            rospy.loginfo(f"/update_boxes -> success={resp.success}, msg={resp.message}")
        except rospy.ROSException:
            rospy.logwarn("Timeout waiting for /update_boxes; boxes will sync at planning time.")

    def _send_gripper(self, width: float, duration_s: float = 0.1):
        """
        width: finger separation per joint (you use two independent prismatic joints).
            0.04 -> open, 0.00 -> close
        """
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

        p0 = JointTrajectoryPoint()
        p0.positions = [None, None]  # controller will treat as start-at-current if omitted
        p0.time_from_start = rospy.Duration(0.0)

        p1 = JointTrajectoryPoint()
        p1.positions = [width, width]
        p1.time_from_start = rospy.Duration(duration_s)

        jt.points = [p1]  # send single timed point is fine; controller ramps to it
        self.tool_pub.publish(jt)

    # ---------- main step ----------
    def step(self, task_name, cur_pose, goal_pose, plan_mode, gripper_mode):
        if not self.traj or self.task_name != task_name:
            # print(f"Arm1: {task_name}")
            self.traj = None
            self.task_name = task_name

            # Ensure services connection for attached collision objects
            self.ensure_services_connected()

            # Push boxes to MoveIt
            self._push_boxes_to_moveit()

            # Plan and execute trajectory
            self.traj = self.plan_with_moveit(goal_pose[0, :3].cpu().numpy(), goal_pose[0, 3:].cpu().numpy())
            self.execute_traj(self.traj)
        
        hand_tip_pos = self.sim.hand1_tip_pos
        big_part_pos = self.sim.big_part_pos[0]
        hand_big_part_dist = torch.norm(hand_tip_pos - big_part_pos, dim=-1).item()
        gripper_pos = self.sim.pos_action[:, self.arm_idx, self.arm_dof:]
        gripper_width = gripper_pos[:, 0].item()
        
        # Check grasp state changes and handle attached collision objects
        was_big_part_grasped = self.is_big_part_grasped
        
        if hand_big_part_dist < 0.05 and gripper_width < 0.03:
            self.is_big_part_grasped = True
        else:
            self.is_big_part_grasped = False
        
        # Handle big part attachment/detachment
        if self.is_big_part_grasped and not was_big_part_grasped:
            # Part was just grasped - attach collision object
            rospy.loginfo("Attaching big part to gripper")
            self._attach_part_to_gripper("big_part")
        elif not self.is_big_part_grasped and was_big_part_grasped:
            # Part was just released - detach collision object
            rospy.loginfo("Detaching big part from gripper")
            self._detach_part_from_gripper("big_part")
        
        if gripper_mode == "auto":
            final_joint_state_err = torch.norm(torch.tensor(self.traj.points[-1].positions, device=self.device) - self.q_real).item()
            if final_joint_state_err == 0.0:
                self._send_gripper(0.0)
                gripper_act = torch.zeros((1, 2), device=self.device)
            else:
                self._send_gripper(0.04)
                gripper_act = torch.ones((1, 2), device=self.device) * 0.04
        elif gripper_mode == "close":
            self._send_gripper(0.0)
            gripper_act = torch.zeros((1, 2), device=self.device)
        else:
            self._send_gripper(0.04)
            gripper_act = torch.ones((1, 2), device=self.device) * 0.04

        arm1_action = torch.tensor(self.q_real, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.sim.pos_action[:, self.arm_idx, :self.arm_dof] = arm1_action
        self.sim.pos_action[:, self.arm_idx, self.arm_dof:] = gripper_act


class ArmController2:
    def __init__(self, sim, task_lib, arm_idx=1):
        # ---------- sim / bookkeeping ----------
        self.sim = sim
        self.arm_idx = arm_idx
        self.device = sim.device
        self.arm_dof = sim.robot_dof
        self.task_name = None
        self.task_lib = task_lib
        self.plan_srv = None
        self.traj = None
        self.is_L_grasped = False
        self.is_R_grasped = False

        # ---------- ROS comms ----------
        self.joint_names = [f"a2_joint{i+1}" for i in range(self.arm_dof)]
        self.ros_pub = rospy.Publisher("/a2/rs007l_arm_controller/command", JointTrajectory, queue_size=1)
        self.tool_pub = rospy.Publisher("/a2/franka_gripper_controller/command", JointTrajectory, queue_size=1)
        self.listener = tf.TransformListener()

        # live buffers for real robot state
        self.q_real = torch.zeros(self.arm_dof, dtype=torch.float32, device=self.device)

        # Attached collision objects for grasped parts
        self.attached_objects = {}  # Track attached collision objects
        self.attach_srv = None  # ROS service for attaching parts
        self.detach_srv = None  # ROS service for detaching parts

        rospy.Subscriber("/joint_states", JointState, self._joint_state_cb, queue_size=1)

    # ---------- callbacks ----------
    def _joint_state_cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        indices = [name_to_idx[jn] for jn in self.joint_names if jn in name_to_idx]
        if len(indices) == self.arm_dof:
            self.q_real = torch.tensor([msg.position[i] for i in indices], dtype=torch.float32, device=self.device)

    # ---------- util ----------
    def ensure_moveit_connected(self):
        """Connect to the MoveIt planning service only when needed."""
        if self.plan_srv is None:
            rospy.loginfo("Connecting to /plan_pose service...")
            rospy.wait_for_service("/plan_pose")
            self.plan_srv = rospy.ServiceProxy("/plan_pose", PlanPose)
            rospy.loginfo("Connected to MoveIt planner service.")

    def ensure_services_connected(self):
        """Connect to the attached collision objects services."""
        if self.attach_srv is None:
            try:
                rospy.wait_for_service('/attach_part', timeout=2.0)
                self.attach_srv = rospy.ServiceProxy('/attach_part', AttachPart)
                rospy.loginfo("Connected to /attach_part service.")
            except rospy.ROSException:
                rospy.logwarn("Timeout waiting for /attach_part service; attached collision objects disabled")
                self.attach_srv = None
        
        if self.detach_srv is None:
            try:
                rospy.wait_for_service('/detach_part', timeout=2.0)
                self.detach_srv = rospy.ServiceProxy('/detach_part', DetachPart)
                rospy.loginfo("Connected to /detach_part service.")
            except rospy.ROSException:
                rospy.logwarn("Timeout waiting for /detach_part service; attached collision objects disabled")
                self.detach_srv = None

    def _attach_part_to_gripper(self, part_type):
        """Attach a grasped part to the robot's end-effector for collision detection."""
        if self.attach_srv is None:
            rospy.logwarn("Attach service not available")
            return
            
        try:
            # Get current part and weld_tip poses from simulation
            if part_type == "L_part":
                part_pos = self.sim.L_part_pos[0]  # Current part position
                part_rot = self.sim.L_part_rot[0]  # Current part rotation
            elif part_type == "R_part":
                part_pos = self.sim.R_part_pos[0]  # Current part position
                part_rot = self.sim.R_part_rot[0]  # Current part rotation
            elif part_type == "big_part":
                part_pos = self.sim.big_part_pos[0]  # Current part position
                part_rot = self.sim.big_part_rot[0]  # Current part rotation
            else:
                rospy.logwarn(f"Unknown part type: {part_type}")
                return
            
            # Get current weld_tip pose (this is where the part is grasped)
            hand2_tip_pos = self.sim.hand2_tip_pos  # Weld tip position
            
            # Get current weld_tip orientation (this is the actual gripper orientation)
            weld_tip_rot = self.sim.hand2_rot[0]  # Current weld tip rotation
            
            # Convert part position to weld_tip frame
            part_pos_weld_tip = part_pos - hand2_tip_pos
            part_pos_weld_tip[2] += 0.019
            # To keep the part in the same orientation as world frame when attached,
            # we need to calculate what the world frame orientation looks like in the weld_tip frame
            # The world frame orientation in weld_tip frame is the inverse of weld_tip orientation
            # world_orientation_in_weld_tip = quat_conjugate(quat_mul(weld_tip_rot, part_rot))
            # part_rot_weld_tip = world_orientation_in_weld_tip
            part_rot_weld_tip = quat_mul(part_rot, quat_conjugate(weld_tip_rot))
            
            # Create pose for the part in weld_tip frame
            part_pose = PoseStamped()
            part_pose.header.frame_id = "a2_weld_tip"
            part_pose.header.stamp = rospy.Time.now()
            
            # Set the calculated relative pose
            part_pose.pose.position.x = part_pos_weld_tip[0].item()
            part_pose.pose.position.y = part_pos_weld_tip[1].item()
            part_pose.pose.position.z = part_pos_weld_tip[2].item()
            part_pose.pose.orientation.x = part_rot_weld_tip[0].item()
            part_pose.pose.orientation.y = part_rot_weld_tip[1].item()
            part_pose.pose.orientation.z = part_rot_weld_tip[2].item()
            part_pose.pose.orientation.w = part_rot_weld_tip[3].item()
            
            rospy.loginfo(f"Attaching {part_type} with relative pose in weld_tip frame: pos=({part_pos_weld_tip[0]:.3f}, {part_pos_weld_tip[1]:.3f}, {part_pos_weld_tip[2]:.3f}), rot=({part_rot_weld_tip[0]:.3f}, {part_rot_weld_tip[1]:.3f}, {part_rot_weld_tip[2]:.3f}, {part_rot_weld_tip[3]:.3f})")
            
            # Call the attach service
            response = self.attach_srv(part_type, "a2_weld_tip", part_pose)
            
            if response.success:
                rospy.loginfo(f"Successfully attached {part_type} to weld_tip")
                self.attached_objects[part_type] = True
            else:
                rospy.logwarn(f"Failed to attach {part_type}: {response.message}")
                
        except Exception as e:
            rospy.logwarn(f"Failed to attach {part_type} to weld_tip: {e}")

    def _detach_part_from_gripper(self, part_type):
        """Detach a part from the robot's end-effector."""
        if self.detach_srv is None:
            rospy.logwarn("Detach service not available")
            return
            
        try:
            # Call the detach service
            response = self.detach_srv(part_type)
            
            if response.success:
                rospy.loginfo(f"Successfully detached {part_type} from gripper")
                if part_type in self.attached_objects:
                    del self.attached_objects[part_type]
            else:
                rospy.logwarn(f"Failed to detach {part_type}: {response.message}")
                
        except Exception as e:
            rospy.logwarn(f"Failed to detach {part_type} from gripper: {e}")

    def plan_with_moveit(self, goal_pos_xyz, goal_quat_xyzw, frame="world", group="a2_manipulator"):
        self.ensure_moveit_connected()

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = frame
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = goal_pos_xyz
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = goal_quat_xyzw

        resp = self.plan_srv(goal, group)
        if not resp.success:
            raise RuntimeError(f"MoveIt planning failed: {resp.message}")
        return resp.traj  # trajectory_msgs/JointTrajectory

    def execute_traj(self, traj):
        self.ros_pub.publish(traj)

    def _set_moveit_boxes_param(self):
        # Build boxes in world frame using dictionary for deduplication
        boxes_dict = {}

        # Table for Arm 2 (table 1)
        table1_center = torch.tensor([0.5, -0.7, 0.15], device=self.device)  # From sim_manager.py
        table1_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # Identity quaternion
        table1_size = [0.6, 1.0, 0.3]  # From sim_manager.py table_dims

        boxes_dict["table1"] = {
            "id": "table1",
            "size": table1_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": table1_center[0].item(), "y": table1_center[1].item(), "z": table1_center[2].item()},
                "orientation": {"x": table1_quat[0].item(), "y": table1_quat[1].item(), "z": table1_quat[2].item(), "w": table1_quat[3].item()},
            },
        }

        # Big part (base plate)
        big_center = self.sim.big_part_anchor
        big_quat = self.sim.big_part_goal_quat[0]
        big_size = [0.30, 0.30, 0.01]

        boxes_dict["big_part_goal"] = {
            "id": "big_part_goal",
            "size": big_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": big_center.tolist()[0], "y": big_center.tolist()[1], "z": big_center.tolist()[2]},
                "orientation": {"x": big_quat.tolist()[0], "y": big_quat.tolist()[1], "z": big_quat.tolist()[2], "w": big_quat.tolist()[3]},
            },
        }

        big_center_cur = self.sim.big_part_pos[0]
        big_quat_cur = self.sim.big_part_rot[0]
        big_size = [0.30, 0.30, 0.01]

        boxes_dict["big_part"] = {
            "id": "big_part",
            "size": big_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": big_center_cur.tolist()[0], "y": big_center_cur.tolist()[1], "z": big_center_cur.tolist()[2]},
                "orientation": {"x": big_quat_cur.tolist()[0], "y": big_quat_cur.tolist()[1], "z": big_quat_cur.tolist()[2], "w": big_quat_cur.tolist()[3]},
            },
        }

        # Small cube near big part
        big_cube_center_sim = (self.sim.big_part_pos + quat_rotate(
            self.sim.big_part_rot, torch.tensor([[0,0,0.025]], device=self.device)
        ))[0]
        big_cube_center = big_cube_center_sim
        big_cube_quat = self.sim.big_part_rot[0]
        big_cube_size = [0.04, 0.04, 0.04]

        boxes_dict["big_part_cube"] = {
            "id": "big_part_cube",
            "size": big_cube_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": big_cube_center.tolist()[0], "y": big_cube_center.tolist()[1], "z": big_cube_center.tolist()[2]},
                "orientation": {"x": big_cube_quat.tolist()[0], "y": big_cube_quat.tolist()[1], "z": big_cube_quat.tolist()[2], "w": big_cube_quat.tolist()[3]},
            },
        }

        # L part - only add if not grasped (to avoid collision with gripper)
        if not self.is_L_grasped:
            L_center = self.sim.L_part_pos[0]
            L_quat = self.sim.L_part_rot[0]
            L_long = [0.04, 0.20, 0.01]

            boxes_dict["l_part"] = {
                "id": "l_part",
                "size": L_long,
                "pose": {
                    "frame_id": "world",
                    "position": {"x": L_center.tolist()[0], "y": L_center.tolist()[1], "z": L_center.tolist()[2]},
                    "orientation": {"x": L_quat.tolist()[0], "y": L_quat.tolist()[1], "z": L_quat.tolist()[2], "w": L_quat.tolist()[3]},
                },
            }

            # L part short segment
            L_short = self.sim.L_part_pos[0] + quat_rotate(self.sim.L_part_rot, torch.tensor([[0.07,-0.08,0.0]], device=self.device))[0]
            z_rot_90 = torch.tensor([0.0, 0.0, 0.7071, 0.7071], device=self.device)  # 90 deg around z
            L_short_quat = quat_mul(self.sim.L_part_rot[0], z_rot_90)
            L_short_size = [0.04, 0.1, 0.01]

            boxes_dict["l_short_part"] = {
                "id": "l_short_part",
                "size": L_short_size,
                "pose": {
                    "frame_id": "world",
                    "position": {"x": L_short.tolist()[0], "y": L_short.tolist()[1], "z": L_short.tolist()[2]},
                    "orientation": {"x": L_short_quat.tolist()[0], "y": L_short_quat.tolist()[1], "z": L_short_quat.tolist()[2], "w": L_short_quat.tolist()[3]},
                },
            }

            # Small cube near L part
            cube_center_sim = (self.sim.L_part_pos + quat_rotate(
                self.sim.L_part_rot, torch.tensor([[0,0,0.025]], device=self.device)
            ))[0]
            cube_center = cube_center_sim
            cube_quat = self.sim.L_part_rot[0]
            cube_size = [0.04, 0.04, 0.04]

            boxes_dict["small_cube"] = {
                "id": "small_cube",
                "size": cube_size,
                "pose": {
                    "frame_id": "world",
                    "position": {"x": cube_center.tolist()[0], "y": cube_center.tolist()[1], "z": cube_center.tolist()[2]},
                    "orientation": {"x": cube_quat.tolist()[0], "y": cube_quat.tolist()[1], "z": cube_quat.tolist()[2], "w": cube_quat.tolist()[3]},
                },
            }

        if not self.is_R_grasped:
            # R part
            R_center = self.sim.R_part_pos[0]
            R_quat = self.sim.R_part_rot[0]
            R_size = [0.04, 0.20, 0.01]

            boxes_dict["r_part"] = {
                "id": "r_part",
                "size": R_size,
                "pose": {
                    "frame_id": "world",
                    "position": {"x": R_center.tolist()[0], "y": R_center.tolist()[1], "z": R_center.tolist()[2]},
                    "orientation": {"x": R_quat.tolist()[0], "y": R_quat.tolist()[1], "z": R_quat.tolist()[2], "w": R_quat.tolist()[3]},
                },
            }

            # Small cube near R part
            R_cube_center_sim = (self.sim.R_part_pos + quat_rotate(
                self.sim.R_part_rot, torch.tensor([[0,0,0.02]], device=self.device)
            ))[0]
            R_cube_center = R_cube_center_sim
            R_cube_quat = self.sim.R_part_rot[0]
            cube_size = [0.04, 0.04, 0.04]

            boxes_dict["r_small_cube"] = {
                "id": "r_small_cube",
                "size": cube_size,
                "pose": {
                    "frame_id": "world",
                    "position": {"x": R_cube_center.tolist()[0], "y": R_cube_center.tolist()[1], "z": R_cube_center.tolist()[2]},
                    "orientation": {"x": R_cube_quat.tolist()[0], "y": R_cube_quat.tolist()[1], "z": R_cube_quat.tolist()[2], "w": R_cube_quat.tolist()[3]},
                },
            }

        # Convert dictionary back to list for ROS parameter
        boxes = list(boxes_dict.values())
        rospy.set_param("/isaac/moveit_boxes", boxes)

    def _push_boxes_to_moveit(self):
        self._set_moveit_boxes_param()
        try:
            rospy.wait_for_service("/update_boxes", timeout=1.0)
            upd = rospy.ServiceProxy("/update_boxes", Trigger)
            resp = upd()
            rospy.loginfo(f"/update_boxes -> success={resp.success}, msg={resp.message}")
        except rospy.ROSException:
            rospy.logwarn("Timeout waiting for /update_boxes; boxes will sync at planning time.")

    def _send_gripper(self, width: float, duration_s: float = 0.1):
        """
        width: finger separation per joint (you use two independent prismatic joints).
            0.04 -> open, 0.00 -> close
        """
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

        p0 = JointTrajectoryPoint()
        p0.positions = [None, None]  # controller will treat as start-at-current if omitted
        p0.time_from_start = rospy.Duration(0.0)

        p1 = JointTrajectoryPoint()
        p1.positions = [width, width]
        p1.time_from_start = rospy.Duration(duration_s)

        jt.points = [p1]  # send single timed point is fine; controller ramps to it
        self.tool_pub.publish(jt)

    # ---------- main step ----------
    def step(self, task_name, cur_pose, goal_pose, plan_mode, gripper_mode):
        if not self.traj or self.task_name != task_name:
            # print(f"Arm2: {task_name}")
            self.traj = None
            self.task_name = task_name

            # Ensure services connection for attached collision objects
            self.ensure_services_connected()

            # Push boxes to MoveIt
            self._push_boxes_to_moveit()

            # Plan and execute trajectory
            self.traj = self.plan_with_moveit(goal_pose[0, :3].cpu().numpy(), goal_pose[0, 3:].cpu().numpy())
            self.execute_traj(self.traj)
        
        hand_tip_pos = self.sim.hand2_tip_pos
        L_part_pos = self.sim.L_part_pos[0]
        R_part_pos = self.sim.R_part_pos[0]
        hand_L_part_dist = torch.norm(hand_tip_pos - L_part_pos, dim=-1).item()
        hand_R_part_dist = torch.norm(hand_tip_pos - R_part_pos, dim=-1).item()
        gripper_pos = self.sim.pos_action[:, self.arm_idx, self.arm_dof:]
        gripper_width = gripper_pos[:, 0].item()
        
        # Check grasp state changes and handle attached collision objects
        was_L_grasped = self.is_L_grasped
        was_R_grasped = self.is_R_grasped
        
        if hand_L_part_dist < 0.05 and gripper_width < 0.03:
            self.is_L_grasped = True
        else:
            self.is_L_grasped = False
        if hand_R_part_dist < 0.05 and gripper_width < 0.03:
            self.is_R_grasped = True
        else:
            self.is_R_grasped = False
        
        # Handle L part attachment/detachment
        if self.is_L_grasped and not was_L_grasped:
            # Part was just grasped - attach collision object
            rospy.loginfo("Attaching L part to gripper")
            self._attach_part_to_gripper("L_part")
        elif not self.is_L_grasped and was_L_grasped:
            # Part was just released - detach collision object
            rospy.loginfo("Detaching L part from gripper")
            self._detach_part_from_gripper("L_part")
        
        # Handle R part attachment/detachment
        if self.is_R_grasped and not was_R_grasped:
            # Part was just grasped - attach collision object
            rospy.loginfo("Attaching R part to gripper")
            self._attach_part_to_gripper("R_part")
        elif not self.is_R_grasped and was_R_grasped:
            # Part was just released - detach collision object
            rospy.loginfo("Detaching R part from gripper")
            self._detach_part_from_gripper("R_part")
        
        if gripper_mode == "auto":
            final_joint_state_err = torch.norm(torch.tensor(self.traj.points[-1].positions, device=self.device) - self.q_real).item()
            if final_joint_state_err == 0.0 and (hand_L_part_dist > 0.03 or hand_R_part_dist > 0.03):
                self._send_gripper(0.0)
                gripper_act = torch.zeros((1, 2), device=self.device)
            else:
                self._send_gripper(0.04)
                gripper_act = torch.ones((1, 2), device=self.device) * 0.04
        elif gripper_mode == "close":
            self._send_gripper(0.0)
            gripper_act = torch.zeros((1, 2), device=self.device)
        else:
            self._send_gripper(0.04)
            gripper_act = torch.ones((1, 2), device=self.device) * 0.04

        arm2_action = torch.tensor(self.q_real, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.sim.pos_action[:, self.arm_idx, :self.arm_dof] = arm2_action
        self.sim.pos_action[:, self.arm_idx, self.arm_dof:] = gripper_act


class ArmController3:
    def __init__(self, sim, task_lib, arm_idx=2):
        # ---------- sim / bookkeeping ----------
        self.sim = sim
        self.arm_idx = arm_idx
        self.device = sim.device
        self.arm_dof = sim.robot_dof
        self.task_name = None
        self.task_lib = task_lib
        self.weld_timer = 0.0
        self.real_timer = 0.0
        self.plan_srv = None
        self.traj = None

        # ---------- ROS comms ----------
        self.joint_names = [f"a3_joint{i+1}" for i in range(self.arm_dof)]
        self.ros_pub = rospy.Publisher("/a3/rs007l_arm_controller/command", JointTrajectory, queue_size=1)
        self.tool_pub = rospy.Publisher("/a3/franka_gripper_controller/command", JointTrajectory, queue_size=1)
        self.listener = tf.TransformListener()

        # live buffers for real robot state
        self.q_real = torch.zeros(self.arm_dof, dtype=torch.float32, device=self.device)

        rospy.Subscriber("/joint_states", JointState, self._joint_state_cb, queue_size=1)

    # ---------- callbacks ----------
    def _joint_state_cb(self, msg):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        indices = [name_to_idx[jn] for jn in self.joint_names if jn in name_to_idx]
        if len(indices) == self.arm_dof:
            self.q_real = torch.tensor([msg.position[i] for i in indices], dtype=torch.float32, device=self.device)

    # ---------- util ----------
    def ensure_moveit_connected(self):
        """Connect to the MoveIt planning service only when needed."""
        if self.plan_srv is None:
            rospy.loginfo("Connecting to /plan_pose service...")
            rospy.wait_for_service("/plan_pose")
            self.plan_srv = rospy.ServiceProxy("/plan_pose", PlanPose)
            rospy.loginfo("Connected to MoveIt planner service.")

    def plan_with_moveit(self, goal_pos_xyz, goal_quat_xyzw, frame="world", group="a3_manipulator"):
        self.ensure_moveit_connected()

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = frame
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = goal_pos_xyz
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = goal_quat_xyzw

        resp = self.plan_srv(goal, group)
        if not resp.success:
            raise RuntimeError(f"MoveIt planning failed: {resp.message}")
        return resp.traj  # trajectory_msgs/JointTrajectory

    def execute_traj(self, traj):
        self.ros_pub.publish(traj)

    def _set_moveit_boxes_param(self):
        # Build boxes in world frame using dictionary for deduplication
        boxes_dict = {}

        # Example: big part (use center pose + quaternion you already have)
        big_center = self.sim.big_part_pos[0]
        big_quat = self.sim.big_part_rot[0]
        
        big_size   = [0.30, 0.30, 0.01]                         # replace with your actual extents

        boxes_dict["big_part"] = {
            "id": "big_part",
            "size": big_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": big_center.tolist()[0], "y": big_center.tolist()[1], "z": big_center.tolist()[2]},
                "orientation": {"x": big_quat.tolist()[0], "y": big_quat.tolist()[1], "z": big_quat.tolist()[2], "w": big_quat.tolist()[3]},
            },
        }

        # Example: L part (or small part) as a simple box around it
        L_center = self.sim.L_part_pos[0]
        L_quat = self.sim.L_part_rot[0]
        L_long   = [0.04, 0.20, 0.01]                           # tune

        boxes_dict["l_part"] = {
            "id": "l_part",
            "size": L_long,
            "pose": {
                "frame_id": "world",
                "position": {"x": L_center.tolist()[0], "y": L_center.tolist()[1], "z": L_center.tolist()[2]},
                "orientation": {"x": L_quat.tolist()[0], "y": L_quat.tolist()[1], "z": L_quat.tolist()[2], "w": L_quat.tolist()[3]},
            },
        }

        L_short = self.sim.L_part_pos[0] + quat_rotate(self.sim.L_part_rot, torch.tensor([[0.07,-0.08,0.0]], device=self.device))[0]
        # Rotate 90 degrees around z-axis relative to L_part orientation
        z_rot_90 = torch.tensor([0.0, 0.0, 0.7071, 0.7071], device=self.device)  # 90 deg around z
        L_short_quat = quat_mul(self.sim.L_part_rot[0], z_rot_90)
        L_short_size = [0.04, 0.1, 0.01]

        boxes_dict["l_short_part"] = {
            "id": "l_short_part",
            "size": L_short_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": L_short.tolist()[0], "y": L_short.tolist()[1], "z": L_short.tolist()[2]},
                "orientation": {"x": L_short_quat.tolist()[0], "y": L_short_quat.tolist()[1], "z": L_short_quat.tolist()[2], "w": L_short_quat.tolist()[3]},
            },
        }

        R_center = self.sim.R_part_pos[0]
        R_quat = self.sim.R_part_rot[0]
        R_size   = [0.04, 0.20, 0.01]                           # tune

        boxes_dict["r_part"] = {
            "id": "r_part",
            "size": R_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": R_center.tolist()[0], "y": R_center.tolist()[1], "z": R_center.tolist()[2]},
                "orientation": {"x": R_quat.tolist()[0], "y": R_quat.tolist()[1], "z": R_quat.tolist()[2], "w": R_quat.tolist()[3]},
            },
        }

        # Example: small cube near L part (your previous cube_center)
        cube_center_sim = (self.sim.L_part_pos + quat_rotate(
            self.sim.L_part_rot, torch.tensor([[0,0,0.02]], device=self.device)
        ))[0]
        cube_center = cube_center_sim
        cube_quat = self.sim.L_part_rot[0]
        cube_size = [0.04, 0.04, 0.04]

        boxes_dict["small_cube"] = {
            "id": "small_cube",
            "size": cube_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": cube_center.tolist()[0], "y": cube_center.tolist()[1], "z": cube_center.tolist()[2]},
                "orientation": {"x": cube_quat.tolist()[0], "y": cube_quat.tolist()[1], "z": cube_quat.tolist()[2], "w": cube_quat.tolist()[3]},
            },
        }

        R_cube_center_sim = (self.sim.R_part_pos + quat_rotate(
            self.sim.R_part_rot, torch.tensor([[0,0,0.02]], device=self.device)
        ))[0]
        R_cube_center = R_cube_center_sim
        R_cube_quat = self.sim.R_part_rot[0]
        cube_size = [0.04, 0.04, 0.04]

        boxes_dict["r_small_cube"] = {
            "id": "r_small_cube",
            "size": cube_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": R_cube_center.tolist()[0], "y": R_cube_center.tolist()[1], "z": R_cube_center.tolist()[2]},
                "orientation": {"x": R_cube_quat.tolist()[0], "y": R_cube_quat.tolist()[1], "z": R_cube_quat.tolist()[2], "w": R_cube_quat.tolist()[3]},
            },
        }

        # Convert dictionary back to list for ROS parameter
        boxes = list(boxes_dict.values())
        rospy.set_param("/isaac/moveit_boxes", boxes)

    def _push_boxes_to_moveit(self):
        self._set_moveit_boxes_param()
        try:
            rospy.wait_for_service("/update_boxes", timeout=1.0)
            upd = rospy.ServiceProxy("/update_boxes", Trigger)
            resp = upd()
            # rospy.loginfo(f"/update_boxes -> success={resp.success}, msg={resp.message}")
        except rospy.ROSException:
            rospy.logwarn("Timeout waiting for /update_boxes; boxes will sync at planning time.")

    def _send_gripper(self, width: float, duration_s: float = 0.1):
        """
        width: finger separation per joint (you use two independent prismatic joints).
            0.04 -> open, 0.00 -> close
        """
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

        p0 = JointTrajectoryPoint()
        # read current finger pos from /joint_states if you store it; fallback to current command:
        # here we just start at current q_real end segments if you mirror them; safe to omit
        p0.positions = [None, None]  # controller will treat as start-at-current if omitted
        p0.time_from_start = rospy.Duration(0.0)

        p1 = JointTrajectoryPoint()
        p1.positions = [width, width]
        p1.time_from_start = rospy.Duration(duration_s)

        jt.points = [p1]  # send single timed point is fine; controller ramps to it
        self.tool_pub.publish(jt)
    # ---------- main step ----------
    def step(self, task_name, cur_pose, goal_pose, plan_mode, gripper_mode):
        if not self.traj or self.task_name != task_name:
            # print(task_name)
            self.traj = None
            self.task_name = task_name

            self._push_boxes_to_moveit()

            if self.weld_timer>=2.0 and self.real_timer >= 2.0: 
                self.weld_timer = 0.0
                self.real_timer = 0.0
                self._send_gripper(0.04)
                gripper_act = torch.ones((1, 2), device=self.device) * 0.04

            self.traj = self.plan_with_moveit(goal_pose[0, :3].cpu().numpy(), goal_pose[0, 3:].cpu().numpy())
            self.current_plan = True
            # self.traj = traj.points
            self.execute_traj(self.traj)
        
        if gripper_mode == "auto" and self.real_timer < 2.0:
            final_joint_state_err = torch.norm(torch.tensor(self.traj.points[-1].positions, device=self.device) - self.q_real).item()
            if final_joint_state_err == 0.0:
                self._send_gripper(0.02)
                gripper_act = torch.ones((1, 2), device=self.device) * 0.02
                self.real_timer += self.sim.sim_params.dt
                self.weld_timer += self.sim.sim_params.dt
            else:
                self._send_gripper(0.04)
                gripper_act = torch.ones((1, 2), device=self.device) * 0.04
        else:
            self._send_gripper(0.04)
            gripper_act = torch.ones((1, 2), device=self.device) * 0.04

        # self._weld_controller()
        arm3_action = torch.tensor(self.q_real, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.sim.pos_action[:, self.arm_idx, :self.arm_dof] = arm3_action
        self.sim.pos_action[:, self.arm_idx, self.arm_dof:] = gripper_act

        return self.weld_timer, self.real_timer