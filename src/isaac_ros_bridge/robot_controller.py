from isaacgym.torch_utils import *
from isaac_ros_bridge.utils.franka_utils import *
from isaac_ros_bridge.arm_control import simple_controller, weld_controller
from isaac_ros_bridge.planner.motion_planner import rrt_star_plan, rrt_plan
from isaac_ros_bridge.models.spot_weld_offsets import L_part_offset, R_part_offset
from geometry_msgs.msg import PoseStamped
from isaac_ros_bridge.srv import PlanPose
import torch

# === MoveIt IK Interface ===
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import rospy
import tf
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

class ArmController1:
    def __init__(self, sim, task_lib, arm_idx=0):
        self.sim = sim
        self.arm_idx = arm_idx
        self.device = sim.device
        self.waypoints = []
        self.waypoint_idx = torch.zeros(sim.num_envs, dtype=torch.long, device=self.device)
        self.arm_dof = sim.robot_dof
        self.robot_mids = sim.robot_mids
        self.prev_task_name = None
        self.task_name = None
        self.task_lib = task_lib

    def step(self, task_name, cur_pose, goal_pose, plan_mode, gripper_mode):
        if not self.waypoints or self.task_name != task_name:
            self.waypoints = []
            self.waypoint_idx = torch.zeros(self.sim.num_envs, dtype=torch.long, device=self.device)
            if self.task_name == None:
                self.prev_task_name = task_name
            else:
                self.prev_task_name = self.task_name
            self.task_name = task_name
            start = cur_pose[:, :3]
            goal = goal_pose[:, :3]
            if plan_mode == 'plan':
                mid = elevated_midpoint(start[0], goal[0], z_offset=0.18)
                obstacles = [(
                    torch.tensor([-0.9, -0.8, 0.0], device=self.device),
                    torch.tensor([-0.3, -0.2, 0.3], device=self.device)
                )]

                path_a = rrt_plan(start[0], mid, self.sim.reachable_pos, obstacles, step_size=0.07, goal_thresh=0.01, device=self.device)
                path_b = rrt_plan(mid, goal[0], self.sim.reachable_pos, obstacles, step_size=0.07, goal_thresh=0.01, device=self.device)
                full_path = path_a + path_b[1:]

                dense_path = interpolate_waypoints(full_path, step=0.01)
                self.waypoints = [pt.unsqueeze(0).repeat(self.sim.num_envs, 1) for pt in dense_path]
                self.waypoints.append(goal)
            else:
                self.waypoints.append(goal)

        
        prev_rot = self.task_lib[self.prev_task_name][1][:, 3:]
        hand_pos = cur_pose[:, :3]
        hand_rot = cur_pose[:, 3:]
        goal_rot = goal_pose[:, 3:]
        self.waypoint_idx, self.sim.pos_action = simple_controller(hand_pos, hand_rot, self.waypoints, 
                                                                    goal_rot, prev_rot,
                                                                    self.waypoint_idx, self.sim.j_eef1, 
                                                                    self.sim.dof_pos, self.sim.pos_action, 
                                                                    self.robot_mids, self.arm_idx, 
                                                                    self.sim.num_envs, self.arm_dof, plan_mode,
                                                                    self.sim.big_part_pos, self.sim.grasp_offset, gripper_mode)

class ArmController2:
    def __init__(self, sim, task_lib, arm_idx=1):
        self.sim = sim
        self.arm_idx = arm_idx
        self.device = sim.device
        self.waypoints = []
        self.waypoint_idx = torch.zeros(sim.num_envs, dtype=torch.long, device=self.device)
        self.arm_dof = sim.robot_dof
        self.robot_mids = sim.robot_mids
        self.task_name = None
        self.task_lib = task_lib

    def step(self, task_name, cur_pose, goal_pose, plan_mode, gripper_mode):
        if not self.waypoints or self.task_name != task_name:
            self.waypoints = []
            self.waypoint_idx = torch.zeros(self.sim.num_envs, dtype=torch.long, device=self.device)
            if self.task_name == None:
                self.prev_task_name = task_name
            else:
                self.prev_task_name = self.task_name
            self.task_name = task_name
            start = cur_pose[:, :3]
            goal = goal_pose[:, :3]
            if plan_mode == "plan":
                mid = elevated_midpoint(start[0], goal[0], z_offset=0.18)
                obstacles = [(
                    torch.tensor([-0.9, -0.8, 0.0], device=self.device),
                    torch.tensor([-0.3, -0.2, 0.3], device=self.device)
                )]

                path_a = rrt_plan(start[0], mid, self.sim.reachable_pos2, obstacles, step_size=0.07, goal_thresh=0.01, device=self.device)
                path_b = rrt_plan(mid, goal[0], self.sim.reachable_pos2, obstacles, step_size=0.07, goal_thresh=0.01, device=self.device)
                full_path = path_a + path_b[1:]

                dense_path = interpolate_waypoints(full_path, step=0.01)
                self.waypoints = [pt.unsqueeze(0).repeat(self.sim.num_envs, 1) for pt in dense_path]
                self.waypoints.append(goal)
            else:
                self.waypoints.append(goal)
        hand_pos = cur_pose[:, :3]
        hand_rot = cur_pose[:, 3:]
        goal_rot = goal_pose[:, 3:]
        if int(task_name[-1]) <= 2:
            part_pos = self.sim.L_part_pos
        else:
            part_pos = self.sim.R_part_pos
        prev_rot = self.task_lib[self.prev_task_name][1][:, 3:]
        self.waypoint_idx, self.sim.pos_action = simple_controller(hand_pos, hand_rot, self.waypoints, 
                                                                    goal_rot, prev_rot,
                                                                    self.waypoint_idx, self.sim.j_eef2, 
                                                                    self.sim.dof_pos, self.sim.pos_action, 
                                                                    self.robot_mids, self.arm_idx, 
                                                                    self.sim.num_envs, self.arm_dof, plan_mode,
                                                                    part_pos, self.sim.grasp_offset, gripper_mode)

class ArmController3:
    def __init__(self, sim, task_lib, arm_idx=2):
        # ---------- sim / bookkeeping ----------
        self.sim = sim
        self.arm_idx = arm_idx
        self.device = sim.device
        self.waypoints = []
        self.waypoint_idx = torch.zeros(sim.num_envs, dtype=torch.long, device=self.device)
        self.ros_waypoint_idx = torch.zeros(sim.num_envs, dtype=torch.long, device=self.device)
        self.arm_dof = sim.robot_dof
        self.robot_mids = sim.robot_mids
        self.task_name = None
        self.task_lib = task_lib
        self.prev_task_goal = None
        self.weld_timer = 0.0
        self.real_timer = 0.0
        self.waypoint_mode = []
        self.plan_srv = None
        self.traj = None

        # ---------- ROS comms ----------
        self.joint_names = [f"a3_joint{i+1}" for i in range(self.arm_dof)]
        self.ros_pub = rospy.Publisher("/a3/rs007l_arm_controller/command", JointTrajectory, queue_size=1)
        self.tool_pub = rospy.Publisher("/a3/franka_gripper_controller/command", JointTrajectory, queue_size=1)
        self.listener = tf.TransformListener()

        # live buffers for real robot state
        self.ee_pos_real = torch.zeros(3, device=self.device)
        self.ee_rot_real = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device)
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
        # Build boxes in world frame
        boxes = []

        # Example: big part (use center pose + quaternion you already have)
        big_center = self.sim.big_part_pos[0]
        big_quat = self.sim.big_part_rot[0]
        
        big_size   = [0.30, 0.30, 0.01]                         # replace with your actual extents

        boxes.append({
            "id": "big_part",
            "size": big_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": big_center.tolist()[0], "y": big_center.tolist()[1], "z": big_center.tolist()[2]},
                "orientation": {"x": big_quat.tolist()[0], "y": big_quat.tolist()[1], "z": big_quat.tolist()[2], "w": big_quat.tolist()[3]},
            },
        })

        # Example: L part (or small part) as a simple box around it
        L_center = self.sim.L_part_pos[0]
        L_quat = self.sim.L_part_rot[0]
        L_long   = [0.04, 0.20, 0.01]                           # tune

        boxes.append({
            "id": "l_part",
            "size": L_long,
            "pose": {
                "frame_id": "world",
                "position": {"x": L_center.tolist()[0], "y": L_center.tolist()[1], "z": L_center.tolist()[2]},
                "orientation": {"x": L_quat.tolist()[0], "y": L_quat.tolist()[1], "z": L_quat.tolist()[2], "w": L_quat.tolist()[3]},
            },
        })

        L_short = self.sim.L_part_pos[0] + quat_rotate(self.sim.L_part_rot, torch.tensor([[0.07,-0.08,0.0]], device=self.device))[0]
        # Rotate 90 degrees around z-axis relative to L_part orientation
        z_rot_90 = torch.tensor([0.0, 0.0, 0.7071, 0.7071], device=self.device)  # 90 deg around z
        L_short_quat = quat_mul(self.sim.L_part_rot[0], z_rot_90)
        L_short_size = [0.04, 0.1, 0.01]

        boxes.append({
            "id": "l_short_part",
            "size": L_short_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": L_short.tolist()[0], "y": L_short.tolist()[1], "z": L_short.tolist()[2]},
                "orientation": {"x": L_short_quat.tolist()[0], "y": L_short_quat.tolist()[1], "z": L_short_quat.tolist()[2], "w": L_short_quat.tolist()[3]},
            },
        })

        R_center = self.sim.R_part_pos[0]
        R_quat = self.sim.R_part_rot[0]
        R_size   = [0.04, 0.20, 0.01]                           # tune

        boxes.append({
            "id": "r_part",
            "size": R_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": R_center.tolist()[0], "y": R_center.tolist()[1], "z": R_center.tolist()[2]},
                "orientation": {"x": R_quat.tolist()[0], "y": R_quat.tolist()[1], "z": R_quat.tolist()[2], "w": R_quat.tolist()[3]},
            },
        })

        # Example: small cube near L part (your previous cube_center)
        cube_center_sim = (self.sim.L_part_pos + quat_rotate(
            self.sim.L_part_rot, torch.tensor([[0,0,0.03]], device=self.device)
        ))[0]
        cube_center = cube_center_sim
        cube_quat = self.sim.L_part_rot[0]
        cube_size = [0.04, 0.04, 0.04]

        boxes.append({
            "id": "small_cube",
            "size": cube_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": cube_center.tolist()[0], "y": cube_center.tolist()[1], "z": cube_center.tolist()[2]},
                "orientation": {"x": cube_quat.tolist()[0], "y": cube_quat.tolist()[1], "z": cube_quat.tolist()[2], "w": cube_quat.tolist()[3]},
            },
        })

        R_cube_center_sim = (self.sim.R_part_pos + quat_rotate(
            self.sim.R_part_rot, torch.tensor([[0,0,0.03]], device=self.device)
        ))[0]
        R_cube_center = R_cube_center_sim
        R_cube_quat = self.sim.R_part_rot[0]
        cube_size = [0.04, 0.04, 0.04]

        boxes.append({
            "id": "r_small_cube",
            "size": cube_size,
            "pose": {
                "frame_id": "world",
                "position": {"x": R_cube_center.tolist()[0], "y": R_cube_center.tolist()[1], "z": R_cube_center.tolist()[2]},
                "orientation": {"x": R_cube_quat.tolist()[0], "y": R_cube_quat.tolist()[1], "z": R_cube_quat.tolist()[2], "w": R_cube_quat.tolist()[3]},
            },
        })

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
            print(task_name)
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