from isaacgym.torch_utils import *
from isaac_ros_bridge.utils.franka_utils import *
from isaac_ros_bridge.arm_control import simple_controller, weld_controller
from isaac_ros_bridge.planner.motion_planner import rrt_plan
from isaac_ros_bridge.models.spot_weld_offsets import L_part_offset, R_part_offset
import torch

# === MoveIt IK Interface ===
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import rospy
import tf
from sensor_msgs.msg import JointState

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

        # ---------- ROS comms ----------
        self.joint_names = [f"joint{i+1}" for i in range(self.arm_dof)]
        self.ros_pub = rospy.Publisher("/rs007l_arm_controller/command", JointTrajectory, queue_size=1)
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

    def _lookup_real_ee_pose(self):
        try:
            trans, rot = self.listener.lookupTransform("base_link", "link6", rospy.Time(0))
            # Apply transform to match IsaacGym sim frame
            sim_rot = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)  # 180° Z rot
            sim_trans = torch.tensor([-0.45, 0.0, 0.1], device=self.device)

            pos = torch.tensor(trans, dtype=torch.float32, device=self.device)
            quat = torch.tensor(rot, dtype=torch.float32, device=self.device)
            self.ee_pos_real = quat_rotate(sim_rot.unsqueeze(0), pos.unsqueeze(0)).squeeze(0) + sim_trans
            self.ee_rot_real = quat_mul(quat_conjugate(sim_rot.unsqueeze(0)), quat.unsqueeze(0)).squeeze(0)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    # ---------- util ----------
    def _obb_from_pose(self, center, quat, half_extents):
        signs = torch.tensor([[ -1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], device=center.device)
        corners_local = signs * half_extents.unsqueeze(0)
        corners_world = quat_rotate(quat.unsqueeze(0).expand(8,4), corners_local) + center.unsqueeze(0)
        return corners_world.min(dim=0).values, corners_world.max(dim=0).values

    # def _real_weld_controller(self, goal_rot, dt, gripper_mode):
    #     self._lookup_real_ee_pose()

    #     hand_pos = self.ee_pos_real.unsqueeze(0)
    #     hand_rot = self.ee_rot_real.unsqueeze(0)

    #     if self.waypoint_idx[0] >= len(self.waypoints):
    #         return

    #     advance_thresh = 0.013
    #     max_lookahead = 8  # Number of future waypoints to include
    #     lookahead_traj = []

    #     idx = self.waypoint_idx[0].item()
    #     q_start = self.q_real.clone()

    #     for step_ahead in range(max_lookahead):
    #         i = idx + step_ahead
    #         if i >= len(self.waypoints):
    #             break

    #         goal_pos = self.waypoints[i][0]
    #         pos_err = goal_pos - hand_pos[0]
    #         ori_err = orientation_error(goal_rot, hand_rot)[0]

    #         dpose = torch.cat([
    #             torch.clamp(pos_err, -0.01, 0.01),
    #             torch.clamp(ori_err, -0.2, 0.2)
    #         ], dim=-1).unsqueeze(-1)

    #         # Build fake dof buffer using real q
    #         dof_real = self.sim.dof_pos.clone()
    #         dof_real[0, self.arm_idx, :self.arm_dof] = q_start.unsqueeze(-1)

    #         # IK to compute delta_q for this step
    #         delta_q = control_ik_nullspace(
    #             dpose.unsqueeze(0), self.sim.j_eef3[:1], 1, dof_real,
    #             self.robot_mids, self.arm_idx, self.arm_dof
    #         )[0]

    #         q_next = q_start + delta_q
    #         q_next = tensor_clamp(
    #         q_next,
    #         self.sim.robot_lower_limits_tensor[:self.arm_dof],
    #         self.sim.robot_upper_limits_tensor[:self.arm_dof],
    #         )
    #         lookahead_traj.append(q_next.clone())
    #         q_start = q_next  # advance seed for next point

    #         # Optional: simulate EEF moving forward
    #         hand_pos += quat_rotate(hand_rot, (goal_pos - hand_pos[0]).unsqueeze(0))

    #     # === Publish lookahead trajectory ===
    #     traj = JointTrajectory()
    #     traj.header = Header(stamp=rospy.Time.now())
    #     traj.joint_names = self.joint_names

    #     for i, q in enumerate(lookahead_traj):
    #         pt = JointTrajectoryPoint()
    #         pt.positions = q.tolist()
    #         pt.time_from_start = rospy.Duration((i + 1) * 0.1)  # 150ms per step
    #         traj.points.append(pt)

    #     self.ros_pub.publish(traj)

    #     # === Maintain same timer + waypoint advancement logic ===
    #     goal_pos = self.waypoints[self.waypoint_idx[0]][0]
    #     pos_err = goal_pos - hand_pos[0]
    #     if torch.norm(pos_err) < advance_thresh:
    #         self.waypoint_idx += 1
    #         self.waypoint_idx.clamp_(max=len(self.waypoints) - 1)

    #     if torch.norm(pos_err) < advance_thresh and self.waypoint_idx[0] == len(self.waypoints) - 1 and gripper_mode == "auto":
    #         if self.real_timer < 2.0:
    #             self.real_timer += dt
    #     else:
    #         self.real_timer = 0.0

    #     return self.real_timer

    # ---------- real robot stepwise control ----------
    def _real_weld_controller(self, goal_rot, dt, gripper_mode, plan_mode):
        self._lookup_real_ee_pose()

        hand_pos = self.ee_pos_real.unsqueeze(0)  # (1, 3)
        hand_rot = self.ee_rot_real.unsqueeze(0)  # (1, 4)

        # Early exit if we're already done
        if self.ros_waypoint_idx[0] >= len(self.waypoints):
            return

        # Current target waypoint (only 1 env)
        goal_pos = self.waypoints[self.ros_waypoint_idx[0]][0]  # (3,)
        pos_err = goal_pos - hand_pos[0]  # (3,)

        threshold = 0.025
        reached = torch.norm(pos_err) < threshold

        # Gripper logic
        if reached and self.ros_waypoint_idx[0] == len(self.waypoints) - 1:
            if gripper_mode == "auto" and self.real_timer < 2.0:
                grip_acts = torch.ones((1, 2), device=self.device) * 0.02
                self.real_timer += dt
                self.weld_timer += dt
            else:
                grip_acts = torch.ones((1, 2), device=self.device) * 0.04
        else:
            grip_acts = torch.ones((1, 2), device=self.device) * 0.04
            if reached:
                self.ros_waypoint_idx[0] += 1

        self.ros_waypoint_idx.clamp_(max=len(self.waypoints) - 1)

        # Re-fetch goal pose in case index changed
        goal_pos = self.waypoints[self.ros_waypoint_idx[0]][0]

        # Construct dpose
        dpose = torch.cat([
            torch.clamp(goal_pos - hand_pos[0], -0.015, 0.015),
            torch.clamp(orientation_error(goal_rot, hand_rot)[0], -0.2, 0.2)
        ]).unsqueeze(0).unsqueeze(-1)  # shape (1, 6, 1)

        # Use real joint states for nullspace IK
        dof_real = self.sim.dof_pos.clone()
        dof_real[0, self.arm_idx, :self.arm_dof] = self.q_real.unsqueeze(-1)

        delta_q = control_ik_nullspace(
            dpose, self.sim.j_eef3[:1], 1, dof_real,
            self.robot_mids, self.arm_idx, self.arm_dof
        )
        q_next = delta_q[0] + self.q_real

        # Publish joint trajectory to real robot
        traj = JointTrajectory()
        traj.header = Header(stamp=rospy.Time.now())
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = q_next.tolist()
        step_distance = torch.norm(delta_q[0]).item()
        pt.time_from_start = rospy.Duration(min(0.1, max(0.05, step_distance / 1.0)))
        traj.points = [pt]
        self.ros_pub.publish(traj)

        return self.real_timer, q_next, grip_acts


    # ---------- main step ----------
    def step(self, task_name, cur_pose, goal_pose, plan_mode, gripper_mode):
        if (not self.waypoints) or (self.task_name != task_name):
            self.waypoints.clear(); self.ros_waypoint_idx.zero_(); self.waypoint_idx.zero_(); self.task_name = task_name
            start, goal = cur_pose[:, :3], goal_pose[:, :3]
            if plan_mode == "plan":
                if self.prev_task_goal is not None: 
                    start = self.prev_task_goal
                big_low,big_high = self._obb_from_pose(self.sim.big_part_pos[0], self.sim.big_part_rot[0], torch.tensor([0.15,0.15,0.015], device=self.device))
                l_low,l_high = self._obb_from_pose(self.sim.L_part_pos[0], self.sim.L_part_rot[0], torch.tensor([0.02,0.1,0.01], device=self.device))
                cube_center = self.sim.L_part_pos + quat_rotate(self.sim.L_part_rot, torch.tensor([[0,0,0.03]], device=self.device))
                c_low = cube_center.squeeze(0)-torch.tensor([0.02,0.02,0.02], device=self.device)
                c_high= cube_center.squeeze(0)+torch.tensor([0.02,0.02,0.02], device=self.device)
                obs=[(big_low,big_high),(l_low,l_high),(c_low,c_high)]
                path=rrt_plan(start[0],goal[0],self.sim.reachable_pos3,obs,step_size=0.04,goal_thresh=0.01,device=self.device,safety_radius=0.11)
                dense=interpolate_waypoints(path,step=0.04)
                self.waypoints=[pt.unsqueeze(0).repeat(self.sim.num_envs,1) for pt in path]+[goal]
            else:
                self.waypoints.append(goal)
                self.prev_task_goal=goal

        self._lookup_real_ee_pose()
        hand_pos = self.sim.hand3_pos
        hand_rot = self.sim.hand3_rot
        goal_rot = goal_pose[:,3:]

        if self.weld_timer>=2.0 and self.real_timer >= 2.0: 
            self.weld_timer=0.0
            self.real_timer = 0.0

        # self.waypoint_idx, self.sim.pos_action, self.weld_timer = weld_controller(
        #     hand_pos, hand_rot, self.waypoints, goal_rot, self.waypoint_idx,
        #     self.sim.j_eef3, self.sim.dof_pos, self.sim.sim_params.dt, self.sim.pos_action,
        #     self.robot_mids, self.arm_idx, self.sim.num_envs, self.arm_dof,
        #     self.weld_timer, gripper_mode)
        # print(self.waypoint_idx, self.waypoints[self.waypoint_idx[0]], self.sim.hand3_pos)
        # print(self.waypoint_idx, torch.norm(self.waypoints[self.waypoint_idx[0]] - self.sim.hand3_pos))
        self.real_timer, q_next, grip_act = self._real_weld_controller(goal_rot, self.sim.sim_params.dt, gripper_mode, plan_mode)
        arm3_action = torch.cat([q_next.unsqueeze(0), grip_act], dim=1).squeeze(0)
        self.sim.pos_action[:, self.arm_idx] = arm3_action
        # embed()
        return self.weld_timer, self.real_timer