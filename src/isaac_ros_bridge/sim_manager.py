from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
from isaac_ros_bridge.utils.franka_utils import *
from isaac_ros_bridge.models.spot_weld_offsets import *
from isaac_ros_bridge.planner.motion_planner import *
from isaac_ros_bridge.planner.task_planner import *

import torch
import numpy as np
import math
import os


class RobotCellSim:
    def __init__(self, num_envs=1, controller_type="ik"):
        self.num_envs = num_envs
        self.num_arms = 3
        self.controller_type = controller_type
        self.device = "cuda:0"

        self._init_gym()
        self._create_sim()
        self._create_assets()
        self._create_envs()
        self._setup_camera()
        self._prepare_tensors()

    def _init_gym(self):
        self.gym = gymapi.acquire_gym()

    def _setup_camera(self):
        # point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def _create_sim(self):
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = True

        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.contact_offset = 0.001
        self.sim_params.physx.friction_offset_threshold = 0.001
        self.sim_params.physx.friction_correlation_distance = 0.0005
        self.sim_params.physx.num_threads = 4
        self.sim_params.physx.use_gpu = True

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")

    def _create_table(self):
        self.table_dims = gymapi.Vec3(0.6, 1.0, 0.3)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)
    
    def _create_parts(self, asset_root):
        # create L-part asset
        L_part_file = "urdf/L_part.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = True
        self.L_part_asset = self.gym.load_asset(self.sim, asset_root, L_part_file, asset_options)

        # create R-part asset
        R_part_file = "urdf/Rect_part.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = True
        self.R_part_asset = self.gym.load_asset(self.sim, asset_root, R_part_file, asset_options)

        # load big part asset
        big_part_file = "urdf/big_part.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = True
        self.big_part_asset = self.gym.load_asset(self.sim, asset_root, big_part_file, asset_options)

    def _create_assets(self):
        # Automatically resolve full asset path based on current script location
        this_dir = os.path.dirname(os.path.realpath(__file__))
        print(this_dir)
        asset_root = os.path.join(this_dir, "../../scripts/assets")

        robot_asset_file = "urdf/RS007/rs007l_panda.urdf"

        self._create_parts(asset_root)
        self._create_table()

        # load robot asset
        robot_opts = gymapi.AssetOptions()
        robot_opts.armature = 0.01
        robot_opts.fix_base_link = True
        robot_opts.disable_gravity = True
        robot_opts.flip_visual_attachments = True
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_opts)

        self.robot_hand_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        self.robot_dof = self.robot_hand_num_dofs - 2

        # configure robot dofs
        self.robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        self.robot_lower_limits = self.robot_dof_props["lower"]
        self.robot_upper_limits = self.robot_dof_props["upper"]
        self.robot_lower_limits_tensor = torch.tensor(self.robot_lower_limits, device=self.device)
        self.robot_upper_limits_tensor = torch.tensor(self.robot_upper_limits, device=self.device)
        self.robot_ranges = self.robot_upper_limits - self.robot_lower_limits
        self.robot_mids = 0.3 * (self.robot_upper_limits + self.robot_lower_limits)

        self.robot_dof_props["driveMode"][:self.robot_dof].fill(gymapi.DOF_MODE_POS)
        self.robot_dof_props["stiffness"][:self.robot_dof].fill(400.0)
        self.robot_dof_props["damping"][:self.robot_dof].fill(40.0)
        self.robot_dof_props["driveMode"][self.robot_dof:].fill(gymapi.DOF_MODE_POS)
        self.robot_dof_props["stiffness"][self.robot_dof:].fill(800.0)
        self.robot_dof_props["damping"][self.robot_dof:].fill(40.0)

        default_dof = np.zeros(self.robot_hand_num_dofs, dtype=np.float32)
        self.default_dof_pos_1 = default_dof.copy()
        self.default_dof_pos_2 = default_dof.copy()
        self.default_dof_pos_3 = default_dof.copy()
        self.default_dof_pos_1[:self.robot_dof] = np.array([0, -0.0325, -1.5809, 0, -1.8860, -1.5678])
        self.default_dof_pos_2[:self.robot_dof] = np.array([0, -0.0325, -1.5809, 0, -1.8860, -1.5678])
        self.default_dof_pos_3[:self.robot_dof] = self.robot_mids[:self.robot_dof]
        # set grippers to open
        self.default_dof_pos_1[self.robot_dof:] = self.robot_upper_limits[self.robot_dof:]
        self.default_dof_pos_2[self.robot_dof:] = self.robot_upper_limits[self.robot_dof:]
        self.default_dof_pos_3[self.robot_dof:] = self.robot_upper_limits[self.robot_dof:]

        self.default_dof_state_1 = np.zeros(self.robot_hand_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state_2 = np.zeros(self.robot_hand_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state_3 = np.zeros(self.robot_hand_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state_1["pos"] = self.default_dof_pos_1
        self.default_dof_state_2["pos"] = self.default_dof_pos_2
        self.default_dof_state_3["pos"] = self.default_dof_pos_3

        # get link index of panda hand, which we will use as end effector.
        self.robot_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        self.robot_hand_index = self.robot_link_dict["panda_hand"]

    def _create_envs(self):
        self.envs = []
        self.arm_root_idxs = []
        self.hand_idxs = []
        self.big_part_idxs = []
        self.L_part_idxs = []
        self.R_part_idxs = []
        self.part_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []

        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.num_per_row = int(math.sqrt(self.num_envs))

        arm_poses = [gymapi.Transform() for _ in range(self.num_arms)]
        self.table_poses = [gymapi.Transform() for _ in range(self.num_arms)]

        arm_poses[0].p = gymapi.Vec3(0.0, 0.7, 0)
        arm_poses[0].r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 2)
        self.table_poses[0].p = gymapi.Vec3(0.5, 0.7, self.table_dims.z / 2)

        arm_poses[1].p = gymapi.Vec3(0.0, -0.7, 0)
        arm_poses[1].r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 2)
        self.table_poses[1].p = gymapi.Vec3(0.5, -0.7, self.table_dims.z / 2)

        arm_poses[2].p = gymapi.Vec3(-0.45, 0.0, 0.1)
        arm_poses[2].r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi)

        L_part_pose = gymapi.Transform()
        R_part_pose = gymapi.Transform()

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        self.table_rot = torch.zeros((self.num_envs, 4), device=device)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, self.num_per_row)
            self.envs.append(env)

            for arm_idx in range(self.num_arms):
                # add arm
                arm = self.gym.create_actor(env, self.robot_asset, arm_poses[arm_idx], f"arm{arm_idx}", i, -1, 0)
                arm_root_idx = self.gym.get_actor_index(env, arm, gymapi.DOMAIN_SIM)
                self.arm_root_idxs.append(arm_root_idx)

                self.gym.set_actor_dof_properties(env, arm, self.robot_dof_props)

                # set initial dof states
                if arm_idx == 0:
                    self.gym.set_actor_dof_states(env, arm, self.default_dof_state_1, gymapi.STATE_ALL)
                elif arm_idx == 1:
                    self.gym.set_actor_dof_states(env, arm, self.default_dof_state_2, gymapi.STATE_ALL)
                else:
                    self.gym.set_actor_dof_states(env, arm, self.default_dof_state_3, gymapi.STATE_ALL)

                # set initial position targets
                if arm_idx == 0:
                    self.gym.set_actor_dof_position_targets(env, arm, self.default_dof_pos_1)
                elif arm_idx == 1:
                    self.gym.set_actor_dof_position_targets(env, arm, self.default_dof_pos_2)
                else:
                    self.gym.set_actor_dof_position_targets(env, arm, self.default_dof_pos_3)

                # get inital hand pose
                hand_handle = self.gym.find_actor_rigid_body_handle(env, arm, "panda_hand")
                hand_pose = self.gym.get_rigid_transform(env, hand_handle)
                self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
                self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

                # get global index of hand in rigid body state tensor
                hand_idx = self.gym.find_actor_rigid_body_index(env, arm, "panda_hand", gymapi.DOMAIN_SIM)
                self.hand_idxs.append(hand_idx)

                # add table
                if arm_idx == 0:
                    table_handle = self.gym.create_actor(env, self.table_asset, self.table_poses[arm_idx], f"table{arm_idx}", i, 0)

                    # add big part
                    big_part_pose = gymapi.Transform()
                    big_part_pose.p = self.table_poses[arm_idx].p
                    big_part_pose.p.z = self.table_dims.z
                    # big_part_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi / 2)
                    big_part_handle = self.gym.create_actor(env, self.big_part_asset, big_part_pose, "big_part", i, 0)
                    self.big_part_root_idx = self.gym.get_actor_index(env, big_part_handle, gymapi.DOMAIN_SIM)

                    big_part_idx = self.gym.get_actor_rigid_body_index(env, big_part_handle, 0, gymapi.DOMAIN_SIM)
                    self.big_part_idxs.append(big_part_idx)
                if arm_idx == 1:
                    self.table_rot[i, 0] = self.table_poses[arm_idx].r.x
                    self.table_rot[i, 1] = self.table_poses[arm_idx].r.y
                    self.table_rot[i, 2] = self.table_poses[arm_idx].r.z
                    self.table_rot[i, 3] = self.table_poses[arm_idx].r.w
                    table_handle = self.gym.create_actor(env, self.table_asset, self.table_poses[arm_idx], f"table{arm_idx}", i, 0)

                    # add L part
                    L_part_pose.p.x = self.table_poses[arm_idx].p.x
                    L_part_pose.p.y = self.table_poses[arm_idx].p.y
                    L_part_pose.p.z = self.table_dims.z + 0.5 * 0.01
                    L_part_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 2)
                    L_part_handle = self.gym.create_actor(env, self.L_part_asset, L_part_pose, "L_part", i, 0)
                    self.L_part_root_idx = self.gym.get_actor_index(env, L_part_handle, gymapi.DOMAIN_SIM)
                    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                    self.gym.set_rigid_body_color(env, L_part_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                    L_part_idx = self.gym.get_actor_rigid_body_index(env, L_part_handle, 0, gymapi.DOMAIN_SIM)
                    self.L_part_idxs.append(L_part_idx)
                    self.part_idxs.append(L_part_idx)

                    # add Rec part
                    R_part_pose.p.x = self.table_poses[arm_idx].p.x + 0.05
                    R_part_pose.p.y = self.table_poses[arm_idx].p.y - 0.15
                    R_part_pose.p.z = self.table_dims.z + 0.5 * 0.01
                    R_part_handle = self.gym.create_actor(env, self.R_part_asset, R_part_pose, "R_part", i, 0)
                    self.R_part_root_idx = self.gym.get_actor_index(env, R_part_handle, gymapi.DOMAIN_SIM)
                    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                    self.gym.set_rigid_body_color(env, R_part_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                    R_part_idx = self.gym.get_actor_rigid_body_index(env, R_part_handle, 0, gymapi.DOMAIN_SIM)
                    self.R_part_idxs.append(R_part_idx)
                    self.part_idxs.append(R_part_idx)

    def _prepare_tensors(self):
        self.gym.prepare_sim(self.sim)

        self.hand_idxs = to_torch(self.hand_idxs, device=device, dtype=int).view(self.num_envs, self.num_arms)

        self.init_pos = torch.tensor(self.init_pos_list).view(self.num_envs, self.num_arms, 3).to(self.device)
        self.init_rot = torch.tensor(self.init_rot_list).view(self.num_envs, self.num_arms, 4).to(self.device)

        # hand orientation for grasping
        down_q = torch.stack(self.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((self.num_envs, 4))

         # box corner coords, used to determine grasping yaw
        big_part_long_axis = torch.tensor([1.0, 0.0, 0.0], device=device) # unit vector along X of big part.
        big_part_corners = big_part_long_axis.repeat(self.num_envs, 1)

        L_part_long_axis = torch.tensor([1.0, 0.0, 0.0], device=device)  # unit vector along Y of L part.
        L_part_corners = L_part_long_axis.repeat(self.num_envs, 1)  # (self.num_envs, 3)

        # downard axis
        self.down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

        # get jacobian1 tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        self._jacobian1 = self.gym.acquire_jacobian_tensor(self.sim, "arm0")
        self.jacobian1 = gymtorch.wrap_tensor(self._jacobian1)

        self._jacobian2 = self.gym.acquire_jacobian_tensor(self.sim, "arm1")
        self.jacobian2 = gymtorch.wrap_tensor(self._jacobian2)

        self._jacobian3 = self.gym.acquire_jacobian_tensor(self.sim, "arm2")
        self.jacobian3 = gymtorch.wrap_tensor(self._jacobian3)

        # jacobian entries corresponding to franka hand
        self.j_eef1 = self.jacobian1[:, self.robot_hand_index - 1, :, :self.robot_dof]
        self.j_eef2 = self.jacobian2[:, self.robot_hand_index - 1, :, :self.robot_dof]
        self.j_eef3 = self.jacobian3[:, self.robot_hand_index - 1, :, :self.robot_dof]


        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, self.num_arms, self.robot_hand_num_dofs, 1)

        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)

        self._root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(self._root_states).view(self.num_envs, -1, 13)

        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.saved_root_tensor = self.root_state_tensor.clone()

        # Big part grasping orientation
        big_part_rot = self.rb_states[self.big_part_idxs, 3:7]
        yaw_q_1 = box_grasping_yaw(big_part_rot, big_part_corners)
        self.goal_rot_gripp1 = quat_mul(yaw_q_1, down_q)

        # L part grasping orientation
        L_part_rot = self.rb_states[self.L_part_idxs, 3:7]
        yaw_q_2 = box_grasping_yaw(L_part_rot, L_part_corners)
        self.goal_rot_gripp2 = quat_mul(yaw_q_2, down_q)

        # L part pos offset for grasping only.
        self.part_grasp_offset = torch.stack(self.num_envs * [torch.tensor([0, 0, 0.025])]).to(device).view((self.num_envs, 3))

        # Big part welding pos initialization
        self.big_part_welding_pos = []
        self.big_part_welding_pos.append(big_L_welding_offset) 
        self.big_part_welding_pos.append(big_R_welding_offset)
        self.big_part_welding_pos = torch.cat(self.big_part_welding_pos).to(device).unsqueeze(0)
        self.big_L_welding_offset = big_L_welding_offset.to(device).unsqueeze(0)
        self.big_R_welding_offset = big_R_welding_offset.to(device).unsqueeze(0)

        # L part welding pos initialization
        self.L_part_welding_offset = L_part_welding_offset.to(device).unsqueeze(0)
        self.L_part_welding_pos = torch.zeros_like(self.L_part_welding_offset).to(device)
        # R part welding pos initialization
        self.R_part_welding_offset = R_part_welding_offset.to(device).unsqueeze(0)
        self.R_part_welding_pos = torch.zeros_like(self.R_part_welding_offset).to(device)
        # Predefined achor position for the big part to be in world frame.
        self.big_part_anchor = torch.tensor([0.0, -0.0, 0.75], device=device)
        self.big_part_goal_quat = torch.tensor([[0.7071068, 0, 0, -0.7071068]] * self.num_envs).to(device)

        self.spot_weld_nums = [7, 4]

        # q_gripper_to_part = inverse(q_hand_world) * q_part_world
        q_gripper_to_part = quat_mul(quat_conjugate(self.goal_rot_gripp1), big_part_rot)
        self.q_hand1_goal = quat_mul(self.big_part_goal_quat, quat_conjugate(q_gripper_to_part))

        self.next_part = False
        self.fixed_L_part_offset_pos = None
        self.fixed_L_part_offset_rot = None
        self.l_part_attached = False

        table_center = torch.zeros((self.num_envs, 3), device=device)
        table_center[:, 0] = self.table_poses[1].p.x
        table_center[:, 1] = self.table_poses[1].p.y
        table_center[:, 2] = self.table_poses[1].p.z

        table_half = torch.tensor([0.3, 0.5, 0.15], device=device).unsqueeze(0)
        lower_bound = table_center - quat_rotate(self.table_rot, table_half)
        upper_bound = table_center + quat_rotate(self.table_rot, table_half)
        self.obstacles_2 = [(lower_bound, upper_bound)]

        this_dir = os.path.dirname(os.path.realpath(__file__))
        reachable_sapce_dir = os.path.join(this_dir, "utils/RS007L_manifold.npy")
        self.reachable_space = np.load(reachable_sapce_dir)
        reachable_pos = self.reachable_space[:, :3]
        reachable_pos = torch.tensor(reachable_pos, device=device, dtype=torch.float32)

        base_pos1 = self.saved_root_tensor[0, self.arm_root_idxs[0], 0:3] 
        base_quat1 = self.saved_root_tensor[0, self.arm_root_idxs[0], 3:7]
        N = reachable_pos.shape[0]
        base_pos_expand1 = base_pos1.unsqueeze(0).expand(N, 3)
        base_quat_expand1 = base_quat1.unsqueeze(0).expand(N, 4)
        self.reachable_pos = quat_apply(base_quat_expand1, reachable_pos) + base_pos_expand1

        base_pos2 = self.saved_root_tensor[0, self.arm_root_idxs[1], 0:3] 
        base_quat2 = self.saved_root_tensor[0, self.arm_root_idxs[1], 3:7]
        N = reachable_pos.shape[0]
        base_pos_expand2 = base_pos2.unsqueeze(0).expand(N, 3)
        base_quat_expand2 = base_quat2.unsqueeze(0).expand(N, 4)
        self.reachable_pos2 = quat_apply(base_quat_expand2, reachable_pos) + base_pos_expand2

        base_pos3 = self.saved_root_tensor[0, self.arm_root_idxs[2], 0:3] 
        base_quat3 = self.saved_root_tensor[0, self.arm_root_idxs[2], 3:7]
        print(base_pos3)
        N = reachable_pos.shape[0]
        base_pos_expand3 = base_pos3.unsqueeze(0).expand(N, 3)
        base_quat_expand3 = base_quat3.unsqueeze(0).expand(N, 4)
        self.reachable_pos3 = quat_apply(base_quat_expand3, reachable_pos) + base_pos_expand3

        self.big_part_pos = self.rb_states[self.big_part_idxs, :3]
        self.big_part_rot = self.rb_states[self.big_part_idxs, 3:7]

        self.L_part_pos = self.rb_states[self.L_part_idxs, :3]
        self.L_part_rot = self.rb_states[self.L_part_idxs, 3:7]

        self.R_part_pos = self.rb_states[self.R_part_idxs, :3]
        self.R_part_rot = self.rb_states[self.R_part_idxs, 3:7]

        hand_poses = self.rb_states[self.hand_idxs, :3]
        hand_rots = self.rb_states[self.hand_idxs, 3:7]

        self.hand1_pos = hand_poses[:, 0]
        self.hand1_rot = hand_rots[:, 0]

        self.hand2_pos = hand_poses[:, 1]
        self.hand2_rot = hand_rots[:, 1]

        self.hand3_pos = hand_poses[:, 2]
        self.hand3_rot = hand_rots[:, 2]

        # how far the hand should be from box for grasping
        self.grasp_offset = 0.11
        self.grasp_offset_arm2 = 0.1
        # The hand to big part position offset in part frame
        self.hand_to_big = torch.tensor([0.0, 0.0, 0.117], device=device).unsqueeze(0)
        self.hand_to_small = torch.tensor([0.0, 0.0, 0.132], device=device).unsqueeze(0)

         # Setup big part spot weld positions in the world frame.
        for i in range(self.big_L_welding_offset.shape[1]):
            self.big_part_welding_pos[:, i] = self.big_part_anchor + quat_rotate(self.big_part_goal_quat, self.big_L_welding_offset[:, i])
        
        for i in range(self.big_R_welding_offset.shape[1]):
            self.big_part_welding_pos[:, i + self.spot_weld_nums[0]] = self.big_part_anchor + quat_rotate(self.big_part_goal_quat, self.big_R_welding_offset[:, i])

        # Setup L part spot weld positions in the world frame. 
        for i in range(self.L_part_welding_offset.shape[1]):
            self.L_part_welding_pos[:, i] = self.L_part_pos + quat_rotate(self.L_part_rot, self.L_part_welding_offset[:, i])
        
        # Setup R part spot weld positions in the world frame.
        for i in range(self.R_part_welding_offset.shape[1]):
            self.R_part_welding_pos[:, i] = self.R_part_pos + quat_rotate(self.R_part_rot, self.R_part_welding_offset[:, i])

    def step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        # fix L-part
        if self.next_part and not self.l_part_attached:
            big_part_pos_now = self.root_state_tensor[:, self.big_part_root_idx, 0:3]
            big_part_rot_now = self.root_state_tensor[:, self.big_part_root_idx, 3:7]
            l_part_pos_now = self.root_state_tensor[:, self.L_part_root_idx, 0:3]
            l_part_rot_now = self.root_state_tensor[:, self.L_part_root_idx, 3:7]

            self.fixed_L_part_offset_pos = quat_rotate(quat_conjugate(big_part_rot_now), l_part_pos_now - big_part_pos_now)
            self.fixed_L_part_offset_rot = quat_mul(quat_conjugate(big_part_rot_now), l_part_rot_now)

            self.l_part_attached = True

        if self.l_part_attached:
            big_part_pos_now = self.root_state_tensor[:, self.big_part_root_idx, 0:3]
            big_part_rot_now = self.root_state_tensor[:, self.big_part_root_idx, 3:7]

            l_part_pos_target = quat_rotate(big_part_rot_now, self.fixed_L_part_offset_pos) + big_part_pos_now
            l_part_rot_target = quat_mul(big_part_rot_now, self.fixed_L_part_offset_rot)

            self.root_state_tensor[:, self.L_part_root_idx, 0:3] = l_part_pos_target
            self.root_state_tensor[:, self.L_part_root_idx, 3:7] = l_part_rot_target
            self.root_state_tensor[:, self.L_part_root_idx, 7:10] = 0.0
            self.root_state_tensor[:, self.L_part_root_idx, 10:13] = 0.0
            self.root_state_tensor[:, self.arm_root_idxs[0]] = self.saved_root_tensor[:, self.arm_root_idxs[0]]
            self.root_state_tensor[:, self.arm_root_idxs[1]] = self.saved_root_tensor[:, self.arm_root_idxs[1]]
            self.root_state_tensor[:, self.arm_root_idxs[2]] = self.saved_root_tensor[:, self.arm_root_idxs[2]]

            self.gym.set_actor_root_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor)
            )

        self.big_part_pos = self.rb_states[self.big_part_idxs, :3]
        self.big_part_rot = self.rb_states[self.big_part_idxs, 3:7]

        self.L_part_pos = self.rb_states[self.L_part_idxs, :3]
        self.L_part_rot = self.rb_states[self.L_part_idxs, 3:7]

        self.R_part_pos = self.rb_states[self.R_part_idxs, :3]
        self.R_part_rot = self.rb_states[self.R_part_idxs, 3:7]

        hand_poses = self.rb_states[self.hand_idxs, :3]
        hand_rots = self.rb_states[self.hand_idxs, 3:7]

        self.hand1_pos = hand_poses[:, 0]
        self.hand1_rot = hand_rots[:, 0]

        self.hand2_pos = hand_poses[:, 1]
        self.hand2_rot = hand_rots[:, 1]

        self.hand3_pos = hand_poses[:, 2]
        self.hand3_rot = hand_rots[:, 2]

        self.pos_action[..., :self.robot_dof] = tensor_clamp(
            self.pos_action[..., :self.robot_dof],
            self.robot_lower_limits_tensor[:self.robot_dof],
            self.robot_upper_limits_tensor[:self.robot_dof],
        )
        
        # Hardcode the initial joint 1 pos of the welding robot to 90 degrees so it faces the working environment.
        self.pos_action[:, 2, 0] = -1.57
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "pause":
                embed()

    def update_viewer(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def is_viewer_closed(self):
        return self.gym.query_viewer_has_closed(self.viewer)

    def cleanup(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
