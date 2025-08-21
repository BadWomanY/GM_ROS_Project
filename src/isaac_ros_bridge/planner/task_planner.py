from isaacgym import gymtorch
import torch
from collections import deque
from isaac_ros_bridge.utils.franka_utils import *
from isaac_ros_bridge.models.spot_weld_offsets import *
from IPython import embed

"""Libraries for GTSP solving."""
import numpy as np
from scipy.spatial.distance import cdist
from itertools import permutations

class TaskPlanner():
    def __init__(self, sim):
        self.sim = sim
        self.device = sim.device
        self.hand1_to_part_offset = torch.tensor([0.0, 0.0, 0.117], device=self.device).unsqueeze(0)

        # Initialize task queues for each robot
        self.r1_task_queue = deque()
        self.r2_task_queue = deque()
        self.r3_task_queue = deque()

        self.task_lib = dict()

        # Initialize current tasks for each robot
        self.r1_task = None
        self.r2_task = None
        self.r3_task = None

        # Number of parts and their spot weld numbers.
        self.part1_weld_num = 7
        self.part2_weld_num = 5
        self.pre_ids = [3, 6]
        self.post_ids = [0, 1, 2, 4, 5]
        self.pre_ids_R = [7, 11]
        self.post_ids_R = [9, 10]

        self.arm3_flag = False
    
    def compute_weld_quat_from_normal(self, normal: torch.Tensor, device):
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        reference = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        ref_proj = reference - (reference @ normal.transpose(0, 1)) * normal
        ref_proj = ref_proj / torch.norm(ref_proj, dim=-1, keepdim=True)
        y_axis = torch.cross(normal, ref_proj, dim=-1)
        x_axis = torch.cross(y_axis, normal, dim=-1)
        rot_matrix = torch.stack([x_axis, y_axis, normal], dim=-2)
        return matrix_to_quaternion(rot_matrix.unsqueeze(0)).to(device)

    
    def get_order_with_start(self, weld_positions, start_pose=None):
        """
        Simple TSP solver (brute force) to find optimal order of welds.
        Args:
            weld_positions: list of (1,3) torch tensors for each weld.
            start_pose: (1,3) torch tensor for start/end-effector pose.
        Returns:
            List of indices (0..len(weld_positions)-1) representing optimal visiting order.
        """
        import numpy as np
        from itertools import permutations
        from scipy.spatial.distance import cdist

        # Convert to numpy
        weld_np = torch.cat(weld_positions, dim=0).cpu().numpy()
        if start_pose is not None:
            start_np = start_pose.cpu().numpy()
            weld_np = np.vstack([start_np, weld_np])  # start at index 0

        N = len(weld_positions)
        cost_matrix = cdist(weld_np, weld_np)

        nodes = list(range(1, N + 1)) if start_pose is not None else list(range(N))
        best_path = None
        min_cost = float("inf")

        for perm in permutations(nodes):
            total_cost = 0
            if start_pose is not None:
                total_cost += cost_matrix[0, perm[0]]
            for i in range(len(perm) - 1):
                total_cost += cost_matrix[perm[i], perm[i + 1]]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = perm

        return [i - 1 if start_pose is not None else i for i in best_path]



    def create_task_lib_r1(self):
        # Robot 1 sub-tasks
        big_part_grasp_pos = self.sim.big_part_pos.clone() + self.sim.part_grasp_offset
        big_part_grasp_rot = self.sim.goal_rot_gripp1  # Grasping orientation in quat.
        big_part_grasp_pose = torch.cat([big_part_grasp_pos, big_part_grasp_rot], dim=1)
        self.r1_grasp = ['T_1_1', big_part_grasp_pose, [], [], 'no-plan', 'auto', False]
        self.r1_task_queue.append(self.r1_grasp)
        self.task_lib['T_1_1'] = self.r1_grasp

        hand1_mate_pos = self.sim.big_part_anchor + quat_rotate(self.sim.big_part_goal_quat, self.sim.part_grasp_offset)
        hand1_mate_rot = self.sim.q_hand1_goal
        hand1_mate_pose = torch.cat([hand1_mate_pos, hand1_mate_rot], dim=1)
        self.r1_mate = ['T_1_2', hand1_mate_pose, ['T_1_1', 'T_2_1'], [f"T_3_{self.part1_weld_num}"], 'plan', 'close', False]
        self.r1_task_queue.append(self.r1_mate)
        self.task_lib['T_1_2'] = self.r1_mate

        # Hand 1 change orientation for part 2 welding.
        hand1_mate_rot2 = self.sim.q_hand1_goal2
        hand1_mate_pose2 = torch.cat([hand1_mate_pos, hand1_mate_rot2], dim=1)
        self.r1_mate = ['T_1_3', hand1_mate_pose2, ['T_1_2', f"T_3_{self.part1_weld_num + 1}"], [], 'no-plan', 'close', False]
        self.r1_task_queue.append(self.r1_mate)
        self.task_lib['T_1_3'] = self.r1_mate

    def create_task_lib_r2(self):
        # Robot 2 sub-tasks
        L_part_grasp_pos = self.sim.L_part_pos + quat_rotate(self.sim.L_part_rot, self.sim.part_grasp_offset)
        part_grasp_rot = self.sim.goal_rot_gripp2  # Grasping orientation in quat for small part grasping.
        L_part_grasp_pose = torch.cat([L_part_grasp_pos, part_grasp_rot], dim=1)
        self.r2_L_grasp = ['T_2_1', L_part_grasp_pose, [], [], 'no-plan', 'auto', False]
        self.r2_task_queue.append(self.r2_L_grasp)
        self.task_lib['T_2_1'] = self.r2_L_grasp

        weld_big_L = self.sim.big_part_welding_pos[:, :7]
        weld_L = self.sim.L_part_welding_pos
        part_offset = L_part_offset
        hand2_mate_pos, self.L_part_goal_rot = part_mate_hand_pos(self.sim.big_part_anchor,
                                              self.sim.big_part_goal_quat,
                                              self.sim.part_grasp_offset,
                                              part_offset,
                                              weld_big_L,
                                              weld_L,
                                              self.device)
        
        part_q = self.sim.L_part_rot.clone()
        self.hand2_to_Lpart_rot_offset = quat_mul(quat_conjugate(self.sim.goal_rot_gripp2), part_q)
        hand2_mate_rot = quat_mul(self.L_part_goal_rot, quat_conjugate(self.sim.goal_rot_gripp2))
        self.L_part_goal_rot = quat_mul(hand2_mate_rot, quat_conjugate(self.hand2_to_Lpart_rot_offset))

        hand2_mate_pose = torch.cat([hand2_mate_pos, hand2_mate_rot], dim=1)
        self.r2_L_mate = ['T_2_2', hand2_mate_pose, ['T_2_1'], [f'T_3_{len(self.pre_ids)}'], 'plan', 'close', False]
        self.r2_task_queue.append(self.r2_L_mate)
        self.task_lib['T_2_2'] = self.r2_L_mate

        # Robot 2 return home
        home_pos = self.sim.init_pos[:, 1]
        home_rot = self.sim.init_rot[:, 1]
        home_pose = torch.cat([home_pos, home_rot], dim=1)
        self.r2_return = ['T_2_3', home_pose, ['T_2_2'], [], 'no-plan', 'open', False]
        self.r2_task_queue.append(self.r2_return)
        self.task_lib['T_2_3'] = self.r2_return

        # R part grasping task.
        R_part_grasp_pos = self.sim.R_part_pos + quat_rotate(self.sim.R_part_rot, self.sim.part_grasp_offset)
        part_grasp_rot = self.sim.goal_rot_gripp2_R  # Grasping orientation in quat for small part grasping.
        R_part_grasp_pose = torch.cat([R_part_grasp_pos, part_grasp_rot], dim=1)
        self.r2_R_grasp = ['T_2_4', R_part_grasp_pose, ['T_2_3'], [], 'no-plan', 'auto', False]
        self.r2_task_queue.append(self.r2_R_grasp)
        self.task_lib['T_2_4'] = self.r2_R_grasp

        # R part buffer/lift task - lift the part up before mating
        R_part_buffer_pos = R_part_grasp_pos.clone()
        R_part_buffer_pos[:, 2] += 0.20  # Lift 10cm above grasp position
        R_part_buffer_pose = torch.cat([R_part_buffer_pos, part_grasp_rot], dim=1)
        self.r2_R_buffer = ['T_2_5', R_part_buffer_pose, ['T_2_4', 'T_1_3'], [], 'plan', 'close', False]
        self.r2_task_queue.append(self.r2_R_buffer)
        self.task_lib['T_2_5'] = self.r2_R_buffer

        weld_big_R = self.sim.big_part_welding_pos[:, 7:]
        weld_R = self.sim.R_part_welding_pos
        part_offset = R_part_offset
        hand2_mate_pos, self.R_part_goal_rot = part_mate_hand_pos(self.sim.big_part_anchor,
                                              self.sim.big_part_goal_quat2,
                                              self.sim.part_grasp_offset,
                                              part_offset,
                                              weld_big_R,
                                              weld_R,
                                              self.device)
        
        part_q = self.sim.R_part_rot.clone()
        self.hand2_to_part_rot_offset = quat_mul(quat_conjugate(self.sim.goal_rot_gripp2_R), part_q)
        hand2_mate_rot = quat_mul(self.R_part_goal_rot, quat_conjugate(self.sim.goal_rot_gripp2_R))
        self.R_part_goal_rot = quat_mul(hand2_mate_rot, quat_conjugate(self.hand2_to_part_rot_offset))

        hand2_mate_pose = torch.cat([hand2_mate_pos, hand2_mate_rot], dim=1)
        self.r2_R_mate = ['T_2_6', hand2_mate_pose, ['T_2_5'], [], 'plan', 'close', False]
        self.r2_task_queue.append(self.r2_R_mate)
        self.task_lib['T_2_6'] = self.r2_R_mate

        # Robot 2 return home
        home_pos = self.sim.init_pos[:, 1]
        home_rot = self.sim.init_rot[:, 1]
        home_pose = torch.cat([home_pos, home_rot], dim=1)
        self.r2_return = ['T_2_7', home_pose, [f'T_3_{self.part1_weld_num + 1 + len(self.pre_ids_R)}'], [], 'no-plan', 'open', False]
        self.r2_task_queue.append(self.r2_return)
        self.task_lib['T_2_7'] = self.r2_return

    def create_task_lib_r3(self):
        hover_offset = 0.15
        grasp_offset = self.sim.grasp_offset_arm2  # use the actual value from sim

        # === Orientation setup from user logic ===
        desired_z_top = quat_axis(self.L_part_goal_rot, 0)   # X-axis in world frame
        reference_up = quat_axis(self.L_part_goal_rot, 2)    # Z-axis up
        weld_top = look_at_quat(desired_z_top, reference_up)

        desired_z_side = quat_axis(self.L_part_goal_rot, 1)  # Y-axis
        weld_side = look_at_quat(desired_z_side, reference_up)

        # === Part 1 weld IDs ===
        self.pre_ids = [3, 6]
        self.post_ids = [0, 1, 2, 4, 5]
        pre_positions = [self.sim.big_part_welding_pos[:, i, :] for i in self.pre_ids]
        post_positions = [self.sim.big_part_welding_pos[:, i, :] for i in self.post_ids]
        start_pose = self.sim.init_pos[:, 2]

        pre_order = self.get_order_with_start(pre_positions, start_pose=start_pose)
        task_count = 0
        # === Pre-weld tasks (Part 1) ===
        for local_idx in pre_order:
            weld_idx = self.pre_ids[local_idx]
            weld_pos = self.sim.big_part_welding_pos[:, weld_idx, :].clone()
            weld_rot = weld_top if weld_idx <= 3 else weld_side

            hover_weld_pos = weld_pos.clone()
            if weld_idx <= 3:
                # weld_pos[:, 2] += grasp_offset
                hover_weld_pos[:, 2] += hover_offset
            else:
                # weld_pos[:, 0] -= grasp_offset
                hover_weld_pos[:, 0] -= hover_offset

            weld_pose = torch.cat([weld_pos, weld_rot], dim=1)

            # hover_name = f"T_3_{task_count+1}"
            weld_name = f"T_3_{task_count+1}"
            # self.r3_task_queue.append([hover_name, hover_pose, [], [], 'plan', 'open', False])
            self.r3_task_queue.append([weld_name, weld_pose, [], [], 'plan', 'auto', False])
            # self.task_lib[hover_name] = self.r3_task_queue[-2]
            self.task_lib[weld_name] = self.r3_task_queue[-1]
            task_count += 1
            last_pose = weld_pos

        # === Post-weld tasks (Part 1) ===
        post_order = self.get_order_with_start(post_positions, start_pose=last_pose)
        for local_idx in post_order:
            weld_idx = self.post_ids[local_idx]
            weld_pos = self.sim.big_part_welding_pos[:, weld_idx, :].clone()
            weld_rot = weld_top if weld_idx <= 3 else weld_side

            hover_weld_pos = weld_pos.clone()
            if weld_idx <= 3:
                # weld_pos[:, 2] += grasp_offset
                hover_weld_pos[:, 2] += hover_offset
            else:
                # weld_pos[:, 0] -= grasp_offset
                hover_weld_pos[:, 0] -= hover_offset

            # hover_pose = torch.cat([hover_weld_pos, weld_rot], dim=1)
            weld_pose = torch.cat([weld_pos, weld_rot], dim=1)

            # hover_name = f"T_3_{task_count+1}"
            weld_name = f"T_3_{task_count+1}"
            # self.r3_task_queue.append([hover_name, hover_pose, [f"T_3_{task_count}"], [], 'plan', 'open', False])
            self.r3_task_queue.append([weld_name, weld_pose, [], [], 'plan', 'auto', False])
            # self.task_lib[hover_name] = self.r3_task_queue[-2]
            self.task_lib[weld_name] = self.r3_task_queue[-1]
            task_count += 1
        
        # === Return home task ===
        home_pos = torch.tensor([[-0.12681742, -0.01299958, 1.19022809]], device=self.device)
        home_rot = torch.tensor([[ 0.91465718, -0.11491458, 0.38563732, 0.0384803 ]], device=self.device)
        home_pose = torch.cat([home_pos, home_rot], dim=1)
        home_task = [f"T_3_{task_count+1}", home_pose, [f"T_3_{task_count}"], [], 'plan', 'open', False]
        self.r3_task_queue.append(home_task)
        self.task_lib[f"T_3_{task_count+1}"] = home_task
        task_count += 1
        last_pose = home_pos

        # === Part 2 weld IDs ===
        self.pre_ids_R = [7, 11]
        self.post_ids_R = [9, 10]
        pre_positions_R = [self.sim.big_part_welding_pos[:, i, :] for i in self.pre_ids_R]
        post_positions_R = [self.sim.big_part_welding_pos[:, i, :] for i in self.post_ids_R]

        pre_order_R = self.get_order_with_start(pre_positions_R, start_pose=last_pose)

        # === Pre-weld tasks (Part 2) ===
        for local_idx in pre_order_R:
            weld_idx = self.pre_ids_R[local_idx]
            weld_pos = self.sim.big_part_welding_pos[:, weld_idx, :].clone()
            weld_rot = weld_top

            hover_weld_pos = weld_pos.clone()
            # weld_pos[:, 2] += grasp_offset
            hover_weld_pos[:, 2] += hover_offset

            hover_pose = torch.cat([hover_weld_pos, weld_rot], dim=1)
            weld_pose = torch.cat([weld_pos, weld_rot], dim=1)

            # hover_name = f"T_3_{task_count+1}"
            weld_name = f"T_3_{task_count+1}"
            # self.r3_task_queue.append([hover_name, hover_pose, [f"T_3_{task_count}", "T_2_7"], [], 'plan', 'open', False])
            self.r3_task_queue.append([weld_name, weld_pose, ["T_2_6"], [], 'plan', 'auto', False])
            # self.task_lib[hover_name] = self.r3_task_queue[-2]
            self.task_lib[weld_name] = self.r3_task_queue[-1]
            task_count += 1
            last_pose = weld_pos

        # === Post-weld tasks (Part 2) ===
        post_order_R = self.get_order_with_start(post_positions_R, start_pose=last_pose)
        for local_idx in post_order_R:
            weld_idx = self.post_ids_R[local_idx]
            weld_pos = self.sim.big_part_welding_pos[:, weld_idx, :].clone()
            weld_rot = weld_top

            hover_weld_pos = weld_pos.clone()
            # weld_pos[:, 2] += grasp_offset
            hover_weld_pos[:, 2] += hover_offset

            hover_pose = torch.cat([hover_weld_pos, weld_rot], dim=1)
            weld_pose = torch.cat([weld_pos, weld_rot], dim=1)

            # hover_name = f"T_3_{task_count+1}"
            weld_name = f"T_3_{task_count+1}"
            # self.r3_task_queue.append([hover_name, hover_pose, [f"T_3_{task_count}"], [], 'plan', 'open', False])
            self.r3_task_queue.append([weld_name, weld_pose, [], [], 'plan', 'auto', False])
            # self.task_lib[hover_name] = self.r3_task_queue[-2]
            self.task_lib[weld_name] = self.r3_task_queue[-1]
            task_count += 1
            last_pose = weld_pos

        # === Return home task ===
        home_pos = torch.tensor([[-0.169, -0.011, 1.169]], device=self.device)
        home_rot = torch.tensor([[ 0.91465718, -0.11491458, 0.38563732, 0.0384803 ]], device=self.device)
        home_pose = torch.cat([home_pos, home_rot], dim=1)
        home_task = [f"T_3_{task_count+1}", home_pose, [f"T_3_{task_count}"], [], 'plan', 'open', False]
        self.r3_task_queue.append(home_task)
        self.task_lib[f"T_3_{task_count+1}"] = home_task




    def assignment_func(self, cur_arm_task:list, task_queue:deque, 
                        hand_pos:torch.Tensor, part_pos:torch.Tensor, timer=0, real_timer=0):
        if task_queue:
            if cur_arm_task is None or cur_arm_task[-1]:
                available_task = task_queue[0]
                assign_dependency = available_task[2]
                if len(assign_dependency) == 0:
                    cur_arm_task = task_queue.popleft()
                else:
                    task_status = []
                    for task_name in assign_dependency:
                        task_status.append(self.task_lib[task_name][-1])
                    if len(task_status) == sum(task_status):
                        cur_arm_task = task_queue.popleft()
                    else:
                        return cur_arm_task, task_queue
            
        # else:
        gripper_mode = cur_arm_task[-2]
        target = cur_arm_task[1][:, :3]
        hand_err = torch.norm(hand_pos - target, dim=-1)
        if int(cur_arm_task[0][2]) != 3:
            if gripper_mode == "close" or gripper_mode == "auto":
                part_dist = torch.norm(part_pos - hand_pos, dim=-1).unsqueeze(-1)
                gripper_sep = self.sim.dof_pos[:, int(cur_arm_task[0][2])-1, self.sim.robot_dof] + self.sim.dof_pos[:, int(cur_arm_task[0][2])-1, self.sim.robot_dof+1]
                gripped = (gripper_sep < 0.045) & (part_dist < self.sim.grasp_offset + 0.5 * 0.03)
                reached = (hand_err < 0.012).squeeze() & gripped.squeeze()

            else:
                reached = (hand_err < 0.03).squeeze()
            
            complete_dependency = cur_arm_task[3]
            if len(complete_dependency) == 0 and reached.item():
                self.task_lib[cur_arm_task[0]][-1] = True
            elif len(complete_dependency) > 0:
                task_status = []
                for task_name in complete_dependency:
                    task_status.append(self.task_lib[task_name][-1])
                if len(task_status) == sum(task_status):
                    self.task_lib[cur_arm_task[0]][-1] = True
        else:
            reached = (hand_err < 0.012).squeeze()
            complete_dependency = cur_arm_task[3]
            if reached.item() and gripper_mode == "auto":
                self.task_lib[cur_arm_task[0]][-1] = True if timer >= 2.0 and real_timer >= 2.0 else False
                # self.task_lib[cur_arm_task[0]][-1] = True if timer >= 2.0 else False
            
            if reached.item() and gripper_mode == "open":
                self.task_lib[cur_arm_task[0]][-1] = True


        return cur_arm_task, task_queue

    def task_assignment(self, timer=0, real_timer=0):
        # Process tasks for each robot.
        self.r1_task, self.r1_task_queue = self.assignment_func(self.r1_task, self.r1_task_queue, 
                                                                self.sim.hand1_tip_pos, self.sim.big_part_pos)
        if self.r2_task == None or int(self.r2_task[0][-1]) <= 2:
            arm2_part_pos = self.sim.L_part_pos
        else:
            arm2_part_pos = self.sim.R_part_pos
            self.sim.L_part = True
        if self.task_lib[f'T_3_{self.part1_weld_num + 1 + 2}'][-1]:
            self.sim.R_part = True
        self.r2_task, self.r2_task_queue = self.assignment_func(self.r2_task, self.r2_task_queue, 
                                                                self.sim.hand2_tip_pos, arm2_part_pos)

        L_hand_mate = self.r2_L_mate[1][:, :3]
        big_hand_mate = self.r1_mate[1][:, :3]
        arm1_mate_err = torch.norm(self.sim.hand1_tip_pos - big_hand_mate, dim=-1)
        arm2_mate_err = torch.norm(self.sim.hand2_tip_pos - L_hand_mate, dim=-1)
        if torch.sum(arm1_mate_err < 0.01).item() == self.sim.num_envs and torch.sum(arm2_mate_err < 0.01).item() == self.sim.num_envs:
            self.arm3_flag = True
        if self.arm3_flag:
            self.r3_task, self.r3_task_queue = self.assignment_func(self.r3_task, self.r3_task_queue,
                                                                    self.sim.weld_tip_pos, arm2_part_pos, timer, real_timer)
        self.cur_task = [self.r1_task, self.r2_task, self.r3_task]

        return self.cur_task