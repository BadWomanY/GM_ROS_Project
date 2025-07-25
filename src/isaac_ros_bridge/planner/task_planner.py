from isaacgym import gymtorch
import torch
from collections import deque
from isaac_ros_bridge.utils.franka_utils import *
from isaac_ros_bridge.models.spot_weld_offsets import *
from IPython import embed

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

    def create_task_lib_r1(self):
        # Robot 1 sub-tasks
        big_part_grasp_pos = self.sim.big_part_pos.clone()
        big_part_grasp_pos[:, 2] += self.sim.grasp_offset  # Grasping position.
        big_part_grasp_rot = self.sim.goal_rot_gripp1  # Grasping orientation in quat.
        big_part_grasp_pose = torch.cat([big_part_grasp_pos, big_part_grasp_rot], dim=1)
        self.r1_grasp = ['T_1_1', big_part_grasp_pose, [], [], 'no-plan', 'auto', False]
        self.r1_task_queue.append(self.r1_grasp)
        self.task_lib['T_1_1'] = self.r1_grasp

        hand1_mate_pos = self.sim.big_part_anchor + quat_rotate(self.sim.big_part_goal_quat, self.hand1_to_part_offset)
        hand1_mate_rot = self.sim.q_hand1_goal
        hand1_mate_pose = torch.cat([hand1_mate_pos, hand1_mate_rot], dim=1)
        self.r1_mate = ['T_1_2', hand1_mate_pose, ['T_1_1'], [f"T_3_{2*self.part1_weld_num}"], 'plan', 'close', False]
        self.r1_task_queue.append(self.r1_mate)
        self.task_lib['T_1_2'] = self.r1_mate

        # Hand 1 change orientation for part 2 welding.
        hand1_mate_rot2 = self.sim.q_hand1_goal2
        hand1_mate_pose2 = torch.cat([hand1_mate_pos, hand1_mate_rot2], dim=1)
        self.r1_mate = ['T_1_3', hand1_mate_pose2, ['T_1_2'], ['END'], 'no-plan', 'close', False]
        self.r1_task_queue.append(self.r1_mate)
        self.task_lib['T_1_3'] = self.r1_mate

    def create_task_lib_r2(self):
        # Robot 2 sub-tasks
        L_part_grasp_pos = self.sim.L_part_pos + quat_rotate(self.sim.L_part_rot, self.sim.part_grasp_offset)
        L_part_grasp_pos[:, 2] += self.sim.grasp_offset_arm2  # Grasping position.
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
                                              self.sim.hand_to_small,
                                              part_offset,
                                              weld_big_L,
                                              weld_L,
                                              self.device)
        
        part_q = self.sim.L_part_rot.clone()
        self.hand2_to_Lpart_rot_offset = quat_mul(quat_conjugate(self.sim.goal_rot_gripp2), part_q)
        hand2_mate_rot = quat_mul(self.L_part_goal_rot, quat_conjugate(self.sim.goal_rot_gripp2))
        self.L_part_goal_rot = quat_mul(hand2_mate_rot, quat_conjugate(self.hand2_to_Lpart_rot_offset))

        hand2_mate_pose = torch.cat([hand2_mate_pos, hand2_mate_rot], dim=1)
        self.r2_L_mate = ['T_2_2', hand2_mate_pose, ['T_2_1'], ['T_3_4'], 'plan', 'close', False]
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
        R_part_grasp_pos[:, 2] += self.sim.grasp_offset_arm2  # Grasping position.
        part_grasp_rot = self.sim.goal_rot_gripp2  # Grasping orientation in quat for small part grasping.
        R_part_grasp_pose = torch.cat([R_part_grasp_pos, part_grasp_rot], dim=1)
        self.r2_L_grasp = ['T_2_4', R_part_grasp_pose, ['T_2_3'], [], 'no-plan', 'auto', False]
        self.r2_task_queue.append(self.r2_L_grasp)
        self.task_lib['T_2_4'] = self.r2_L_grasp

        weld_big_R = self.sim.big_part_welding_pos[:, 7:]
        weld_R = self.sim.R_part_welding_pos
        part_offset = R_part_offset
        hand2_mate_pos, self.R_part_goal_rot = part_mate_hand_pos(self.sim.big_part_anchor,
                                              self.sim.big_part_goal_quat,
                                              self.sim.hand_to_small,
                                              part_offset,
                                              weld_big_R,
                                              weld_R,
                                              self.device)
        
        part_q = self.sim.R_part_rot.clone()
        self.hand2_to_part_rot_offset = quat_mul(quat_conjugate(self.sim.goal_rot_gripp2), part_q)
        hand2_mate_rot = quat_mul(self.R_part_goal_rot, quat_conjugate(self.sim.goal_rot_gripp2))
        self.R_part_goal_rot = quat_mul(hand2_mate_rot, quat_conjugate(self.hand2_to_part_rot_offset))

        hand2_mate_pose = torch.cat([hand2_mate_pos, hand2_mate_rot], dim=1)
        self.r2_L_mate = ['T_2_5', hand2_mate_pose, ['T_2_4'], [], 'plan', 'close', False]
        self.r2_task_queue.append(self.r2_L_mate)
        self.task_lib['T_2_5'] = self.r2_L_mate
    
    def create_task_lib_r3(self):
        # Robot 3 sub-tasks
        """Part 1 welding tasks."""
        weld_order = [0, 3, 1, 2, 4, 5, 6]
        hover_offset = 0.15

        desired_z = quat_axis(self.L_part_goal_rot, 0)
        reference_up = quat_axis(self.L_part_goal_rot, 2)
        weld_top = look_at_quat(desired_z, reference_up)

        desired_z = quat_axis(self.L_part_goal_rot, 1)
        weld_side = look_at_quat(desired_z, reference_up)
        for i, idx in enumerate(weld_order):
            weld_rot = weld_top if idx <= 3 else weld_side
            weld_pos = self.sim.big_part_welding_pos[:, idx, :].clone()
            hover_weld_pos = weld_pos.clone()
            if idx <= 3:
                weld_pos[:, 2] += self.sim.grasp_offset_arm2
                hover_weld_pos[:, 2] += hover_offset
            else:
                weld_pos[:, 0] -= self.sim.grasp_offset_arm2
                hover_weld_pos[:, 0] -= hover_offset
            hover_weld_pose = torch.cat([hover_weld_pos, weld_rot], dim=1)
            weld_pose = torch.cat([weld_pos, weld_rot], dim=1)

            task_name = f"T_3_{2*i+1}"
            depends_on = [] if i == 0 else [f"T_3_{2*i}"]
            task = [task_name, hover_weld_pose, depends_on, [], 'plan', 'open', False]
            self.task_lib[task_name] = task
            self.r3_task_queue.append(task)
            task_name = f"T_3_{2*i+2}"
            depends_on = ['T_3_1'] if i == 0 else [f"T_3_{2*i+1}"]
            task = [task_name, weld_pose, depends_on, [], 'no-plan', 'auto', False]
            self.task_lib[task_name] = task

            self.r3_task_queue.append(task)
        
        """Welding return home after part 1 welding."""
        home_pos = self.sim.init_pos[:, 2]
        home_rot = weld_rot
        home_pose = torch.cat([home_pos, home_rot], dim=1)
        welding_task_num = len(self.r3_task_queue)
        self.r3_return = [f'T_3_{welding_task_num + 1}', home_pose, [f'T_3_{welding_task_num}'], [], 'no-plan', 'open', False]
        self.r3_task_queue.append(self.r3_return)
        self.task_lib[f'T_3_{welding_task_num + 1}'] = self.r3_return

        """Part 2 welding tasks."""
        #TODO
    
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
                gripper_mode = cur_arm_task[-2]
                target = cur_arm_task[1][:, :3]
                hand_err = torch.norm(hand_pos - target, dim=-1)
                if int(cur_arm_task[0][2]) != 3:
                    if gripper_mode == "close" or gripper_mode == "auto":
                        part_dist = torch.norm(part_pos - hand_pos, dim=-1).unsqueeze(-1)
                        gripper_sep = self.sim.dof_pos[:, 0, self.sim.robot_dof] + self.sim.dof_pos[:, 0, self.sim.robot_dof+1]
                        gripped = (gripper_sep < 0.045) & (part_dist < self.sim.grasp_offset + 0.5 * 0.03)
                        reached = (hand_err < 0.03).squeeze() & gripped.squeeze()

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
                                                                self.sim.hand1_pos, self.sim.big_part_pos)
        if self.r2_task == None or int(self.r2_task[0][-1]) <= 2:
            arm2_part_pos = self.sim.L_part_pos
        else:
            arm2_part_pos = self.sim.R_part_pos
            self.sim.next_part = True
        self.r2_task, self.r2_task_queue = self.assignment_func(self.r2_task, self.r2_task_queue, 
                                                                self.sim.hand2_pos, arm2_part_pos)
        
        L_big_dist = torch.norm(self.sim.big_part_pos - self.sim.L_part_pos, dim=-1).unsqueeze(-1)
        if torch.sum(L_big_dist < 0.18).item() == self.sim.num_envs:
            self.r3_task, self.r3_task_queue = self.assignment_func(self.r3_task, self.r3_task_queue,
                                                                    self.sim.hand3_pos, arm2_part_pos, timer, real_timer)
        self.cur_task = [self.r1_task, self.r2_task, self.r3_task]
        # print(self.task_lib['T_2_2'])
        return self.cur_task