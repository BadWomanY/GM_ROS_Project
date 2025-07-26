from isaacgym.torch_utils import *
import torch

device = "cuda:0"

"""=============================Spot weld locations in local frames============================="""
L_part_offset = torch.tensor([-0.05, -0.13, 0.0]).unsqueeze(0).to(device) # shape: (1, 3)
#TODO: Update this offset based on the simulation internal offset. 
R_part_offset = torch.tensor([0.05, 0.12, 0.0]).unsqueeze(0).to(device) # shape: (1, 3) 

L_part_welding_offset = torch.tensor([
    [0.0,  0.08, 0.0],
    [0.0, 0.06, 0.0],
    [0.0, -0.06, 0.0],
    [0.0, -0.08, 0.0],
    [0.04, -0.08, 0.0],
    [0.07, -0.08, 0.0],
    [0.1, -0.08, 0.0],
    ]).to(device)

R_part_welding_offset = torch.tensor([
    [0.01,  0.08, 0.0],
    [-0.01, 0.08, 0.0],
    [0.0, 0.06, 0.0],
    [0.0, -0.06, 0.0],
    [0.0, -0.08, 0.0]
]).to(device)

big_L_welding_offset = torch.tensor([
    [0.08,  0.0, 0.0],
    [0.06, 0.0, 0.0],
    [-0.06, 0.0, 0.0],
    [-0.08, 0.0, 0.0],
    [-0.08, 0.04, 0.0],
    [-0.08, 0.07, 0.0],
    [-0.08, 0.1, 0.0],
]).to(device)

big_R_welding_offset = torch.tensor([
    [0.08, 0.01, 0.0],
    [0.08, -0.01, 0.0],
    [0.06, 0.0, 0.0],
    [-0.06, 0.0, 0.0],
    [-0.08, 0.0, 0.0]
]).to(device)

big_L_welding_offset += L_part_offset
big_R_welding_offset += R_part_offset