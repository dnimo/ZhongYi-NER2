import torch.distributed as dist
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'

local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

