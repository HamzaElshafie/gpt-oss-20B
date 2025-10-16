import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from dataclasses import dataclass
import math
import os
import json

k = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
print(k)
print(k.shape)
k = k.repeat_interleave(4, dim=2)
print(k)
print(k.shape)