import pandas as pd
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable, grad
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor
    
    
