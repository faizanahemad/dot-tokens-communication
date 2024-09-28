import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true" 
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from torch.optim.lr_scheduler import OneCycleLR  
from datasets import load_dataset  
from transformers import AutoTokenizer  
from model_fsdp import DualModelTransformer, setup_fsdp, save_model, load_model  
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)
from model_strong_baselines import OneModelTransformer
from model_fsdp import DualModelTransformer
import random  
import numpy as np  
from tqdm import tqdm  
import logging  
from torch.utils.tensorboard import SummaryWriter  
import warnings  
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data.distributed import DistributedSampler  
# Set random seeds for reproducibility  
def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    
    
# Model definition function  
def create_model(config, model_cls, fsdp_config):  
    return model_cls(  
        config["large_model_name"],  
        config["small_model_name"],  
        config["stop_tokens"],  
        config["small_model_dim"],  
        config["large_model_dim"],  
        max(config["max_input_length"], config["max_output_length"]),  
        fsdp_config,
        enable_checkpointing=True,
        
    )  