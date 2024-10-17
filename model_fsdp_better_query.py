import torch  
import os

from yarl import Query

from model_fsdp import DualModelTransformer
from model_fsdp_better_supervision import DualModelTransformerBetterSupervision
from SamplingMixin import SamplingMixin
# Set TOKENIZERS_PARALLELISM to true in environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["HF_TOKEN"] = "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"
import torch.nn as nn  
import copy
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
from typing import List, Optional, Union, Tuple  
import torch.nn.functional as F  
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
import torch.distributed as dist  
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  

class DualModelTransformerBetterQuery(DualModelTransformerBetterSupervision, SamplingMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init_subclass__(self, small_model_dim):
        logger.info("Initializing DualModelTransformerBetterQuery")
        # self.query_vector = nn.Parameter(torch.randn(small_model_dim), requires_grad=True)
        

    def get_query_vector(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        device = input_ids.device
        embedding_layer = self.embedding_layer  
        input_embeds = embedding_layer(input_ids)
        batch_size = input_embeds.size(0)
        query_dim = self.query_vector.weight.size(1)
        # query = self.query_vector.expand(batch_size, 1, query_dim)
        # query = self.query_vector.unsqueeze(0).unsqueeze(0).repeat(input_embeds.size(0), 1, 1)
        # check the device of query_vector
        # logger.info("Query vector device: %s", self.query_vector.weight.device)
        # logger.info("Input embeds device: %s", input_embeds.device)
        query = self.query_vector(torch.tensor([0], device=device)).expand(batch_size, 1, -1)
        input_embeds = torch.cat([input_embeds, query], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=device)], dim=1)
        model_in_eval_mode = not self.training
        position_ids = torch.arange(0, input_embeds.size(1), dtype=torch.long, device=device).unsqueeze(0).repeat(input_embeds.size(0), 1)  
        if model_in_eval_mode:
            with torch.no_grad():
                small_output = self.small_model(  
                    inputs_embeds=input_embeds,  
                    position_ids=position_ids,
                    attention_mask=attention_mask,  
                    output_hidden_states=True,  
                    use_cache=False,
                )  
        else:
            small_output = self.small_model(  
                inputs_embeds=input_embeds,  
                position_ids=position_ids,
                attention_mask=attention_mask,  
                output_hidden_states=True,  
                use_cache=False,
            )  
        small_last_hidden = self._get_last_hidden_state(small_output)  # [batch_size, hidden_size]  
        return small_last_hidden