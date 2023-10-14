import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np


class Embedding:
    def __init__(self):
        self.device = 'cuda'
        
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(self.device)
    
    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]



    def t5_embedding(self, context):
        batch_dict = self.tokenizer(context, max_length=512, padding=True, truncation=True, return_tensors='pt')
        # print(batch_dict.input_ids.shape)
        outputs = self.model(**batch_dict.to(self.device))
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return np.array(embeddings.cpu().detach())
