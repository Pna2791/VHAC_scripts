import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import sys

def sys_print(ss):
    sys.stdout.write(ss)
    sys.stdout.flush()

device = 'cuda'
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

print("Start processing")
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(device)


import numpy as np
def t5_embedding(context):
    batch_dict = tokenizer(context, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict.to(device))
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)[0]
    return np.array(embeddings.cpu().detach())


import pandas as pd

file_path = open('data/test_path.txt').read()
df = pd.read_csv(file_path)

question_ebedding_dict = dict()
for question in df.question:
    embedding = t5_embedding(question)
    question_ebedding_dict[question] = embedding
    sys_print('.')

import torch
torch.save(question_ebedding_dict, 'data/embeddings.pt')
print("==="*20)
print("Convert to embedding vectors completed!")