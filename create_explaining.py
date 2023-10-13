import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

def sys_print(ss):
    sys.stdout.write(ss)
    sys.stdout.flush()

print("Loading CreatorPhan/Bloomz_lora_question")
seq_length = 256
model_id = "bigscience/bloomz-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', max_length=seq_length)

from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    max_length=seq_length,
    device_map={"": 0}
)
model_lora = PeftModel.from_pretrained(model, "CreatorPhan/Bloomz_lora_question")

def get_request(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model_lora.device)

    output = model_lora.generate(input_ids=input_ids, max_length=seq_length)[0]
    if len(output) > 64:
        output = output[:64]
    answer = tokenizer.decode(output.cpu(), skip_special_tokens=True)

    return answer[len(prompt):]


import pandas as pd
file_path = open('../data/test_path.txt').read()
df = pd.read_csv(file_path)

question_request_dict = dict()
for question in df.question:
    requests = get_request(f"Liệt kê các yêu cầu của câu hỏi sau: {question}\n\n")
    question_request_dict[question] = requests
    sys_print('.')
    # print(question, '|', requests)

import torch
torch.save(question_request_dict, '../data/quest_explained.pt')
print("==="*20)
print("Explaining completed!")