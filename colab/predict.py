import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

def sys_print(ss):
    sys.stdout.write(ss)
    sys.stdout.flush()

seq_length = 2048
model_id = "bigscience/bloomz-3b"
lora_id = "CreatorPhan/Bloomz_Lora"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', max_length=seq_length)


print("Loading model", lora_id)
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
model_lora = PeftModel.from_pretrained(model, lora_id)
print("Loading completed", lora_id)

def generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_len = len(input_ids[0])
    input_ids = input_ids.to(model_lora.device)

    output = model_lora.generate(input_ids=input_ids, max_new_tokens=256)[0]
    answer = tokenizer.decode(output.cpu()[input_len:], skip_special_tokens=True)
    return answer


context_dict = torch.load('data/context_dict.pt')
answer_dict = dict()
for index, question in enumerate(context_dict):
    docs = context_dict[question]
    print(">>>", question)

    for i in range(14):
        context = docs[i] + docs[i+1] + docs[i+2]
        prompt = f"Dựa vào thông tin <<< {context} >>> \n\n Hãy trả lời câu hỏi sau <<< {question} >>>. Nếu không có thông tin thì hãy trả lời là <<< Không có thông tin >>> . \n\n Câu trả lời của bạn là: "
        output = generate(prompt)
        if "hông có thông tin" not in output:
            answer_dict[index] = output
            print(output)
            break


quest_id = []
answer_predict = []
for index in answer_dict:
    answer = answer_dict[index]
    quest_id.append(index)
    answer_predict.append(answer)
    
import pandas as pd
data = {'quest_id': quest_id, 'answer_predict': answer_predict}

# Create a DataFrame
df = pd.DataFrame(data)
df.to_csv('results.csv', index=False)
print("==="*20)
print("DONE")
