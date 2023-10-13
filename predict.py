import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

def sys_print(ss):
    sys.stdout.write(ss)
    sys.stdout.flush()

seq_length = 2048
model_id = "bigscience/bloomz-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', max_length=seq_length)

print("Loading CreatorPhan/Bloomz_lora_answer")
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
model_lora = PeftModel.from_pretrained(model, "CreatorPhan/Bloomz_lora_answer")

def generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model_lora.device)

    output = model_lora.generate(input_ids=input_ids, max_length=seq_length)[0]
    answer = tokenizer.decode(output.cpu(), skip_special_tokens=True)

    return answer[len(prompt):]



from my_BM25 import BM25_searcher
elasticsearch_url="http://localhost:9205"
index_name="vhac_embed"
retriever = BM25_searcher(host=elasticsearch_url, index_name=index_name)


import pandas as pd
file_path = open('../data/test_path.txt').read()
df = pd.read_csv(file_path)

question_embedding_dict = torch.load('../data/embeddings.pt')
question_request_dict = torch.load('../data/quest_explained.pt')
answer_dict = dict()

for index, question in enumerate(df.question):
    embedding = question_embedding_dict[question]
    request = question_request_dict[question]
    docs = retriever.search_embed(query=question, embedding=embedding, top_k=16)
    # print(question)
    sys_print('.')

    for i in range(14):
        context = docs[i] + docs[i+1] + docs[i+2]
        prompt = f"Dựa vào thông tin >>> {context}\n\nHãy trả lời câu hỏi sau >>> {question}\nPhải đảm bảo các yêu cầu sau >>> {request}\nCâu trả lời của bạn là: "
        output = generate(prompt)
        if "NOINFO" not in output:
            answer_dict[index] = output
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
