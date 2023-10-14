from converter import Embedding
import torch
from preprocessor import PreProcessor



text_pack = []
count = 0
preprocessor = PreProcessor(stop_words="./stopwords.csv", max_len=256, model_name='intfloat/multilingual-e5-large')

for file_name, context in torch.load('./raw_contexts.pt'):
    len_ = len(file_name)
    context = context[len_:].strip()
    for text in preprocessor.split(context):
        text = f"{file_name} | {text}"
        text_pack.append(text)
    count += 1
    if count % 1000 == 0:
        print(count)
print(len(text_pack))


data_pack = []
bs = 8
count = 0
pack = []
converter = Embedding()
while count < len(text_pack):
    pack.append(text_pack[count])
    count += 1
    if count % bs == 0:
        # print(pack)
        embeddings = converter.t5_embedding(pack)
        for i in range(bs):
            data_pack.append([pack[i], embeddings[i]])

        pack = []
    if count % 1000 == 0:
        print(count)

embeddings = converter.t5_embedding(pack)
for i in range(len(pack)):
    data_pack.append([pack[i], embeddings[i]])

torch.save(data_pack,'data_raw_embed_256.pt')