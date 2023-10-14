from langchain.retrievers import BM25Retriever
import torch


data_pack = torch.load('data_raw_embed.pt')
context_dict = dict()
for index, (text, embedding) in enumerate(data_pack):
    context_dict[index] = {
        'text': text,
        'embedding': embedding
    }

retriever = BM25Retriever.from_texts(
    [context_dict[key]['text'] for key in context_dict], k=256
)


embed_dict = dict()
for key in context_dict:
    data = context_dict[key]
    text = data['text']
    embed = data['embedding']
    embed_dict[text] = embed
    
    
true_value = 0
bm_emb = 0
count = 0
top_16 = 0




question_dict = dict()
count = 0
for data in qka_pack:
    keys = data['keys']
    question = data['question']
    answer = data['answer']

    embedding = t5_embedding(question)
    question_dict[count] = {
        'question': question,
        'embedding': embedding,
        'keys': keys
    }
    count += 1
    if count % 1000 == 0:
        print(count)
print(len(question_dict.keys()))



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
question_dict = torch.load('')
for key in question_dict:
    data = question_dict[key]
    keys = data['keys']
    question = data['question']
    results = retriever.get_relevant_documents(question)

    for result in results:
        text = result.page_content
        if keys in text:
            true_value += 1
            break

    for result in results[:16]:
        text = result.page_content
        if keys in text:
            top_16 += 1
            break

    vectors = [embed_dict[result.page_content] for result in results]
    vectors = np.array(vectors)

    query_vector = data['embedding']
    similarities = cosine_similarity([query_vector], vectors)
    k = 16
    top_indices = np.argsort(similarities[0])[-k:][::-1]

    for index in top_indices:
        text = results[index].page_content
        if keys in text:
            bm_emb += 1
            break
