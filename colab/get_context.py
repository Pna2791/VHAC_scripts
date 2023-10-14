import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import BM25Retriever

data_pack = torch.load('data/data_raw_embed_256.pt')
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
    




context_dict = dict()

question_ebedding_dict = torch.load('data/embeddings.pt')
for question in question_ebedding_dict:
    # Get 256 related documents
    results = retriever.get_relevant_documents(question)
    vectors = [embed_dict[result.page_content] for result in results]
    vectors = np.array(vectors)

    # Get 16 related documents
    query_vector = question_ebedding_dict[question]
    similarities = cosine_similarity([query_vector], vectors)
    k = 16
    top_indices = np.argsort(similarities[0])[-k:][::-1]
    
    context_list = []
    for index in top_indices:
        text = results[index].page_content
        context_list.append(text)
    context_dict[question] = context_list
    
    
torch.save(context_dict, 'data/context_dict.pt')