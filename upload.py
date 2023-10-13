from my_BM25 import BM25_searcher

max_length = 512
host="http://localhost:9205"
index_name=f"vhac_embed"
retriever = BM25_searcher(host=host, index_name=index_name)
retriever.create()


import torch
pack = torch.load('../data/data_raw_embed.pt')


print("adding", len(pack), 'document')
retriever.add_data(pack)
print("number of data", retriever.count())