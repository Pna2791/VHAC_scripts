from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import uuid
from typing import Any, Iterable, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BM25_searcher:
    def __init__(self, host="http://localhost:9205", index_name="test") -> None:
        self.host = host
        self.index_name = index_name
        self.client = Elasticsearch(self.host)
        print(self.client.info().body)
        
    def count(self):
        return self.client.count(index=self.index_name)['count']
    
    def create(self, k1=1.5, b=0.75):
        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }
        # mappings = {
        #     "properties": {
        #         "content": {
        #             "type": "text",
        #             "similarity": "custom_bm25",  # Use the custom BM25 similarity
        #         },
        #         "raw_content": {
        #             "type": "text",  # You can change the type to match your data
        #         }
        #     }
        # }
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",  # Use the custom BM25 similarity
                },
                "raw_content": {
                    "type": "dense_vector",
                    "dims": 1024,
                }
            }
        }
        self.client.indices.delete(index=self.index_name, ignore=[400, 404])
        # Create the index with the specified settings and mappings
        self.client.indices.create(index=self.index_name, mappings=mappings, settings=settings)
    
    def add_data(
        self,
        data: Iterable[tuple],
        refresh_indices: bool = True,
    ) -> List[str]:
        requests = []
        ids = []
        for text, raw_text in data:
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": text,
                "raw_content": list(raw_text),
                "_id": _id,
            }
            ids.append(_id)
            requests.append(request)
        bulk(self.client, requests)

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids
    
    def search(self, query, index_name=None, top_k=8):
        query_dict = {
            "query": {"match": {"content": query}},
            "size": top_k
        }
        if index_name:
            res = self.client.search(index=index_name, body=query_dict)
        else:
            res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        for r in res["hits"]["hits"]:
            docs.append(r["_source"]["content"])
        return docs
    
    def search_embed(self, query, embedding, index_name=None, top_k=16, top_bm25=256):
        query_dict = {
            "query": {"match": {"content": query}},
            "size": top_bm25
        }
        if index_name:
            res = self.client.search(index=index_name, body=query_dict)
        else:
            res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        vectors = []
        for r in res["hits"]["hits"]:
            docs.append(r["_source"]["content"])
            vectors.append(r["_source"]["raw_content"])
        
        vectors = np.array(vectors)
        
        similarities = cosine_similarity([embedding], vectors)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        outputs = [docs[i] for i in top_indices]
        return outputs