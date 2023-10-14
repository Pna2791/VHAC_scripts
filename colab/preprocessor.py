from transformers import AutoTokenizer
import re
import pandas as pd

class PreProcessor:
    def __init__(self,model_name='vilm/vietcuna-3b', max_len=512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        
    def split(self, document, pre_fix=""):
        document = document.replace('\n', ' ')
        document = re.sub(' +', ' ', document)
        sentences = document.split('. ')
        context_list = []

        context = ""
        length = 0
        pre = ""
        len__ = 0
        for sentence in sentences:
            sentence += '. '
            
            tokens = self.tokenizer(sentence)
            len_ = len(tokens.input_ids)
            # len_ = len(sentence.split())
            if length + len_ > self.max_len:
                context_list.append(context)
                context = pre
                length = len__
            
            length += len_
            context += sentence

            pre = sentence
            len__ = len_
        context_list.append(pre_fix + context)

        return context_list
        