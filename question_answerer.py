# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:08:13 2025

@author: sehag
"""

import math
import numpy as np
import os
import pandas as pd
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#
class Question_answerer_object(object):
    
    #
    def __init__(self, model_name, max_length, top_k, top_p):
        self.data_set = None
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.load_model(model_name)
        self.data_collator = \
            DataCollatorForLanguageModeling(tokenizer=self.tokenizer, 
                                            mlm=False)
        self.trainer = None
        
    #
    def ask_three_questions(self, path, filename):
        df = pd.read_csv(os.path.join(path, filename))
        
        # The data were randomly ordered in self.split_data so we can 
        # just take the first three questions.
        df = df[0:3]
        
        questions = []
        for i in range(len(df)):
            question = df.iloc[i,1]
            answer_0 = df.iloc[i,2]
            answer_1 = self.generate_answer(question)
            questions.append([ question, answer_0, answer_1 ])
        return questions
        
    #
    def evaluate_on_data_set(self, path, filename):
        similarity = []
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        df = pd.read_csv(os.path.join(path, filename))
        
        # This is taking too long so let's reduce the size of the data to test
        # this.  The data were randomly ordered in self.split_data so we can 
        # just take the first few.
        df = df[0:128]
        
        for i in range(len(df)):
            question = df.iloc[i,1]
            answer_0 = df.iloc[i,2]
            answer_1 = self.generate_answer(question)
            answers = [ answer_0, answer_1 ]
            answer_embeddings = model.encode(answers)
            similarity.append(cosine_similarity([answer_embeddings[0]],
                                                [answer_embeddings[1]])[0][0])
        similarity = np.mean(similarity)
        return similarity
    
    #
    def generate_answer(self, question):
        self.model.to('cuda')
        ids = self.tokenizer.encode(question, return_tensors='pt')
        ids = ids.to('cuda')
        answer = self.model.generate(ids,
                                     do_sample=True,
                                     max_length=self.max_length,
                                     pad_token_id=self.model.config.eos_token_id,
                                     top_k=self.top_k,
                                     top_p=self.top_p)
        answer = self.tokenizer.decode(answer[0], skip_special_tokens=True)
        answer = answer[len(question)+1:]
        return answer
            
    #
    def is_cuda_available(self):
        is_cuda_available = torch.cuda.is_available()
        if is_cuda_available:
            print('CUDA is available')
            torch.device('cuda')
        else:
            print('CUDA is not available')
        return is_cuda_available
            
    #
    def load_model(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
    #
    def load_train_data(self, path, filename, block_size):
        self.data_set = TextDataset(tokenizer=self.tokenizer,
                                    file_path=os.path.join(path, filename),
                                    block_size=block_size)
        
    #
    def save_model(self):
        if self.trainer is not None:
            self.trainer.save_model()
        else:
            print("You must first train a model")
            
    #
    def save_pretrained(self, output_dir):
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
            
    #
    def split_data(self, path, filename, train_filename, test_filename, 
                   validate_filename):
        df = pd.read_csv(os.path.join(path, filename))
        df = df.sample(frac=1)                        #randomly order the data
        chunk_length = math.floor(len(df)/3)
        train_df = df[0:chunk_length]
        test_df = df[chunk_length:2*chunk_length]
        validate_df = df[2*chunk_length:]
        train_df.to_csv(os.path.join(path, train_filename))
        test_df.to_csv(os.path.join(path, test_filename))
        validate_df.to_csv(os.path.join(path, validate_filename))
    
    #
    def train_model(self, output_dir, per_device_train_batch_size,
                    num_train_epochs):
        args = TrainingArguments(output_dir=output_dir,
                                 overwrite_output_dir=False,
                                 per_device_train_batch_size=per_device_train_batch_size,
                                 num_train_epochs=num_train_epochs)
        if self.data_set is not None:
            self.trainer = Trainer(model=self.model,
                                   args=args,
                                   data_collator=self.data_collator,
                                   train_dataset=self.data_set)
            self.trainer.train()
        else:
            print("You must first read in a data set.")