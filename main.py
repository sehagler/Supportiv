# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:03:35 2025

@author: sehag
"""

import os
import pickle

from question_answerer import Question_answerer_object

block_size = 128                        #block size for training
filename = 'mle_screening_dataset.csv'
max_length = 128                        #max length of generated text
model_name = 'gpt2'                     #smallest GPT-2 with 124M parameters
model_path = 'model'
num_sub_epochs = 1
num_super_epochs = 5
data_path = 'data'
per_device_train_batch_size = 8         #8 appears to be the default value
test_filename = 'test.csv'
top_k = 50                              #top k for generated text
top_p = 0.95                            #top p for generated text
train_filename = 'train.csv'
training_data_save_file = 'training.pkl'
validate_filename = 'validate.csv'

mode = 'query'

question_answerer = Question_answerer_object(model_name, max_length, top_k,
                                             top_p)

if question_answerer.is_cuda_available():
    if mode == 'split':
        question_answerer.split_data(data_path, filename, train_filename,
                                     test_filename, validate_filename)
    elif mode == 'train':
        test_mean_similarities = []
        train_mean_similarities = []
        question_answerer.load_train_data(data_path, train_filename, block_size)
        test_mean_similarity = \
            question_answerer.evaluate_on_data_set(data_path, test_filename)
        test_mean_similarity_best = test_mean_similarity
        question_answerer.save_pretrained(model_path)
        for _ in range(num_super_epochs):
            question_answerer.train_model(model_path, per_device_train_batch_size,
                                          num_sub_epochs)
            test_mean_similarity = \
                question_answerer.evaluate_on_data_set(data_path, test_filename)
            train_mean_similarity = \
                question_answerer.evaluate_on_data_set(data_path, train_filename)
            test_mean_similarities.append(test_mean_similarity)
            train_mean_similarities.append(train_mean_similarity)
            if test_mean_similarity > test_mean_similarity_best:
                test_mean_similarity_best = test_mean_similarity
                question_answerer.save_model()
        validate_mean_similarity = \
            question_answerer.evaluate_on_data_set(data_path, validate_filename)
        with open(os.path.join(data_path, training_data_save_file), "wb") as file:
            pickle.dump((test_mean_similarities, 
                         train_mean_similarities,
                         validate_mean_similarity), file)
    elif mode == 'query':
        question_answerer.load_model(model_path)
        questions = \
            question_answerer.ask_three_questions(data_path, train_filename)
        print('')
        print(question_answerer.generate_answer('Can a person inherit Juvenile Huntington disease ?'))
        print('')
        print(question_answerer.generate_answer('How is nemaline myopathy treated ?'))
        print('')
        print(question_answerer.generate_answer('How does Hailey-Hailey disease work ?'))