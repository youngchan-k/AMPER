import pandas as pd
import numpy as np

import tqdm
import os
import random

import matplotlib.pyplot as plt
import time
import datetime


def plot_tag_distribution(question_csv):
    question_info = pd.read_csv(question_csv, usecols = ['question_id','tags'])
    question_tags, all_tag, prop_tag = [], [], []

    # List for proportion w.r.t. 'tag'
    for i in range(len(question_info)):
        question_tags = question_info.loc[i].tags.split(';')
        
        for i in range(len(question_tags)):
            all_tag.append(int(question_tags[i]))
            prop_tag.append(1/len(question_tags))       

    # Check the tag distribution for question as the number (ex. q1 - 1(1), 2(1), 179(1), 181(1))
    tag_distribution = {}

    for i in range(300+1):
        tag_distribution[str(i)] = 0

    for j in range(300+1):
        tag_distribution[str(j)] = all_tag.count(j)
              
    tag_data = pd.DataFrame(tag_distribution, index = ['number'])

    plt.bar(range(300+1), tag_data.loc['number'])
    plt.xlabel("tags")
    plt.ylabel("questions")
    plt.savefig('./question_tags.png')

    # Check the tag distribution for question as proportion (ex. q1 - 1(1/4), 2(1/4), 179(1/4), 181(1/4))
    tag_distribution_prop = {}

    for i in range(300+1):
        tag_distribution_prop[str(i)] = 0

    for i in range(len(all_tag)):
        for j in range(300+1):
            if all_tag[i] == j:
                tag_distribution_prop[str(j)] += prop_tag[i]
            

    tag_data_prop = pd.DataFrame(tag_distribution_prop, index = ['number'])

    plt.bar(range(300+1), tag_data_prop.loc['number'])
    plt.xlabel("tags")
    plt.ylabel("prop of questions")
    plt.savefig('./question_tags_prop.png')


def save_preprocessd_data(file_dir, save_dir, question_csv):
    # Choose the 1% of data randomly
    file_list = os.listdir(file_dir)
    file_list = random.sample(os.listdir(file_dir), len(file_list)//100)

    os.makedirs(save_dir, exist_ok=True)

    # Use only 'question_id', 'correct_answer', 'tags'
    question_info = pd.read_csv(question_csv, usecols = ['question_id','correct_answer','part','tags'])
    tag_data_re = tag_data_prop.transpose()

    # Delete the qudstion data if the proportion of tag is less than 20
    delete_tag = [] 
    for i in range(len(tag_data_re)):
        if tag_data_re.iloc[i].number < 20:
            delete_tag.append(i)

    delete_index = []
    for i in range(len(question_info)):
        question_tags = question_info.loc[i].tags.split(';')
        
        for j in range(len(question_tags)):
            if int(question_tags[j]) in delete_tag:
                delete_index.append(i)

    question_info_del = question_info.drop(index = delete_index)
    question_info_del = question_info_del.reset_index(drop = True)  
    question_info_del.to_csv("./csv/questions_process.csv", index = False)

    # Use EdNet KT1 data
    for file_name in file_list:
        user_list = os.path.join(file_dir + "/" + file_name)
        user_info = pd.read_csv(user_list)
        
        # Delete the column if there is no answer
        user_info = user_info.dropna(axis = 0)
        user_info = user_info.reset_index(drop = True)
    
        # Add the data of 'correct_answer' and 'tags' according to the question in row
        question_info_re = question_info.set_index('question_id')
        question_data = question_info_re.loc[user_info.loc[0:len(user_info)].question_id]
        user_data = pd.concat([user_info, question_data.reset_index(drop = True)], axis = 1)
        
        # Replace the data after adding 'user'
        user_data['user'] = [file_name.replace('.csv', '') for i in range(len(user_data))]
        user_data = user_data[['user', 'timestamp',  'question_id', 'part', 'tags', 'correct_answer', 'user_answer', 'elapsed_time']]
        
        delete_index_user = []
        for i in range(len(user_info)):
            if user_data.loc[i].question_id not in list(question_info_del.question_id):
                delete_index_user.append(i)
                
        user_data = user_data.drop(index = delete_index_user)
        user_data.to_csv(os.path.join(save_dir + "/" + file_name), index = False)

'''
def save_merged_data(file_dir, input_csv):
    all_filelist = os.listdir(file_dir)
    all_data = pd.DataFrame()

    # Merge all of the csv file
    for file_name in all_filelist:
        df = pd.read_csv(file_dir + '/' + file_name)
        all_data = all_data.append(df, ignore_index = True)

    all_data.to_csv(input_csv, index = False)
'''
    

def save_merged_data(file_dir, save_dir, input_csv):
    # Delete the remaining data (less than 21) after divided into the set consisting of 21 data
    os.makedirs(save_dir, exist_ok=True)
    
    data_list = os.listdir(file_dir)

    for file_name in data_list:
        user_list = os.path.join(file_dir + "/" + file_name)
        user_data = pd.read_csv(user_list)

        user_data = user_data.iloc[0:len(user_data) - len(user_data)%21]
    
    if len(user_data) != 0:
        user_data.to_csv(os.path.join(save_dir + "/" + file_name), index = False)

    all_filelist = os.listdir(save_dir)
    all_data = pd.DataFrame()

    count = []

    for file_name in all_filelist:
        df = pd.read_csv(save_dir + '/' + file_name)
        all_data = all_data.append(df, ignore_index = True)
        
        count.append(len(df)//21)

    all_data.to_csv(input_csv, index = False)


def split_data(input_csv, train_csv, label_csv):
    data = pd.read_csv(input_csv)

    correctness = []
    for i in range(len(data)):
        if data.loc[i].user_answer == data.loc[i].correct_answer:
            correctness.append('O')
        else:
            correctness.append('X')

    data['correctness'] = correctness

    
    # Split the data for 20(train) + 1(label)
    input_list = []

    start = 0
    while (start <= len(data)):
        end = start + 20
        input_list.append(data.loc[start:end])
        
        start = end + 1

    if len(input_list[-1]) < 21:
        input_list.pop()
        
    train_list, label_list = [], []
    for i in range(len(input_list)):
        train_list.append(input_list[i].iloc[0:-1])
        label_list.append(input_list[i].iloc[-1])

    # List -> DataFrame
    train_df = pd.concat([train_list[i] for i in range(len(train_list))])
    train_df = train_df.reset_index(drop = True)

    label_df = pd.concat([label_list[i] for i in range(len(label_list))], axis = 1).transpose()
    label_df = label_df.reset_index(drop = True)

    # DataFrame -> csv file
    train_df.to_csv(train_csv, index = False)
    label_df.to_csv(label_csv, index = False)


def main_data_preprocess():
    file_dir = "./data/EdNet-KT1/KT1"
    save_dir = "./data_preprocess"  
    save_dir_final = "./data_preprocess_final"  
    
    question_csv = "./csv/questions.csv"
    input_csv = "./csv/input_data.csv"
    
    print("Start data preprocessing\n")
    
    save_preprocessd_data(file_dir, save_dir, question_csv)
    save_merged_data(save_dir, save_dir_final, input_csv)
    
    print("End data preprocessing\n")