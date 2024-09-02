import pandas as pd
import numpy as np
import random

def user_question_matrix(input_csv, question_csv):
    user_info = pd.read_csv(input_csv)
    question_info = pd.read_csv(question_csv)
    
    # Delete if tag == -1
    delete_question = []
    for i in range(len(question_info)):
        if '-1' in question_info.loc[i].tags.split(';'):
            delete_question.append(i)

    question_info = question_info.drop(index = delete_question)
    question_info = question_info.reset_index(drop = True)
    question_info.to_csv("./csv/questions_process_1.csv", index = False)

    question_list = list(question_info['question_id'])
    user_list = []
    for value in list(user_info['user']):
        if value not in user_list:
            user_list.append(value)

    user_dict = {}
    for i in range(len(user_list)):
        user_dict[user_list[i]] = []

    for i in range(len(user_info)):
        user_dict[user_info.loc[i].user].append((user_info.loc[i].question_id, user_info.loc[i].correctness))

    # Delete the user if data length is less than 20
    delete_user = []
    for i in range(len(user_info)):
        if len(user_dict[user_info.loc[i].user]) < 20:
            delete_user.append(i)

    user_info = user_info.drop(index = delete_user)
    user_info = user_info.reset_index(drop = True)
    user_info.to_csv("./csv/user_process.csv", index = False)

    user_list = []
    for value in list(user_info['user']):
        if value not in user_list:
            user_list.append(value)

    # Choose the first top 20 solving data for all users
    user_data = pd.DataFrame()
    for i in range(len(user_list)):
        user_data_add = user_info[user_info['user'] == user_list[i]].head(20)
        user_data = pd.concat([user_data, user_data_add], ignore_index = True)

    user_data.to_csv("./csv/user_data.csv", index = False)


    user_dict_20 = {}
    for i in range(len(user_list)):
        user_dict_20[user_list[i]] = []

    for i in range(len(user_data)):
        user_dict_20[user_data.loc[i].user].append((user_data.loc[i].question_id, user_data.loc[i].correctness))

    # User-Question matrix
    user_question_matrix = np.empty((len(user_list), len(question_list)), object)
    for i in range(len(user_list)):
        for j in range(len(question_list)):
            if (question_list[j], 'O') in user_dict_20[user_list[i]]:
                user_question_matrix[i][j] = 'O'
                
            elif (question_list[i], 'X') in user_dict_20[user_list[i]]:
                user_question_matrix[i][j] = 'X'
                
            else:
                user_question_matrix[i][j] = ''

    np.save("./user_question_matrix.npy", user_question_matrix)


def recommend_question(user, user_process_csv, question_csv, user_question_matrix, predict_csv, user_random='u146540'):
    user_info = pd.read_csv(user_process_csv)
    question_info = pd.read_csv(question_csv)
    
    user_question_df = pd.DataFrame(user_question_matrix)
    user_question_df.index = [user_list[i] for i in range(len(user_list))]
    user_question_df.columns = [question_list[i] for i in range(len(question_list))]
    
    # Proportion of wrongness w.r.t. question
    ratio_wrong = []

    for i in range(len(question_list)):
        c_list = list(user_question_df[question_list[i]])
        
        c = c_list.count('O')
        w = c_list.count('X')
        if c + w == 0:
            ratio_wrong.append('')
        else:
            ratio_wrong.append(w / (c + w))

    user_question_df.loc['wrong_ratio'] = ratio_wrong

    # Calculate the current score of user
    user_info = pd.read_csv(user_process_csv)
    question_info = pd.read_csv(question_csv)
    
    user_list = list(set(user_info['user']))
    user_list.append('wrong_ratio')
    question_list = list(question_info['question_id'])

    user_question_df.index = [user_list[i] for i in range(len(user_list))]
    user_question_df.columns = [question_list[i] for i in range(len(question_list))]
    
    user_list = []
    for value in list(user_info['user']):
        if value not in user_list:
            user_list.append(value)
    
    user_data = pd.DataFrame()
    for i in range(len(user_list)):
        user_data_add = user_info[user_info['user'] == user_list[i]].head(20)   # 20°³ ¼±ÅÃ
        user_data = pd.concat([user_data, user_data_add], ignore_index = True)
  
    user_score = 0
    for i in range(len(user_info)):
        if user_data.loc[i].correctness == 'O' and user_data.loc[i].user == user:
            
            # Function = 100 * proportion of wrongness for correct answer question
            user_score += 100 * float(user_question_df.loc['wrong_ratio', user_data.loc[i].question_id])
            
    c_predict = pd.read_csv(predict_csv)

    # Calculate the score w.r.t. question_no using dictionary
    question_score = {}
    for j in range(len(c_predict)):
        user = c_predict.loc[j].user_question.split(" ")[0]  # User
        question = c_predict.loc[j].user_question.split(" ")[1]  # Question no
    
        if c_predict.loc[j].predict == 'O':
            predict_score = user_score + 100 * float(user_question_df.loc['wrong_ratio', question])
            question_score[question] = predict_score

    # Choose the top 20 questions
    question_score = sorted(question_score.items(), key = lambda item: item[1], reverse = True)
    score_max = max(([question_score[i][1] for i in range(len(question_score))]))

    question_rec = [question_score[i][0] for i in range(len(question_score)) if question_score[i][1] == score_max]

    random.seed(100)
    user_question_score = random.sample(question_rec, 20)

    print('User : ', user_random)
    print('Recommended question : ', user_question_score)


if __name__ == "__main__":
    input_csv = "./csv/input_data.csv"
    question_csv = "./csv/questions.csv"
    
    user_question_matrix(input_csv, question_csv)
    
    user_process_csv = "./csv/user_process.csv"
    user_info = pd.read_csv(user_process_csv)

    user_list = []
    for value in list(user_info['user']):
        if value not in user_list:
            user_list.append(value)
    
    user = user_list[random.randrange(0, len(user_list))]
    user_question_matrix = np.load("./user_question_matrix.npy")
    predict_csv = "./csv/predict.csv"
    
    recommend_question(user, input_csv, question_csv, user_question_matrix, predict_csv)
    