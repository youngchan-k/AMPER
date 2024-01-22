from data_preprocess import *
from transformer import *
from train import *
from eval import *
from recommendation import *


## Data preprocessing
print("Start data preprocessing!\n")

main_data_preprocess()

train_csv = "./csv/train_data_list.csv"
label_csv = "./csv/label_data_list.csv"

split_data(input_csv, train_csv, label_csv)

print("Finish data preprocessing!\n")


## Train
print("Start training!\n")

user_question_t, answer_t = dataframe_to_list(train_csv, question_csv)
user_question_l, answer_l = dataframe_to_list(label_csv, question_csv)

tokenizer = create_tokenizer(user_question_t, answer_t)
user_question_t, answer_t = tokenize_input(tokenizer)

ckpt_dir = "./logs"
train_main(user_question_t, answer_t, ckpt_dir, tokenizer)

print("Finish training!\n")


## Evaluation
print("Start evaluation!\n")

ckpt = "./logs/checkpoint.ckpt"
predict_csv = "./csv/predict.csv"
tokenizer = create_tokenizer(user_question_t, answer_t)

eval_main(tokenizer, ckpt, user_question_l, answer_l, predict_csv)

print("Finish evaluation!\n")