"""Question recommendation from user-question matrix and predictions."""

import random

import numpy as np
import pandas as pd

from amper import config


def user_question_matrix(input_csv, question_csv):
    user_info = pd.read_csv(input_csv)
    question_info = pd.read_csv(question_csv)

    delete_question = [
        i
        for i in range(len(question_info))
        if "-1" in str(question_info.loc[i].tags or "").split(";")
    ]
    question_info = question_info.drop(index=delete_question)
    question_info = question_info.reset_index(drop=True)
    question_info.to_csv(config.QUESTIONS_PROCESS_1_CSV, index=False)

    question_list = list(question_info["question_id"])
    user_list = []
    for value in user_info["user"]:
        if value not in user_list:
            user_list.append(value)

    user_dict = {u: [] for u in user_list}
    for i in range(len(user_info)):
        user_dict[user_info.loc[i].user].append(
            (user_info.loc[i].question_id, user_info.loc[i].correctness)
        )

    delete_user = [
        i
        for i in range(len(user_info))
        if len(user_dict[user_info.loc[i].user]) < 20
    ]
    user_info = user_info.drop(index=delete_user)
    user_info = user_info.reset_index(drop=True)
    user_info.to_csv(config.USER_PROCESS_CSV, index=False)

    user_list = []
    for value in user_info["user"]:
        if value not in user_list:
            user_list.append(value)

    user_data = pd.DataFrame()
    for u in user_list:
        user_data = pd.concat(
            [
                user_data,
                user_info[user_info["user"] == u].head(20),
            ],
            ignore_index=True,
        )
    user_data.to_csv(config.USER_DATA_CSV, index=False)

    user_dict_20 = {u: [] for u in user_list}
    for i in range(len(user_data)):
        user_dict_20[user_data.loc[i].user].append(
            (user_data.loc[i].question_id, user_data.loc[i].correctness)
        )

    matrix = np.empty((len(user_list), len(question_list)), object)
    for i in range(len(user_list)):
        for j in range(len(question_list)):
            if (question_list[j], "O") in user_dict_20[user_list[i]]:
                matrix[i][j] = "O"
            elif (question_list[j], "X") in user_dict_20[user_list[i]]:
                matrix[i][j] = "X"
            else:
                matrix[i][j] = ""

    np.save(config.USER_QUESTION_MATRIX_PATH, matrix)


def recommend_question(
    user,
    user_process_csv,
    question_csv,
    user_question_matrix_arr,
    predict_csv,
    user_random="u146540",
):
    user_info = pd.read_csv(user_process_csv)
    question_info = pd.read_csv(question_csv)

    try:
        question_info_matrix = pd.read_csv(config.QUESTIONS_PROCESS_1_CSV)
        question_list = list(question_info_matrix["question_id"])
    except FileNotFoundError:
        question_list = list(question_info["question_id"])

    user_list = list(dict.fromkeys(user_info["user"]))

    user_question_df = pd.DataFrame(user_question_matrix_arr)
    user_question_df.index = user_list
    user_question_df.columns = question_list[: user_question_df.shape[1]]

    ratio_wrong = []
    for q in user_question_df.columns:
        c_list = list(user_question_df[q])
        c = c_list.count("O")
        w = c_list.count("X")
        if c + w == 0:
            ratio_wrong.append("")
        else:
            ratio_wrong.append(w / (c + w))

    user_question_df.loc["wrong_ratio"] = ratio_wrong

    user_info = pd.read_csv(user_process_csv)
    question_info = pd.read_csv(question_csv)
    user_list = list(dict.fromkeys(user_info["user"]))
    user_list.append("wrong_ratio")
    question_list = list(question_info["question_id"])

    user_question_df.index = user_list
    user_question_df.columns = question_list[: user_question_df.shape[1]]

    user_list_no_ratio = [u for u in user_list if u != "wrong_ratio"]
    user_data = pd.DataFrame()
    for u in user_list_no_ratio:
        user_data = pd.concat(
            [
                user_data,
                user_info[user_info["user"] == u].head(20),
            ],
            ignore_index=True,
        )

    user_score = 0
    for i in range(len(user_data)):
        if (
            user_data.loc[i].correctness == "O"
            and user_data.loc[i].user == user
        ):
            q_id = user_data.loc[i].question_id
            if q_id in user_question_df.columns:
                user_score += 100 * float(
                    user_question_df.loc["wrong_ratio", q_id]
                )

    c_predict = pd.read_csv(predict_csv)
    question_score = {}

    for j in range(len(c_predict)):
        u = c_predict.loc[j].user_question.split(" ")[0]
        question = c_predict.loc[j].user_question.split(" ")[1]
        if c_predict.loc[j].predict == "O" and question in user_question_df.columns:
            predict_score = user_score + 100 * float(
                user_question_df.loc["wrong_ratio", question]
            )
            question_score[question] = predict_score

    question_score_sorted = sorted(
        question_score.items(), key=lambda item: item[1], reverse=True
    )
    if not question_score_sorted:
        print("User:", user_random, "No recommendations.")
        return
    score_max = max(s[1] for s in question_score_sorted)
    question_rec = [
        s[0] for s in question_score_sorted if s[1] == score_max
    ]

    random.seed(100)
    user_question_score = random.sample(
        question_rec, min(20, len(question_rec))
    )

    print("User : ", user_random)
    print("Recommended question : ", user_question_score)


if __name__ == "__main__":
    user_question_matrix(config.INPUT_CSV, config.QUESTION_CSV)

    user_info = pd.read_csv(config.USER_PROCESS_CSV)
    user_list = list(dict.fromkeys(user_info["user"]))
    user = user_list[random.randrange(0, len(user_list))]
    matrix = np.load(config.USER_QUESTION_MATRIX_PATH)

    recommend_question(
        user,
        config.USER_PROCESS_CSV,
        config.QUESTION_CSV,
        matrix,
        config.PREDICT_CSV,
    )
