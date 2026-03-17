"""AMPER pipeline: preprocess → train → evaluate."""

import os

from amper import config
from amper.data.preprocess import main_data_preprocess, split_data
from amper.evaluation.eval import eval_main
from amper.training.train import (
    create_tokenizer,
    dataframe_to_list,
    tokenize_input,
    train_main,
)


def main():
    os.makedirs(config.CSV_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    print("Start data preprocessing!\n")
    main_data_preprocess()
    split_data(config.INPUT_CSV, config.TRAIN_CSV, config.LABEL_CSV)
    print("Finish data preprocessing!\n")

    print("Start training!\n")
    user_question_t, answer_t = dataframe_to_list(
        config.TRAIN_CSV, config.QUESTION_CSV
    )
    user_question_l, answer_l = dataframe_to_list(
        config.LABEL_CSV, config.QUESTION_CSV
    )

    tokenizer = create_tokenizer(user_question_t, answer_t)
    user_question_t, answer_t = tokenize_input(
        tokenizer, user_question_t, answer_t
    )
    train_main(user_question_t, answer_t, config.LOGS_DIR, tokenizer)
    print("Finish training!\n")

    print("Start evaluation!\n")
    eval_main(
        tokenizer,
        config.CHECKPOINT_PATH,
        user_question_l,
        answer_l,
        config.PREDICT_CSV,
    )
    print("Finish evaluation!\n")


if __name__ == "__main__":
    main()
