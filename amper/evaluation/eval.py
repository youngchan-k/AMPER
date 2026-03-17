"""Evaluation and prediction."""

import re

import pandas as pd
import tensorflow as tf
import tqdm

from amper.model.transformer import transformer


def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    return sentence.strip()


def evaluate(model, sentence, tokenizer, MAX_LENGTH=128):
    sentence = preprocess_sentence(sentence)
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [
        tokenizer.vocab_size + 1
    ]
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )
    output = tf.expand_dims(START_TOKEN, 0)

    for _ in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(model, sentence, tokenizer):
    """Predict correctness for a single sentence. Returns decoded string."""
    prediction = evaluate(model, sentence, tokenizer)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction.numpy() if i < tokenizer.vocab_size]
    )
    return predicted_sentence


def predict_row(row_data, model, tokenizer):
    """Predict for a single row (used with DataFrame.progress_apply)."""
    sentence = row_data["user_question"]
    return predict(model, sentence, tokenizer)


def eval_main(
    tokenizer,
    ckpt,
    user_question_l,
    answer_l,
    predict_csv,
    NUM_LAYERS=2,
    D_MODEL=128,
    NUM_HEADS=4,
    UNITS=256,
    DROPOUT=0.2,
    EPOCHS=10,
    BATCH_SIZE=256,
    BUFFER_SIZE=20000,
):
    VOCAB_SIZE = tokenizer.vocab_size + 2
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )
    model.load_weights(ckpt)

    tqdm.tqdm.pandas()
    label_predict = pd.DataFrame()
    label_predict["user_question"] = user_question_l
    label_predict["label"] = answer_l
    def _predict(s):
        return predict_row({"user_question": s}, model, tokenizer)

    label_predict["predict"] = label_predict["user_question"].progress_apply(
        _predict
    )
    label_predict.to_csv(predict_csv, index=False)


if __name__ == "__main__":
    from amper import config
    from amper.data.preprocess import split_data
    from amper.training.train import (
        dataframe_to_list,
        create_tokenizer,
        tokenize_input,
    )

    split_data(
        config.INPUT_CSV,
        config.TRAIN_CSV,
        config.LABEL_CSV,
    )

    user_question_t, answer_t = dataframe_to_list(
        config.TRAIN_CSV, config.QUESTION_CSV
    )
    user_question_l, answer_l = dataframe_to_list(
        config.LABEL_CSV, config.QUESTION_CSV
    )

    tokenizer = create_tokenizer(user_question_t, answer_t)
    eval_main(
        tokenizer,
        config.CHECKPOINT_PATH,
        user_question_l,
        answer_l,
        config.PREDICT_CSV,
    )
