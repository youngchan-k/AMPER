"""Training loop and data preparation for the transformer."""

import os

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from amper.model.transformer import transformer, CustomSchedule


def dataframe_to_list(csv_file, question_csv):
    train_df = pd.read_csv(csv_file)
    question_info = pd.read_csv(
        question_csv, usecols=["question_id", "correct_answer"]
    )

    question = {}
    for i in range(len(question_info)):
        question[question_info.loc[i].question_id] = question_info.loc[
            i
        ].correct_answer

    correct_answer = []
    correctness = []
    for i in range(len(train_df)):
        correct_answer.append(question[train_df.loc[i].question_id])
        if train_df.loc[i].user_answer == question[train_df.loc[i].question_id]:
            correctness.append("O")
        else:
            correctness.append("X")

    train_df["correct_answer"] = correct_answer
    train_df["correctness"] = correctness

    user_question = []
    answer = []
    for i in range(len(train_df)):
        user_question.append(
            train_df.loc[i].user + " " + train_df.loc[i].question_id
        )
        answer.append(train_df.loc[i].correctness)

    return user_question, answer


def create_tokenizer(inputs, outputs, MAX_LENGTH=128):
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        inputs + outputs, target_vocab_size=2**13
    )
    return tokenizer


def tokenize_input(tokenizer, inputs, outputs, MAX_LENGTH=128):
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [
        tokenizer.vocab_size + 1
    ]
    tokenized_inputs, tokenized_outputs = [], []

    for sentence1, sentence2 in zip(inputs, outputs):
        s1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        s2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        tokenized_inputs.append(s1)
        tokenized_outputs.append(s2)

    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
    )
    return tokenized_inputs, tokenized_outputs


def loss_function(y_true, y_pred, MAX_LENGTH=128):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred, MAX_LENGTH=128):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def train_main(
    user_question_t,
    answer_t,
    ckpt_dir,
    tokenizer,
    NUM_LAYERS=2,
    D_MODEL=128,
    NUM_HEADS=4,
    UNITS=256,
    DROPOUT=0.2,
    EPOCHS=10,
    BATCH_SIZE=256,
    BUFFER_SIZE=20000,
):
    tf.keras.backend.clear_session()

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "inputs": user_question_t,
                "dec_inputs": answer_t[:, :-1],
            },
            {"outputs": answer_t[:, 1:]},
        )
    )

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    VOCAB_SIZE = tokenizer.vocab_size + 2
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    )

    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer, loss=loss_function, metrics=[accuracy]
    )
    model.summary()

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
    )
    model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])


if __name__ == "__main__":
    from amper import config
    from amper.data.preprocess import main_data_preprocess, split_data

    main_data_preprocess()
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
    user_question_t, answer_t = tokenize_input(
        tokenizer, user_question_t, answer_t
    )
    train_main(user_question_t, answer_t, config.LOGS_DIR, tokenizer)
