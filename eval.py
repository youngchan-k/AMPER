from train import *
import re
import tqdm

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()

    return sentence


def evaluate(model, sentence, tokenizer, MAX_LENGTH=128):
    sentence = preprocess_sentence(sentence)
    
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0)


def predict(sentence, tokenizer):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


def eval_main(tokenizer, ckpt, user_question_1, answer_time_l, predict_csv, NUM_LAYERS=2, D_MODEL=128, NUM_HEADS=4, UNITS=256, DROPOUT=0.2, EPOCHS=10, BATCH_SIZE=256, BUFFER_SIZE=20000):
    VOCAB_SIZE = tokenizer.vocab_size + 2
    
    model = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, units=UNITS,
                        d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)
    
    model.load_weights(ckpt)

    tqdm.tqdm.pandas()
    label_predict = pd.DataFrame()

    label_predict['user_question'] = user_question_l
    predict_token = label_predict['user_question'].progress_apply(predict)

    label_predict['label'] = answer_time_l
    label_predict['predict'] = predict_token
    label_predict.to_csv(predict_csv, index = False)


if __name__ == "__main__":
    train_csv = "./csv/train_data_list.csv"
    label_csv = "./csv/label_data_list.csv"
    
    split_data(input_csv, train_csv, label_csv)
        
    user_question_t, answer_t = dataframe_to_list(train_csv, question_csv)
    user_question_l, answer_l = dataframe_to_list(label_csv, question_csv)
        
    ckpt = "./logs/checkpoint.ckpt"
    predict_csv = "./csv/predict.csv"
    tokenizer = create_tokenizer(user_question_t, answer_t)
    eval_main(tokenizer, ckpt, user_question_l, answer_l, predict_csv)