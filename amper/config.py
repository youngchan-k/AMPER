"""Project paths and constants. All paths are relative to the project root."""

import os

# Project root (directory containing amper/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV paths: only questions.csv is required; the rest are created by the pipeline
CSV_DIR = os.path.join(PROJECT_ROOT, "csv")
QUESTION_CSV = os.path.join(CSV_DIR, "questions.csv")
INPUT_CSV = os.path.join(CSV_DIR, "input_data.csv")
TRAIN_CSV = os.path.join(CSV_DIR, "train_data_list.csv")
LABEL_CSV = os.path.join(CSV_DIR, "label_data_list.csv")
PREDICT_CSV = os.path.join(CSV_DIR, "predict.csv")

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EdNet-KT1", "KT1")
DATA_PREPROCESS_DIR = os.path.join(PROJECT_ROOT, "data_preprocess")
DATA_PREPROCESS_FINAL_DIR = os.path.join(PROJECT_ROOT, "data_preprocess_final")

# Checkpoints and outputs
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_PATH = os.path.join(LOGS_DIR, "checkpoint.ckpt")

# Recommendation outputs
USER_QUESTION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "user_question_matrix.npy")
USER_PROCESS_CSV = os.path.join(CSV_DIR, "user_process.csv")
QUESTIONS_PROCESS_CSV = os.path.join(CSV_DIR, "questions_process.csv")
QUESTIONS_PROCESS_1_CSV = os.path.join(CSV_DIR, "questions_process_1.csv")
USER_DATA_CSV = os.path.join(CSV_DIR, "user_data.csv")
