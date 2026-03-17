# AMPER

**Aim → Measure → Predict → Evaluate → Recommend** — a minimally domain-dependent, privacy-preserving framework for Digital Me services.

Digital Me is an AI-driven service that manages user goals in real time: it **measures** current status, **predicts** future states, **evaluates** outcomes, and **recommends** actions to improve them. AMPER implements this paradigm with algorithms that stay largely domain-agnostic, so you can adapt them across use cases while keeping development simple and recommendations accurate.

---

## AMPER Framework

<img src='./AMPER.PNG' alt="AMPER framework overview" width="700"/>

### How it works

The pipeline is built around five steps, driven by user-centric objectives and data:

| Step | Name | Role |
|------|------|------|
| **A** | **Aim** | Define the user’s goal and what “better” means. |
| **M** | **Measure** | Use available data to quantify the user’s current state. |
| **P** | **Predict** | Model future states from the measured data. |
| **E** | **Evaluate** | Assess predicted states against the aim. |
| **R** | **Recommend** | Suggest actions that best move the user toward the target state. |

Data flows **A → M → P → E → R**: from defining the aim and measuring the present, through prediction and evaluation, to actionable recommendations.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/youngchan-k/AMPER.git
cd AMPER
pip install -r requirements.txt
```

**Requirements:** Python 3.x (see `requirements.txt` for TensorFlow, pandas, etc.).

### 2. Get the data

- Download the [EdNet KT1 dataset](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view).
- Place it under **`data/EdNet-KT1/KT1`**.
- Ensure **`csv/questions.csv`** exists (see [Project structure](#project-structure) below).

### 3. Run the pipeline

```bash
python main.py
```

This runs **preprocess → train → evaluate** in one go. Checkpoints and outputs go to `logs/` and `csv/` as configured in `amper/config.py`.

---

## Project structure

```
AMPER/
├── main.py                 # Entry point: preprocess → train → evaluate
├── amper/
│   ├── config.py           # Paths and constants (CSV, data dirs, logs)
│   ├── data/
│   │   └── preprocess.py   # EdNet-KT1 preprocessing, split_data, main_data_preprocess
│   ├── model/
│   │   └── transformer.py  # Transformer, encoder/decoder, attention, masks
│   ├── training/
│   │   └── train.py       # dataframe_to_list, tokenizer, train_main
│   ├── evaluation/
│   │   └── eval.py        # evaluate, predict, eval_main
│   └── recommendation/
│       └── recommend.py   # user_question_matrix, recommend_question
├── csv/                    # Input/output CSVs (questions.csv required)
├── data/                   # Raw EdNet-KT1 data → place in data/EdNet-KT1/KT1
└── logs/                   # Checkpoints
```

---

## References

- **Towards Minimally Domain-Dependent and Privacy-Preserving Architecture and Algorithms for Digital Me Services: EdNet and MIMIC-III Experiments** — *HICSS 2025*
- **AMPER (Aim-Measure-Predict-Evaluate-Recommend): The Paradigm of Digital Me** — *ICEC 2022*
