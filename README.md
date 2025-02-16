# Minimally Domain-Dependent and Privacy-Preserving Digital Me Algorithms
Digital Me is an AI-driven service that allows real-time management to reflect the user's goals, measures, predicts, and evaluates the individual's status, and recommends actions to improve the status. We develop minimally domain-dependent algorithms to simplify the development process and enhance the predictive accuracy and personalized recommendations. 


For more information, please refer to the papers listed below.

**1) Towards Minimally Domain-Dependent and Privacy-Preserving Architecture and Algorithms for Digital Me Services: EdNet and MIMIC-III Experiments** *(HICSS 2025)*
[[Paper](./assets/pdf/DigitalMe_F.pdf) | [Slide](./assets/pdf/DigitalMe_F_slide.pdf)]

**2) AMPER(Aim-Measure-Predict-Evaluate-Recommend): The Paradigm of Digital Me** *(ICEC 2022)*
[[Paper](./assets/pdf/AMPER.pdf) | [Slide](./assets/pdf/AMPER_slide.pdf)]


## AMPER Framework
<img src='./assets/AMPER.PNG'/>

### Procedure Overview
The Digital Me algorithm utilizes user state data, according to user-centric objective **A (Aim)**, to establish **M (Measure)** to measure the current state of the user. It only uses data to predict the user's future states through **P (Predict)**. After evaluating the user's future states through **E (Evaluate)**, it is possible to maximize the user's state improvement by providing an **R (Recommendation)** of behavior for achieving the target state.

### Experiment with EdNet data
We demonstrate the effectiveness of our proposed algorithm structure in enhancing a user's English score. To begin, download the [EdNet KT1 dataset](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view) and place the dataset in *'/data/EdNet-KT1/KT1'*. Then, clone the github repository and run the code as the following instructions.

```
git clone https://github.com/youngchan-k/AMPER.git
python main.py
```