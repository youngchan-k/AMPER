# AMPER (Aim-Measure-Predict-Evaluate-Recommend): The Paradigm of Digital Me

AMPER is a general Digital Me algorithm to manage the individual's state in real-time. We verify our proposed algorithm structure for the user-centric aim of improving the user's English score. Please refer to the [paper](https://youngchan-k.github.io/assets/paper/ICEC2022/AMPER.pdf) and [slide](https://youngchan-k.github.io/assets/paper/ICEC2022/AMPER_slide.pdf) if you want to see more details


## Method
<img src='AMPER.PNG'/>

The Digital Me algorithm uses user state data, according to user-centric objective A (Aim), to establish M (Measure) to measure the current state of the user. It only uses data to predict the user's future states through P (Predict). After evaluating the user's future states through E (Evaluate), it is possible to maximize the user's
state improvement by providing an R (Recommendation) of behavior for achieving the target state.

## Dataset & Installation
Download the [EdNet KT1 dataset](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view) and place the dataset in *'/data/EdNet-KT1/KT1'*
You can download and run the code as follows.

```
git clone https://github.com/youngchan-k/AMPER.git
python main.py
```



