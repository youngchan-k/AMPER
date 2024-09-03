# Towards Minimally Domain-Dependent and Privacy-Preserving Architecture and Algorithms for Digital Me
Digital Me algorithm can improve a user's situation by suggesting optimal actions to reach a desired state. Our goal is to develop minimally domain-dependent algorithms within the 'Digital Me' framework. This approach simplifies the development process and enhances the predictive accuracy and personalized recommendations.

## AMPER(Aim-Measure-Predict-Evaluate-Recommend)
<img src='./assets/AMPER.PNG'/>

### Framework
AMPER framework is a general Digital Me algorithm to recommend activities that optimize the achievement of users' goals. Digital Me is an AI-driven service that allows real-time management to reflect the user's goals, measures, predicts, and evaluates the individual's status, and recommends actions to improve the status.  

### Method
The Digital Me algorithm utilizes user state data, according to user-centric objective A (Aim), to establish M (Measure) to measure the current state of the user. It only uses data to predict the user's future states through P (Predict). After evaluating the user's future states through E (Evaluate), it is possible to maximize the user's state improvement by providing an R (Recommendation) of behavior for achieving the target state.    

### Dataset & Installation
This algorithm operates on a simple principle applicable to general Digital Me services. We demonstrate the effectiveness of our proposed algorithm structure in enhancing a user's English score. To begin, download the [EdNet KT1 dataset](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view) and place the dataset in *'/data/EdNet-KT1/KT1'*. Then, clone the github repository and run the code as the following instructions.

```
git clone https://github.com/youngchan-k/AMPER.git
python main.py
```