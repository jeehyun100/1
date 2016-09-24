# Deep Reinforcement Learning

## Reinforcement learning?
강화 학습(Reinforcement learning)은 기계 학습이 다루는 문제 중에서 다음과 같이 기술 되는 것을 다룬다. 어떤 환경을 탐색하는 에이전트가 현재의 상태를 인식하여 어떤 행동을 취한다. 그러면 그 에이전트는 환경으로부터 포상을 얻게 된다. 포상은 양수와 음수 둘 다 가능하다. 강화 학습의 알고리즘은 그 에이전트가 앞으로 누적될 포상을 최대화 하는 일련의 행동으로 정의되는 정책을 찾는 방법이다.

 - Deep Learning Neural Network는 사람의 뇌구조를 모방
 - 강화학습은 사람의 행동방식을 모방. 사람이 평소에 어떤 것을 배워나가고 일상속에서 행동하는 방법과 상당히 유사.

### RL 용어
 - State: 게임에서의 각 물체들의 위치, 속도, 벽과의 거리 등을 의미한다.
 - Action: 게임을 플레이하는 주인공의 행동을 의미한다. 4가지 방향에 대한 움직임
 - Reward: 게임을 플레이하면서 받는 score나 시간 등 보상
 - Value: 해당 action 또는 state가 미래에 얼마나 큰 reward를 가져올 지에 대한 기대값.
 - Policy: 주인공이 현재의 게임 State에서 Reward를 최대한 얻기 위해 Action을 선택하는 전략. 게임을 플레이하는 방법]을 의미
 - Agent : 사람의 뇌에 해당. 
 - Enviroment : 보상과 상태를 전달

### RL 예제
 [벽돌깨기](https://www.youtube.com/watch?v=iqXKQf2BOSE&feature=youtu.be)

사람의 뇌에 해당하는 이 agent는 처음에 랜덤하게 움직이게 됩니다. 랜덤하게 움직이다가 우연히 공을 치게 되고 게임점수가 올라가게 되면 이 agent는 "아! 이 행동을 하면 점수가 올라가는구나!!"라는 식으로 판단을 해서 점수를 올라가게 했던 행동을 더 하려고 자신을 학습시킵니다. 어느 순간 터널이 뚤려서 한 번에 엄청난 점수를 얻는 사건이 발생합니다. 이제 agent는 알게 되는 것입니다. "터널을 뚫어서 점수를 얻는 것이 더 많은 점수를 얻을 수 있다"라는 것을 말입니다. 강화학습을 알기위한 기본 이론을 용어정리 위주로 설명


# Markov Decision Process
## Markov는 1800년대의 러시아 수학자의 이름입니다. 이 분의 이름 이 하나의 형용사가 되었는데 그 의미는?

처음 어떠한 상태로부터 시작해서 현재 상태까지 올 확률이 바로 전 상태에서 현재 상태까지 올 확률과 같을 때, state는 Markov하다고 일컬어질수 있습니다.
스타크래프트같은 게임이라고 생각하면 게임 중간 어떤 상황은 이전의 모든 상황들에 영향을 받아서 지금의 상황이 된 것이기 때문에 사실은 지금 상황에 이전 상황에 대한 정보들이 모두 담겨있는것입니다. 우리가 접근하는 모든 문제의 state가 Markov property라고 말할 수는 없지만 그럴 경우에도 state는 Markov라고 가정하고 강화학습으로 접근합니다.

## MDP 구성
 - State : 상태, 벽돌깨기의 픽셀값
 - Action : Agent의 행동
 - Transition probability : S에서 A라는 행동을 취할때 S'에 도착하는것
 - Reward : Action을 취할때 Reward값
 - Discount : Reward를 다 더하면 무한대로 가기 때문에 좋은 보상만 선택하는 방법
 - Policy : 어떤 행동을 할지 결정하는 방법

##  Value Function
### State Value Function
상태 S의 가치를 나타내는 펑션. agent가 state 1에 있다고 가정할때 한 Episode가 끝날때까지 action에 따른 reward 값의 합.(Policy는 어떤 행동을 결정하는 방향)
### Action State Value Function
state s에서 action a를 취할 경우의 받을 return에 대한 기대값으로서 어떤 행동을 했을 때 얼
마나 좋을 것인가에 대한 값입니다. Action-value function은 다른 말로 Q-value로서 q-learning이나 deep q-network같은 곳에 사용되는 q라는 것이 이것을 의미

# Bellman Expectation Equation
현재 State와 다음 State 관계를 식으로 풀어낸것

## Bellman Expectation Equation
![Bellman Expectation Equation](http://images.slideplayer.com/26/8652641/slides/slide_14.jpg)
실제 강화학습으로 무엇인가를 학습시킬 때 reward와 state transition probability는 미리 알 수가 없습니다. 경험을 통해서 알아가는 것 입니다. 이러한 정보를 다 알면 MDP를 모두 안다고 표현하며 이러한 정보들이 MDP의 model이 됩니다. 강화학습의 큰 특징은 바로 MDP의 model를 몰라도 학습할 수 있다는 것 입니다. 따라서 reward function과 state transition probability를 모르고 학습 하는 강화학습에서는 Bellman equation으로는 구할 수가 없습니다.


## Bellman Expectation Equation for Q Function
![Bellman Expectation Equation for Q Function](http://images.slideplayer.com/17/5333901/slides/slide_20.jpg)

현재 environment에서 취할 수 있는 가장 높은 값의 reward 총합입니다. 위의 두 식 중에서 두
번째 식, 즉 optimal action-value function의 값을 안다면 단순히 q값이 높은 action을 선택해주면 되므로 이 최적화 문제는 풀렸다라고 볼 수 있습니다.

# Model Free

실제를 MDP에 대한 모델을 모르는 경우가 대부분임 Model Free한 방법이 필요.
실제사용한 경험한 정보를 활용하여 update를 함으로써 Environment의 모델을 몰라도 학습가능 방법

현재의 Policy를 바탕으로 움직여 보면서 Sampling을 통해  Value Function을 update 하는것을 Model Free Prediction, Policy까지 Update하면 Model Free Control이라 함
Sampling 해서 학습하는 방법은 크게 2가지 정보가 있음
 - Monte-Carlo : 무엇인가 랜덤하게 측정하는 방법 일단 Episode끝까지 간다음에 각 상태에 대한 Value를 역으로 계산, Policy업데이트는 Monte-Carlo-Control이라 불림
 - Temporal Difference : Monte Carlo는 Episode가 끝나야지 사용가능 그러나 TD는 스타크래프처럼 끝나지 않는 Episode가 있는 상황에서 사용가능

# Q-Learning
MC와 TD중에 가장 좋은 알고리즘이 Q-Learning임


- ## Q Learning

![QLearning 식](http://www.randomant.net/wp-content/uploads/2016/05/q_learning3.jpg)
- ## Q Learning Algorithm
![QLearning 알고리즘](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/pseudotmp9.png)

![Q function Table](http://mnemstudio.org/ai/path/images/r_matrix1.gif)

여기서 Deep Learning 이 등장함. Deep Learning은 Non Linear한 함수를 잘 추정함.
상태 값을 넣으면 Value값을 추출하는 함수를 Deep Learning을 통하여 학습하여 만들어냄


- ## Deep Q Leaning

### 제가 만든 벽돌깨기 예제
[벽돌깨기 예제](https://gym.openai.com/evaluations/eval_GmcqEZGjSFaXxgYMy0gg)