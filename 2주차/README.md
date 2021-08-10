# 8 / 9 (월)

### 1. 강의 복습
* Introduction of Deep Learning
    - 딥러닝의 핵심 : 데이터, 모델, loss function, algorithm
    - 데이터는 풀고자 하는 문제와 관련있다.
    - 모델은 데이터를 가지고 원하는 답을 내는 것이다.
    - loss function은 정답과의 근사치로, 모델이 어떻게 학습할지에 관한 것이다.
    - algorithm은 이 loss function을 어떻게 줄여나갈지에 관한 것이다.
    - 최근에 각광받는 모델들은 AlexNet, DQN, Encoder/Decoder, Adam Optimizer, GAN, ResNet, Transformer, Bert, GPT-X, Self-Supervised Learning 등이 있다.
    
* Neural Networks, Multi-Layer Perceptron
    - Neural Network는 비선형함수와 matrix multiplication의 연속으로 이루어진 함수의 근사이다.
    - 비선형함수 activation function의 존재로 여러층 쌓는 것이 가능하다.
    - 각각의 목적에따라 loss function의 식이 달라진다.(선형회귀 - MSE, 분류문제 - CE, 확률문제 - MLE)
    
* Data Visualization
    - 데이터 시각화란 데이터를 그래픽요소로 매핑하여 시각적으로 표현하는 것이다.
    - 데이터의 전체적인 분포, 또는 개별 데이터를 시각화할 수 있다.
    - 데이터의 종류에는 정형데이터(csv), 시계열 데이터(기온, 주가, 음성, 비디오 등), 지리/지도 데이터, 관계데이터(그래프), 계층적데이터로 나눌 수 있다.
    - 또, 데이터를 수치형과 범주형으로 나눌 수 있다.
        - 수치형(numerical) :
            - 연속형 (continuous) : 길이, 무게, 온도
            - 이산형 (discrete) : 주사위 눈금, 사람 수
        - 범주형 (categorical) :
            - 명목형 (nominal) : 혈액형, 종교
            - 순서형 (ordinal) : 학년, 별점, 등급
    - 시각화에는 마크(mark)와 채널(channel)로 이루어져 있다.
        - 마크는 점, 선, 면으로 이루어진 데이터 시각화이다.
        - 채널은 각 마크를 변경할 수 있는 요소이다.
            - 위치, 모양, 길이, 면적, 부피, 기울기, 색 등이 있다.
    - 전주의적 속성을 적절히 이용하여 시각적 분리를 일으킬 수 있다. 
---

### 2. 과제 수행 과정 / 결과물 정리
Vision Transformer를 구현하는 과제이다. Transformer를 아직 배우지않아 구글링을 하여 진행해나갔다.


참고링크 : https://github.com/FrancescoSaverioZuppichini/ViT

https://yhkim4504.tistory.com/5


Mnist 데이터 학습에 MLP대신 Transformer를 이용하여 구현하는 과제로 ViT는 크게 세 부분으로 나누어진다.
* Patch Embedding
    - MNIST data 28 * 28을 4 * 4 patch로  나누었다.
    - 나눈 patch를 flatten하여 linear를 통과, class_token을 concat한다.
    - 그리고 나뉜 patch의 위치를 알려주는 positions를 더한다.
* Encoder
    - 이 부분은 잘 이해하지 못했다. 
* Classification head
    - 이 부분 역시 잘 이해하지 못했다.

---
### 3. 피어세션 정리

피어세션엔 과제를 아직 수행하지 못해 피어분들과 선택과제에 대한 이야기를 나누지 못했다.

지난주 피어세션 마지막에 배운 SVD에 대해 어떤 기준으로 압축할 수 있는지 이야기했다.
- Sigma-Singular value가 가장 높은 것부터 추출. 즉, 가장 유의미한 정보순으로 나열 후 뒤부터 탈락시키며 압축시킨다.

또, 멘토링 시간을 목요일에서 수요일로 옮겼다.

---

### 4. 학습 회고

선택과제의 난이도 급증하여 수행하는데 어려움을 느꼈다.

하지만 본격적으로 딥러닝에 입문하는 단계이므로 잘 따라가야겠다고 생각했다.

# 8 / 10 (화)

### 1. 강의 복습
* Optimization
    - **Generalization**
        - 새로운 데이터에 대해 모델이 얼마나 잘 예측하는지를 말한다.
        - Training error는 작아지더라도 어느 시점 후에는 test error는 커지는 현상이 일어난다.
        - 이 둘의 차이를 generalization gap이라고 한다.
    - **Underfitting, Overfitting**
    - **Cross-Validation (K-fold validation)**
        - Train data와 valid data를 나누는 한 방법.
        - 학습에 참여하는 data는 train과 valid만 가능하지, test data는 학습에 참여해선 안된다.
    - **Bias and Variance tradeoff**
        - Cost를 최소화 한다는 것은 bias와 variance, noise를 최소화 한다는 의미이며, 서로가 trade-off 관계이다.
    - **Bootstrapping**
        - 전체 학습 데이터중 중복을 허용하여 random sampling하여 학습시키는 것을 말한다.
        - 여러 모델을 만들어 예측에 활용한다.
    - **Bagging, Boosting(Ensemble)**
        - Bagging (Bootstrap Aggregation)은 Bootstrapping의 결과로 생긴 여러 모델을 다수결, 평균등으로 답을 내는 것을 말한다.
        - Boosting은 weak learner를 여러개 만들어 순차적으로 학습시킨다. 각각의 모델은 전 모델의 틀린점에 집중하여 학습하게 된다.
        - Bagging은 독립적으로 돌아가지만, Boosting은 순차적으로 돌아간다.
    
* Gradient Descent Method
    - Stochastic gradient method - 업데이트 한번에 샘플 1개만 이용
    - Mini-batch gradient method - 업데이트 한번에 샘플 batch-size만큼 이용
    - Batch gradient method - 업데이트 한번에 전체 학습데이터를 이용
    
* Batch-size
    - 일반적으로 작은 batch size (128, 256 등)가 더 낫다고 한다.
    - 큰 batch size는 sharp minimum에 도달한다.
    - 작은 batch size는 Flat minimumr에 도달하기 때문에 testing funtcion과 별로 차이가 나지 않기 때문이다.
    
* Gradient Descent Method
    - **Stochastic gradient descent**
        - gradient에 learning rate만큼 곱해서 빼주어 W를 계산해준다.
        - learning rate를 잡는 것이 어려운 단점이 있다.
    - **Momentum**
        - 관성이 있는 것처럼 작동하게 해준다.
        - 계산한 gradient에 이전 step에서의 accumulated gradient의 momentum을 적용해 accumulated gradient를 계산한다.
        - 한번 흘러간 gradient 방향을 어느정도 유지시켜주기 때문에 gradient가 많이 바뀌더라도 학습이 잘되는  장점이 있다.
    - **Nesterov accelerated gradient (NAG)**
        - Momentum과 비슷하나 계산한 gradient 대신에 lookahead gradient(먼저 이동해보고 나서의 gradient)를 이용한다.
        - Momentum보다 수렴이 빠르게 되는 장점이 있다.
    - **Adagrad**
        - Adaptive learning rate로 학습률이 계속해서 변화한다.
        - G(Sum of gradient squares)를 분모에 넣음으로써 지금까지 많이 변한 parameter는 작게 변하고, 작게 변한 parameter는 크게 변하게 만들어 준다.
        - 하지만 학습이 길어질수록 G가 커지기 때문에 학습이 거의 되지 않는다.
    - **Adadelta**
        - Adagrad의 문제를 해결하기 위해 window-size를 정한다.
        - learning rate이 없기 때문에 변경할 요소가 거의 없어서 잘 활용되지 않는다.
    - **RMSprop**
        - Adadelta와 비슷하지만 stepsize를 넣어 변경할 요소를 만들었다.
    - **Adam**
        - Momentum과 RMS를 합친 것과 같다.
        - epsilon은 0으로 나누어 지는것을 막는 용도이지만, 이 값이 실제적으로 중요하다.
    
* Regularization 
    - 학습에 반대되는 규제로, 학습이 잘 안되더라도 test data에도 잘 적용되어 generalization gap을 줄이는 방법이다.
    - **Early Stopping**
      - validation data의 정확도가 떨어지는 시점에서 학습을 멈춘다.
    - **Parameter norm penalty(Weight Decay)**
      - loss function에 parameter의 크기를 추가해 parameter의 값이 작은쪽으로 학습되게 한다.
      - function space에 smoothness를 더해 generalization이 잘된다고 한다.
    - **Data augmentation**
      - 데이터가 한정적이기 때문에 데이터를 회전, 변환해 수를 늘리는 기법이다.
    - **Noise robustness**
      - 입력 데이터와 weight에 noise를 추가하는 기법이다.
    - **Label Smoothing**
      - 학습 데이터끼리 서로 섞어주어 label을 섞어준다.
      - decision boundary를 부드럽게 해주는 기법이다.
      - Mixup으로 이미지 분류의 성능이 향상될 수 있다.
    - **Dropout**
      - 확률적으로 일부 neuron의 값을 0으로 하여 forward시킨다.
      - neuron이 robust feature를 잡을 수 있다고 한다.
    - **Batch normalization**
      - 각각의 layer를 정규화시키는 기법이다.

* Bar Plot
    - 직사각형 막대를 이용하여 데이터의 값을 표현하는 그래프이다.
    - 막대그래프, Bar chart, Bar graph라고도 한다.
    - 범주에 따른 수치 값을 비교하기 좋다.
    - Bar plot에서는 한개의 feature만 보여주기 때문에 여러 group을 보여주기 위해 다양한 방법이 쓰인다.
        - **Multiple bar plot**
          - plot을 여러개 그려서 보여주는 방법
        - **Stacked bar plot**
          - plot을 쌓여서 보여주는 방법
          - 맨 밑의 bar의 분포는 보기 쉽지만, 그 외의 분포는 파악하기 어렵기 때문에 주석등을 이용한다.
          - positive, negative로 표현할 수도 있다.
          - 전체에서 비율을 나타내는 percentage stacked bar chart도 존재한다.
        - **Overlapped bar plot**
          - plot을 겹쳐서 보여주는 방법
          - 2개 그룹만 비교할때 효과적이고, 같은 축을 사용하기 때문에 비교하기도 쉽다.
        - **Grouped bar plot**
          - bar를 이웃되게 배치하는 방법
          - 분포와 가독성이 좋다.
          - 그룹이 적을때 효과적이다.
    - 정확한 bar plot을 그리기 위해서는 몇 가지 방법을 이용한다.
        - Principle of Propotion Ink
            - 실제 값과 표현되는 그래픽의 잉크 사용량은 비례해야 한다.
        - 데이터 정렬하기
            - 데이터의 종류에 따라 정렬하는 기준이 다르다.
        - 적절한 공간 활용
            - 여백과 공간을 활용해 가독성을 높힐수 있다.
        - 필요없는 복잡함은 지양한다.

---

### 2. 과제 수행 과정 / 결과물 정리
필수과제 Optimization은 실습과 함께 진행하였고, Adam Optimizer가 Momentum보다 성능이 좋았다.

또, Momentum은 SGD보다 성능이 좋다는 것을 확인했다.

선택과제 Adversarial AutoEncoder를 구현하면서 모델의 구조를 아직 배우지 않아 잘 파악하지 못했다.

하지만 코드를 구현하는 것은 크게 어렵지 않게 할 수 있었다.

또, 학습하는데 개인 노트북에서는 시간이 오래걸려 colab을 처음 이용했다.

처음 이용해서 파일 저장하는 부분에서 많이 헷갈리고 오래걸렸다.

---

### 3. 피어세션 정리

어제에 이어서 선택과제에 대한 많은 질문을 하였다.

각자 나름대로의 학습방법으로 선택과제를 수행한 것 같았다.

구글링을 통해 과제를 구현하다 보니 ViT 모델에 대해 완벽하게 학습했다고 보긴 힘들었지만, 최대한 답변할 수 있는 부분은 서로 답해주었다.

하지만 나머지 질문들은 멘토님께 질문드리기로 했다.

---

### 4. 학습 회고

앞으로 도메인에 대해 특강을 들었는데, 결국 CV와 NLP 모두 독립적인 분야가 아니라 서로 Deep Learning이라는 공통된 분야라는 것을 깨달았다.

또, 데이터 시각화라는 분야에 대해 생소했는데 어떤 분야인지 알게 되었고, 사람에 중점이 되어있는 분야라는 것을 알게 되었다.

아직 학습하지 못한 Transformer, Encoder 같은 모델에 대해서는 학습한 이후에 선택과제를 다시 보아야겠다고 생각했다. 