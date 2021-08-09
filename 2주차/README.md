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