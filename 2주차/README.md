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


# 8 / 11 (수)

### 1. 강의 복습
* Convolution
    - filter의 종류에 따라 이미지에 blur, emboss, outline등의 효과를 얻을 수 있다.
    - 예를 들어 32x32x3 이미지에 5x5x3 filter를 4개 사용하여 convolution하게 되면 결과는 28x28x1이 4개, 즉 28x28x4가 생긴다.
    - Convolution Neural Network는 convolution layer, pooling layer, fully connected layer로 이루어져있다.
    - 이 중 convolution과 pooling layer는 feature extraction을 수행하고 fully connected layer는 decision making을 한다.
    - CNN에서 parameter의 수가 많을수록 학습이 어렵고 generalization performance가 떨어진다.
    - 따라서 parameter의 수가 적게 설계하는 것이 중요하다.
    - Stride - filter를 얼마나 움직일지에 대한 것이다.
    - Padding - Boundary에서 값을 덧붙이는 것을 말한다. (Zero padding)
    - 예를 들어 40x50x128 -> 40x50x64로 filter가 3x3일때, parameter를 구한다면,
        - filter의 channel은 input feature map의 channel과 같다.(3x3x128)
        - 이러한 channel은 64개 있다고 할 수 있다.
        - 따라서 parameter의 수는 3 x 3 x 128 x 64 = 73728이다.
    - AlexNet의 parameter수의 대부분은 마지막 fully connected layer가 지배적이다.
        - 따라서 이 숫자를 줄이기 위해 CNN을 깊게 쌓고, 마지막을 작게 쌓는다.
    - 1x1 Convolution으로 dimension(channel)을 줄여 parameter수를 줄일 수 있다.(e.g. bottleneck architecture)

* Modern CNN - ImageNet Large-Scale Visual Recognition Challenge(ILSVRC)에서 수상을 한 5개의 Network
    - **AlexNet**
        - 5개의 CNN과 3개의 dense layer로 구성된다.
        - ReLU를 사용하여 gradient vanishing을 없앴다.
        - Data Augmentation과 Dropout을 이용했다.
        - 지금은 당연한 기법들이 당시에는 그렇지 않았다.
    - **VGGNet**
        - 3x3 Convolution filter를 사용했다.
        - 3x3을 2번 사용하는 것이 5x5 1번 사용하는 것보다 더 효율적이다. (같은 receptive field를 가지면서도 paremeter 수가 더 적다.)
    - **GoogleNet**
        - Inception Block을 활용해 parameter수를 줄였다. (하나의 input을 여러 path를 거쳐 마지막에 concat함)
        - 1x1 Convolution으로 dimension 수를 줄였다.
        - 예를 들어 128 channel feature map에 3x3 convolution filter를 적용해 128 channel output을 만든다고 하면,
            - parameter의 수는 3 x 3 x 128 x 128 = 147456이다.
            - 하지만 1x1 convolution을 이용해 중간에 channel을 32로 줄이게 된다면 1 x 1 x 128 x 32 + 3 x 3 x 32 x 128 = 40960이다.
    - **ResNet**
        - Neural Network가 깊어질수록 학습에 어려움이 있다.
        - 하지만 input을 다시한번 더해주는 residual을 추가하여 이를 해결했다. (skip connection)
        - 1x1 Convolution을 활용해 dimension을 줄이고 output의 channel도 조정했다. (bottleneck architecture)
    - **DenseNet**
        - residual을 하지않고 concat으로 이어 붙였다.
        - 늘어난 channel을 1x1 Convolution으로 다시 줄였다를 반복하는 Network.
            
* Semantic Segmentation
    - Semantic Segmentation은 각 pixel이 어떤 object에 속하는지 분류하는 문제로, 자율주행 등에 이용된다.
    - Fully Convolution Network는 feature map을 flatten하여 fully connected layer를 통과시키는 것과 같은 연산이다. (Convolutionalization)
    - 이와 같은 network는 input image의 크기에 상관없다는 장점이 있다. (image가 커지면 마지막에 heat map을 생성)
    - heat map을 가지고 segmentation을 수행할 수 있다.
    - output을 원래의 dense pixel로 upsampling 한다(Deconvolution)
    - Deconvolution은 convolution의 완벽한 역연산은 아니지만, parameter의 수를 계산할 때 역연산이라고 생각하면 좋다.
    
* Object detection
    - R-CNN은 region으로 나누어 각각을 AlexNet을 통과시키는 방법이다.
    - SPPNet은 bounding box를 찾아 CNN을 한번만 통과시킨다.
    - Region Proposal Network는 anchor box를 만들어 물체가 있을 만한 곳을 찾는다.
    - YOLO(You Only Look Once) - 한번에 multiple bounding box를 이용한다.


---

### 2. 과제 수행 과정 / 결과물 정리
CNN은 실습을 따라가면서 수행하였고, 크게 어렵지 않았다. MLP보다 성능이 뛰어난 것을 확인했다.

Mixture Density Network 과제는 양이 많았지만 구글링하며 나름대로 한줄 한줄 실행하며 이해해보았다.

먼저, 기존의 function approximation은 x가 하나 주어지면, y가 하나로 나오는 함수였다.

하지만, 우리가 원하는 함수가 역함수로 되어서 만약, x가 하나 주어졌을때 y가 여러개 나오는 함수라면 어떻게 해야하는가가 문제였다.

그래서 반대로 생각해 y가 하나 주어지면 이를 gaussian function들의 합으로 생각해 계산한다는 것이다.

각각의 gaussian function의 parameter인 가중치 W, 평균 m, 표준편차 s를 학습시키는 것이다.

loss function의 경우, 주어진 x에 대해 y가 나올 확률로 정의했기때문에 cross-entropy식을 이용했다.

---

### 3. 피어세션 정리
오늘은 모더레이터로서 피어세션을 진행해나갔다.

CNN 강의를 관련해 질문이 있어서 정리를 해보았다.

* Dense layer의 가중치 또한 학습이 진행되는가?
    - 에러에 따라 모든 (Conv, FCN)의 가중치들의 학습이 진행된다.
* Convolution 연산에서 가중치의 차원은 (kernel_W, kernel_H, in_channel, out_channel)과 같다. 그렇다면 in_channel 기준으로 같은 값으로 연산이 진행되는지(Broadcasting) 또는 각 in_channel마다 다른 값을 가지는지 궁금하다.
    - 강의에서 학습할 가중치의 개수가 “kernel_W * kernel_H * in_channel * out_channel” 이라고 설명한 것으로 미루어 보아, 각각 다른 in_channel 가중치는 각각 다른 값을 가질 것이라고 생각된다.
    - e.g. in_channel 기준 각각 다른 가중치의 값은 다음과 같다. (parameter[0,0,:,0])

그리고 내가 이해한대로 다른 캠퍼분들께 선택과제 MDN을 설명했다.

---

### 4. 학습 회고

모더레이터로써 선택과제에 대해 내 나름대로 설명했는데, 다른분들이 많이 이해하신것 같았다.

그리고 아직 배우지않은 모델에 대해서는 추가적으로 학습이 필요하다고 생각했다.

# 8 / 12 (목)

### 1. 강의 복습

* Recurrent Neural Network
    - Sequential Data를 다루기 위해 사용되는 모델이다. (입력의 차원을 알 수 없다.)
    - Naive model은 과거의 모든정보를 고려해 앞으로 나올 말을 예측한다.
    - Autoregressive model은 과거의 몇 개의 정보만 고려한다고 정한다.
    - 이 중 Markov model(first order autoregressive model)은 한 step만 고려한다.
        - joint distribution을 표현하기 쉽지만 현실적이지 않다.
    - latent autoregressive model은 hidden(latent) state를 만들어 과거의 모든정보를 요약한다고 생각한다.
    - RNN은 short-term dependency 문제를 가지고 있다. (가까운 정보만 고려되고 한참 멀리있는 정보가 잘 고려되지 않는 것)
    - RNN에는 exploding/vanishing gradient 문제가 있다.
    - 이를 보완하는 Long Short Term Memory(LSTM) 모델이 있다.
    
* LSTM
    - input 단어 x_t
    - output hidden state h_t
    - 이전 cell로 부터 받는 previous cell state, previous hidden state h_t-1
    - 이후 cell로 보내주는 next cell state, next hidden state h_t가 있다.
    - 각각의 cell에는 forget gate, input gate, output gate가 있다.
    - forget gate는 현재입력 x_t와 이전의 output h_t-1을 가지고 previous cell state를 얼마나 기억할 지를 정해준다.
    - input gate는 현재입력 x_t와 이전의 output h_t-1을 가지고 cell state에 얼마나 반영할지를 정해준다.
    - 그 후 update cell에서 이 두 gate의 결과를 가지고 cell state를 update해 next cell state로 나간다.
    - output gate는 현재입력 x_t와 이전의 output h_t-1을 가지고 cell state와 조합해 next hidden state로 나간다.
    
* Gated Recurrent Unit
    - LSTM보다 간단한 구조를 갖고 있다.
    - reset gate, update gate를 갖고 있고, cell state가 없고 hidden state만 있다.
    - 하지만 Transformer의 등장으로 LSTM, GRU를 대체하고 있다.
    
* Transformer
    - Sequential Data를 다루기 위한 모델로, sequential data가 trimmed, omitted, permuted된 것을 해결하기 위해 제안되었다.
    - 재귀적인 구조대신 **Attention**을 활용한다.
    - 어떤 문장이 입력으로 들어오게 되면 RNN은 이를 재귀적으로 수행했다.
    - 하지만 Transformer는 문장을 한번에 encoding하는 것이 가능하다.
    - Transformer는 같은 단어에 대해서도 주변 단어와의 관계를 통해 output이 달라질 여지가 있으므로 다른 모델에 비해 flexible하다는 점이 있다.
    - 하지만 n개의 단어를 동시에 처리하려면 만들어야하는 attention map이 n^2이므로 메모리를 많이 사용한다.
    - Transformer는 encoding부분, encoder와 decoder 사이의 부분, decoding부분으로 나눌 수 있다.
- Encoding
    - 각각의 encoder는 stacked된 구조를 가지며, encoder는 **Self-Attention**과 **Feed Forward Neural Network**로 이루어져있다.
    - Self-Attention은 한 단어가 나머지 단어들에 영향을 받는 부분이고(dependent), Feed Forward는 그렇지 않다(independent).
    - positional encoding을 통해 단어의 위치가 다른 문장에 대해서도 다르게 작동한다.
    - Self-Attention은 positional encoding된 단어에 대해 **Query, Key, Value** 벡터를 만들어낸다.
    - Encoding하고 싶은 단어의 Query와 자신을 포함한 나머지 단어의 Key를 내적해 **Score**를 계산한다.
    - Score를 key의 dimension을 가지고 normalize하고 softmax를 취해 value와 weighted sum을 통해 encoding할 수 있다.
    - query와 key의 dimension은 같아야하고, value의 dimension과는 다를 수 있다.
    - 이와 같은 연산을 행렬을 이용하면 쉽게 구할 수 있다.
- Multi-Head Attention(MHA)
    - Attention을 여러번 수행해 여러 encoding 결과를 만든다.
    - 여러 encoding 결과를 concat한 후 다시 input의 dimension과 맞추기 위해(같은 encoder 구조를 사용하기 위해) linear map 행렬과 곱한다.

* Encoder와 Decoder 사이
    - input단어의 Key와 Value를 보낸다.
    - decoder 단어의 query를 가지고 최종 출력을 만든다.
    
* Decoder
    - 학습할 때 masking을 통해 미래의 정보를 볼 수 없게 한다.
    - Output은 Autoregressive로 단어의 분포를 가지고 sampling해 나온다.
    
- Vision Transformer는 이미지분류에 Transformer Encoder를 활용했다.
- DALL-E는 문장을 가지고 새로운 이미지를 만든다.

---

### 2. 과제 수행 과정 / 결과물 정리
LSTM으로 MNIST data에 대해서 좋은 결과가 나온다는 것을 확인했다.

MHA에서는 Query, Key, Value shape이 많이 헷갈렸다. 그리고 transformer로 뭔가 학습한 것이 아니라 모델만 만든것이라 학습이 잘되는지 확인하지 못했다.

그리고 구조가 실습을 진행하면서 잘 기억나지 않아 수업자료를 다시 참고하며 진행했다.

---

### 3. 피어세션 정리

오늘 배운 내용들이 많이 중요하면서 어려웠던 내용이라 관련해 질문이 있었다.

LSTM의 각각의 gate에 대해 다시한번 짚고 넘어갔다.

그리고 MHA에서 Q와 K, V의 개수가 다르더라도 된다고 했는데, 이 부분에 대해서는 나도 이해하지 못했다.

선택과제 해설 전에 피어분들과 솔루션을 보고 나름대로 답을 맞춰보았다.


---

### 4. 학습 회고

오늘배운 Transformer는 결국 NLP, CV 모두 중요한 모델이고, 상당히 좋은 성능을 내고 있어 복습이 꼭 필요하다고 생각했다.

그리고 멘토님과의 오피스아워에서 들었던것처럼 선택과제에 대해서도 너무 조급해 하지 않아야겠다고 생각했다.

# 8 / 13 (금)

### 1. 강의 복습

* Generative Model
    - 단순히 이미지를 만드는 것 뿐만이 아닌 분류하는 문제까지 해결할 수 있는 모델을 explicit model이라고 한다.
    - binary image가 n개의 픽셀로 이루어졌다면, 이 이미지는 2^n개의 state를 가지며 2^n-1의 parameter로 표현할 수 있다.
    - 하지만 parameter가 너무 많아지므로 만약 픽셀끼리 독립적이라고 가정하면, 2^n개의 state를 가지지만, n의 parameter로 표현할 수 있다.
    - fully dependent는 parameter가 너무 많고, independent는 너무 적어서 그 중간을 취하게 된다.
    - Conditional Independence에서 중요한 점은 Chain Rule에서 condition을 줄일 수 있다는 것이다.
    - Markov Assumption을 가정했을때는 parameter가 2n-1로 줄게된다.
    
* Auto-regressive Model
    - MNIST 이미지 784개의 pixel을 순서를 어떻게 매기는지가 중요하다.
    - 이중 NADE는 이전의 모든 pixel을 다 고려해 계산하는 모델이다. (Explicit model)
    - Pixel RNN은 RNN을 이용해 rgb를 예측하고, pixel 순서에 따라 Row LSTM, Diagonal BiLSTM 등으로 나뉜다.
    
* Variational Auto-Encoder
    - Variational Inference는 Posterior distribution을 찾기 위해 variational distribution으로 근사하는 과정을 말한다.
    - 이때 사용되는 것이 KL-Divergence이다.
    - 하지만 KL-Divergence를 줄이는 것이, ELBO(Evidence Lower BOund)를 높히는 것과 같은 의미를 가진다.
    - ELBO는 다시 Reconstruction Term과 Prior Fitting Term으로 나뉜다.
        - Reconstruction은 Encoder와 Decoder를 거치고 나서 원본과의 차이(loss)를 말하는 것이다.
        - Prior Fitting Term은 latent가 prior과 비슷할지를 말하는 것이다.
    - VAE는 implicit model로 얼마나 그럴싸 한지 알 수 없다.
    - Prior Fitting Term은 미분가능해야하지만, prior distribution에 Gaussian 외에는 미분가능한 함수가 별로 없다.
    
* Adversarial Auto-Encoder
    - VAE는 encoder의 prior fitting term이 KL을 사용하기 때문에 gaussian이 강제되는 문제가 있다.
    - AAE는 GAN을 활용해 prior fitting term을 대체한 모델이다.
    
* GAN
    - Generator와 Discriminator가 minimax game을 통해 generator를 얻는 모델이다.
    - GAN은 결국 실제 G와 학습하고자 하는 G 사이에 Jenson-Shannon Divergence를 줄이는 것과 같다.
    
* Line plot
    - 연속적으로 변화하는 값을 순서대로 나타내고 선으로 연결한 그래프
    - 시간/순서에 따른 변화를 나타내는데 적합하고 추세를 살피기 위해 사용된다.
    - line을 구분하는 요소에는 색상, 마커, 선의 종류가 있다.
    - Noise로 인해 인지적인 방해가 생길 수 있으므로 smoothing을 이용해 전체적인 추세를 파악할 수 있다.
    - 꼭 clean한 line plot이 좋은 것만은 아니다. (정확한 정보를 얻고 싶을때와 전체적인 추세를 알고 싶을때가 다르다)
    - 간격을 일정하게 설정하고, 있는 데이터에만 마커를 표시한다.
    - 보간(interpolation)은 추세를 알고 싶을때 사용한다.
    - 한 데이터에 대해 단위가 다른것을 표현할 때 이중 축을 사용할 수 있다.
    - 범례대신 line 끝에 라벨을 이용한다.
    - min, max를 annotation 하는 것이 도움이 될 수 있다.
    - uncertainty를 표시하기위해 연한색으로 표시 할 수 있다.

* Scatter plot
    - 점을 사용하여 두 feature간의 관계를 나타내는 그래프
    - 색, 모양, 크기로 점을 구분한다.
    - Scatter plot을 이용하여 상관 관계를 확인할 수 있다.
    - 또, cluster, gap in values, outlier도 확인할 수 있다.
    - 점이 많아 분포를 파악하기 힘들때 투명도 조절, 히스토그램, Contour plot을 이용한다.
    - 인과관계와 상관관계는 다르다.
    - 추세선 정보를 사용해 패턴을 보여줄 수 있다.

---

### 2. 과제 수행 과정 / 결과물 정리

선택과제 솔루션을 보며 내 코드와 비교하며 틀린부분은 없는지 수정해보았다.

---

### 3. 피어세션 정리

랜덤피어세션을 통해 다른 팀들은 논문을 찾아보기로 했다는 것을 알게되었다.

그래서 우리 조도 강의중에 이야기나온 중요한 논문을 리뷰하기로했다.

또, 팀 회고록 작성을 하고 각자 이야기해보았다.

팀원들의 솔직한 생각들을 들을수 있어서 값진 시간이었고, 더욱 친해진 것 같다고 생각했다.

---

### 4. 학습 회고

이번주는 정말 바쁘고 빠르게 지나간 것같다.

나도 이번주엔 적극적으로 참여한것같아 뿌듯했고, 다음주도 힘내서 해야겠다.