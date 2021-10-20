# 10 / 18 (월)

### 1. 강의 복습

Fully Convolutional Network
- VGG network의 backbone을 사용하고, linear을 convolution으로 대체했다.
- Transposed Convolution을 이용해 Pixel wise prediction을 수행했다.
- Convolution layer는 FC와 달리 위치정보를 해치지 않고, 입력에 상관없이 동작이 가능하다.
- Transposed Convolution은 down sampling된 feature map에 up sampling할때 학습되는 kernel이다.
- Deconvolution이라고 혼용하는 용어지만, 엄밀한 의미에서 input과 값이 같지 않으므로 역연산은 아니다.
- Up Sampling 과정에서 한번에 이미지 크기를 키우면 정보의 손실이 있기 때문에 skip-connection을 이용했다.


---

### 2. 과제 수행 과정 / 결과물 정리

baseline code를 실행해서 간단한 결과를 눈으로 확인했다.

---

### 3. 피어세션 정리

앞으로 대회에 대해 이야기하며 사용할 라이브러리, 앙상블 방법등을 이야기했다.

또, 결과를 다듬는 과정인 후처리도 중요할 것이라고 이야기했다.

---

### 4. 학습 회고

데이터에 대해 알아보고 싶은점이 있었다.

validation set은 어떤 방법으로 나누어 진것인지와 segmentation 픽셀이 겹치는 것은 없는지를 확인해보고 싶었다.




# 10 / 19 (화)

### 1. 강의 복습

FCN model의 한계점을 극복한 모델들
- FCN model은 객체의 크기가 크거나 작으면 잘 예측하지 못했다.
- 또, deconvolution 절차가 간단해 detail한 모습이 사라지는 문제가 있다.
  
Decoder를 개선한 모델로 DeconvNet과 SegNet이 있다.

- DeconvNet은 encoder와 decoder를 대칭으로 만든 모델이다.
- MaxPooling에 대한 대칭으로 UnPooling연산을 하는데, 지워진 경계에 대한 정보를 기록했다가 복원하는 연산이다.(return_indices=True)
- Deconv 연산은 그 안의 내용을 채워넣는 역할을 한다.
- 얕은 층은 전반적인 모습의 특징을 잡아내고, 깊은 층은 세부적인 모습의 특징을 잡아낸다.
- SegNet은 자율주행 분야에서 빠른 segmentation을 하기위해 고안되었다.
- SegNet은 DeconvNet 가운데의 Convolution을 없애 parameter를 줄였고, Sparse를 Dense로 바꾸어주는 Deconv를 Conv로 대체했다.

Skip Connection을 활용한 모델로 FC DenseNet과 Unet이 있다.
- FC DenseNet은 FCN의 sum대신 concat으로 skip connection을 구현했다.
- Unet은 Encoder의 정보를 Decoder로 skip connection으로 전달해주었다.

또, Receptive Field를 확장한 모델로 DeepLab v1과 DilatedNet이 있다.
- Dilated(Atrous) Convolution은 parameter의 수는 늘리지 않고 receptive field를 넓이기 위해 고안된 방법으로 kernel 내부의 값들 중간에 0으로 채워넣는 방법이다.
- 또한 마지막에 Dense CRF 후처리를 통해 더욱 정교한 Segmentation이 가능하다.


---

### 2. 과제 수행 과정 / 결과물 정리

Date를 fiftyone 모듈로 살펴보려고 했다.

하지만, segmentation은 각각의 pixel이 영역을 나타내는 것이 아니라 물체의 테두리만 나타낸다는 것을 알았다.

pycocotools의 anntomask를 이용해 mask화 할 수 있다는 것을 알게되었다.

또, wandb의 log를 이용하면 학습 결과를 이미지로 볼 수 있다는 것을 알게되었다.


---

### 3. 피어세션 정리

Segmentation의 후처리로 CRF가 굉장히 중요할 것이다.

또, ensemble 결과가 좋지 않을 경우엔 후처리가 더 중요할 것이다.

---

### 4. 학습 회고

아직 학습을 시작한 것은 아니지만, 나중에 학습 데이터에 대해 의문이 들경우 EDA를 진행해야겠다고 생각했다.

