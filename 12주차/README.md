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


# 10 / 20 (수)

### 1. 강의 복습

FCN model의 한계점을 극복한 모델들
- Receptive Field를 키워나가던 Deeplab은 v2에서 여러가지 dilated rate를 가지는 branch형태로 바뀌었다.
- 여러가지 dilated rate을 가진 conv는 서로다른 크기의 object에 집중한다.
- PSPNet은 주변의 특징을 고려하여 mismatched relationship을 해결하고자 했고, category간의 관계를 사용하여 Confusion categories를 해결하고자 했고, 작은 객체들도 global contextual information을 이용해 Inconspicuous Classes 문제를 해결하고자 했다.
- 또, Global Average Pooling을 이용해 주변 정보를 파악해서 객체를 예측하는데 사용했다.
- Deeplab v3는 global average pooling과 concat을 사용했다.
- Deeplab v3+는 Encoder-Decoder의 구조를 가지며 encoder에서 손실된 정보를 decoder에서 점진적으로 복원하는 구조를 가진다.
- 또 backbone으로 modified Xception을 사용했는데, Depthwise와 Pointwise convolution을 하였다.


---

### 2. 과제 수행 과정 / 결과물 정리

baseline코드에서 PSPNet으로 바꾸어 학습을 진행시켰다.


---

### 3. 피어세션 정리

서로 다른 모델을 맡아 실험을 진행하기로 했다.

---

### 4. 학습 회고

여러가지 augmentation을 적용해보며 성능을 실험해봐야겠다.




# 10 / 21 (목)

### 1. 강의 복습

Unet 계열의 모델들
- Unet은 Contracting Path와 Expanding Path가 U자 형태로 나열이 되어있어 이름이 붙여졌다.
- Contracting path는 일반적인 특징을 추출하는 단계이고 Expanding path는 localization을 수행한다.
- 또, 같은 level의 결과를 skip-connection으로 concat하는 형태이다.
- 의료분야에서 있을법한 Augmentation으로 Random Elastic deformation을 이용했다.
- 또한 인접한 셀을 분리하기 위해 경계부분에 가중치를 제공했다.
- 하지만 Unet은 깊이가 4로 고정되어있어 데이터셋마다 성능을 보장하지 못하고 skip-connection도 단순하다.
- Unet++는 encoder를 공유하는 다양한 깊이의 Unet을 만들었다.
- 또한 DenseNet에서 사용한 Dense skip connection으로 보다 복잡한 skip connection을 구현했다. 이는 여러 depth를 ensemble한 효과를 갖는다.
- 또한 hybrid loss를 이용했다.
- 그러나 parameter와 memory가 증가했고, 다양한 scale에서의 connection이 되지않는 한계가 있다.
- Unet3+는 Full-scale skip connection으로 다양한 scale(Conventional, inter, intra)의 connection을 가졌다.
- 또, noise로 인한 false-positive를 방지하기 위해 Classification-guided module을 사용했다.

---

### 2. 과제 수행 과정 / 결과물 정리

Unet과 PSPNet을 각각 50 epoch씩 학습시켜 결과를 비교했다.

---

### 3. 피어세션 정리

각자의 모델의 성능을 비교하였다.

---

### 4. 학습 회고

baseline이 jupyter notebook으로 되어있는데, 