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