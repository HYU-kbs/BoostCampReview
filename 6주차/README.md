# 9 / 6 (월)

### 1. 강의 복습

- Computer Vision
  - AI는 사람의 지능을 컴퓨터 시스템으로 구현하는 것을 말한다.
  - 사람의 지능은 인지능력, 지각능력, 기억 및 사고능력 등을 말한다.
  - 지각능력(perception)이 바탕이 되어야 다른 능력의 학습도 가능하다.
  - 컴퓨터 그래픽스(CG)는 사물을 렌더링하는 분야이다.
  - 반대로 컴퓨터 비전(CV)은 CG의 반대이다.(Inverse rendering)
  - 지각 능력의 구현에는 사람의 시각 능력에 대한 이해가 필요하다.
  - 과거의 머신러닝에는 feature extraction을 사람이 구현했다.
  - 지금의 딥러닝에는 입력과 출력만 가지고 end-to-end로 구현이 가능하다.
  - 이미지 분류 문제는 이상적으로 KNN 알고리즘으로 해결가능하다.
  - single fully connected layer로 학습한 분류기도 input의 약간의 변형에는 대응하지 못한다.
  - CNN은 많은 CV task의 backbone 모델로 활용된다.
  - AlexNet은 더 깊은 층과 ImageNet으로 학습하고, ReLU와 Dropout을 사용했다.
  - CNN에서 FC layer로 가는 층 사이에는 AvgPool또는 flatten이 있어야한다.
  - Local Response Normalization(LRN)은 현재는 잘 사용하지 않고 batch normalization을 더 사용한다.
  - 또, 현재는 11x11처럼 큰 필터 사이즈는 사용되지 않는다.
  - Receptive field는 한 CNN layer에서 input의 어느정도를 참고하는지 말한다.
  - VGGNet은 AlexNet보다 더 깊고 간단한 구조를 갖고있다.
  - 작은 kernel 사이즈를 깊게쌓아 큰 receptive field를 가졌다.

---

### 2. 과제 수행 과정 / 결과물 정리

VGG model 구조를 이용해 QuickDraw 데이터셋을 classification 하는 과제였다.

과제 자체는 크게 어렵지 않았다. 다만 약간의 baseline에서 오타가 있어서 고치는 데 시간이 걸렸다.

Pretrained된 VGG model의 feature extraction 부분을 freeze하여 사용한 것이 성능이 더 좋았다.

---

### 3. 피어세션 정리

피어세션에서 새 캠퍼들과 그라운드룰을 정했다.

그리고 앞으오 notion을 활용하여 회의록 작성과 질문을 미리 올려놓기로 했다.

---

### 4. 학습 회고

LEVEL2의 첫날은 전에 배운 내용들이 많아서 크게 어렵지 않았다.

내일은 시각화 강의도 함께 들어야겠다고 생각했다.

