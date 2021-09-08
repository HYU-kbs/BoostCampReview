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


# 9 / 7 (화)

### 1. 강의 복습

- Computer Vision
  - 딥러닝 분야에서 데이터의 부족은 항상 문제가 되어왔다.
  - Data Augmentation은 train dataset이 bias된 문제를 완화해준다.
  - 종류로는 Brightness, Rotate, Flip, Crop, Affine Transform, CutMix, RandAugment 등이 있다.
  - Transfer Learning으로 미리 배운 지식들을 다른 데이터셋에도 적용할 수 있다.
  - Knowledge Distillation은 이미 학습한 teacher model을 student model에 지식을 전해주는 학습 방법이다.
  - pseudo-labeling에 활용되기도 한다.
  - Softmax의 temperature를 사용해 더 유용한 정보를 얻기도 한다.
  - Distillation loss는 teacher와 student의 inference의 KLdiv이다.
  - Student loss는 student의 inference와 true label의 cross entropy이다.
  - Semi-supervised learning은 unlabeled data를 pseudo-labeling한 후에 학습에 참여시킨다.

- Polar coordinate
  - Polar plot은 x, y가 아닌 r과 theta를 이용하여 표현한다.
  - 회전이나 주기성을 나타내기에 적합하다.
  - Radar Chart는 별모양으로 생겨 star plot이라고도 한다.
  - 데이터의 quality를 표현하기 적합하다.
  - 각 feature는 독립적이면서 척도가 같아야한다.
  - 다각형의 면적보다는 feature의 순서에따라 바뀌기도 한다.
  - feature가 많아질수록 가독성이 떨어진다.
- Pie Chart
  - 원을 부채꼴로 분할하여 표현하는 통계차트이다.
  - 많이 사용하지만 비교가 어렵고 유용성이 떨어져 bar plot과 함께 사용하자.
- MissingNo는 결측치를 체크할 수 있는 시각화 라이브러리이다.
- TreeMap은 계층적 데이터를 직사각형을 사용하여 포함관계를 표현한다.
- WaffleChart는 와플형태로 discrete하게 값을 나타내는 차트이다.
- Venn은 집합에서 사용하는 벤 다이어그램이다.

---

### 2. 과제 수행 과정 / 결과물 정리

없음

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

필수과제 transform의 normalize 부분에 대해 이야기했다.

Augmentation한 후에 생기는 이미지의 검은 부분(빈 부분)에 대해 이야기했다.

ResNet의 skip connection과 KL div와 CrossEntropy에 대해 이야기했다.


---

### 4. 학습 회고

첫주는 굉장히 여유가 있는 한 주인것 같다.

남는 시간에 논문을 읽어야겠다고 생각했다.