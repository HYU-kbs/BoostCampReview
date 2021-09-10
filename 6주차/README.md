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

그리고 앞으로 notion을 활용하여 회의록 작성과 질문을 미리 올려놓기로 했다.

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



# 9 / 8 (수)

### 1. 강의 복습

- Computer Vision
  - AlexNet에서 VGGNet으로 넘어가면서 깊은 모델이 receptive field가 넓어져 좋은 성능을 낸다는 것을 알았다.
  - 하지만 너무 깊게 쌓으면 계산이 복잡할 뿐만 아니라 gradient vanishing/exploding이 일어나 성능이 떨어진다.
  - GoogleNet은 Inception Module의 구조를 가지며 한 layer에서 다양한 크기의 kernel을 사용했다.
  - 또, 1x1 conv를 이용해 dimension을 낮추어 사용했다.
  - Auxiliary classifier를 두어 back propagation이 잘 이루어지도록 했다. 이는 test time에 사용되지 않는다.
  - ResNet은 shortcut connection을 이용해 vanishing 문제를 해결했다.(back propagation 경로의 수가 늘어났다.)
  - 이후 DenseNet은 각각의 layer output을 다음 layer뿐만 아니라 그 이후에도 연결시켜 주었다.
  - SENet은 채널에서의 weight를 곱해줘 attention score를 계산해서 중요도를 계산한다.
  - EfficientNet은 width, depth, resolution에서 적절한 방법으로 향상을 시켜 성능을 높혔다.
  - Deformable Convolution으로 고정된 구조가 아닌 여러 영역에서 feature를 추출할 수 있다.



- Data Visualization
  - Interactive 시각화는 사용자가 원하는 인사이트가 각자 다를 수 있기 때문에 사용한다.
  - 정적 시각화는 원하는 메세지를 압축해서 담는 발표자료에 적합하다.
  - Matplotlib도 인터랙티브를 지원하지만, 주피터나 local에서만 사용 가능하다.
  - Plotly와 Plotly Express는 문서화와 예시가 잘 되어 있다.
  - Bokeh는 matplotlib과 유사하지만 문서화가 부족하다.
  - Altair는 문법이 pythonic하지 않다는 단점이 있다.

---

### 2. 과제 수행 과정 / 결과물 정리

이번 과제는 다양한 augmentation을 활용하면서 성능을 비교하는 과제이다.

먼저, seed를 고정하는 코드를 붙여넣고 기본적인 augmentation을 적용한 것을 학습시켰다.

그리고 albumentation으로 blur를 적용한 것과 비교했다.

또 resize를 하고 blur한 것과 blur를 적용하고 resize를 한것도 비교했다.

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

He initialization과 Xavier initialization은 어떤 activation function을 사용하는지에 따라 결정된다.

또, 과제에 대한 생각을 모두 공유해보았다.

- Blur를 사용했을 때, 흰 배경이 많이 줄어들어 성능이 좋아진다.
- CNN은 공통적인 특성을 찾는데 장점이 있는 방법인데, 일반적인 선보다 blur된 선들이 특징을 찾는데 더 유용하다.
- 크게 Resize된 이미지에는 사람의 손떨림 같은 noise가 더 부각되어 학습에 방해가 될 수 있다.
- Blur를 적용한 뒤 Resize를 적용하면 성능이 더 좋다는 것을 실험적으로 확인하였다.
- Resize를 적용한다는 것을 AveragePooling으로 이해한다면, blur를 먼저 적용했을때 검은 선들이 먼저 배경으로 퍼지고 resize를 하면 정보가 덜 소실된다.
- CNN을 freeze하고 학습시키는 것이 ImageNet과 완전히 다른 image를 학습시키기 때문에 성능이 더 떨어진다.

---

### 4. 학습 회고

과제의 난이도는 어렵지 않았지만 이것저것 생각할 것이 참 많았다.

금요일의 솔루션 세션때 확실한 답을 얻었으면 좋겠다.


# 9 / 9 (목)

### 1. 강의 복습

- Computer Vision
  - Semantic Segmentation은 한 픽셀이 어떤 카테고리에 속해있는지 masking하는 task이다.
  - 의학 이미지나 자율주행 등에 활용된다.
  - Fully Convolutional Networks(FCN)는 segmentation의 첫 end-to-end 모델이다.
  - 마지막 layer가 fc가 아닌 1x1 cnn이므로 입력사이즈에 상관없이 출력이 가능하다.
  - 하지만 FCN의 결과인 feature map은 해상도가 줄여진 output을 얻게 되는데, 이때 upsampling을 이용한다.
  - Upsampling의 종류에는 Transposed convolution과 Upsample and convolution이 있다.
  - Transposed convolution은 적절한 kernel size와 stride를 이용해 upsampling하지만, overlap문제가 발생한다.
  - Upsample and convolution은 영상처리에 이용되는 Nearest Neighbor이나 Bilinear같은 interpolation과 학습가능한 convolution을 같이 이용한다.
  - FCN에서 처음 layer와 후반 layer 모두 중요한 정보를 담고있기 때문에 최종출력에 중간의 결과들을 활용한다.
  - UNet역시 FCN과 비슷한 아이디어를 사용, contracting path를 지나 작은 activation map을 구한다.
  - 또, expanding path를 지나 skip connection을 이용해 upsampling한다.
  - UNet에서 각 feature map은 항상 짝수이도록 신경써야한다.
  - DeepLab은 Conditional Random Fields(CRFs)와 Dilated Convolution(Atrous Convolution)이라는 개념을 소개했다.
  - CRFs는 각 물체의 boundary가 확산하면서 정교한 결과를 얻을 수 있다.
  - Dilated convolution은 커널이 upsampliing으로 확산할 때 약간의 간격을 두어 receptive field가 더 넓어지도록 한다.

---

### 2. 과제 수행 과정 / 결과물 정리

이번 과제는 VGGNet의 마지막 FC layer를 1x1 convolution으로 바꾸어 일종의 heat map을 관찰하는 과제였다.

과제는 어렵지 않았지만, 이를 heatmap처럼 바꿀수 있다는 사실이 신기했다.

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

Cross-entropy에서
- 정보의 양이 많으면 정보의 희귀성이 떨어지고, 정보의 양이 적으면 정보의 희귀성이 높아진다.
- 희귀할 수록 유의미한 정보이다.

FCN같은 경우에는 input의 크기가 고정될 필요가 없는 것인가?

- 그럴 필요가 없다. FCN에서 학습하는 대상이 커널뿐이기 때문에 커널은 어떤 이미지든 슬라이스를 할 수 있기 때문이다.

---

### 4. 학습 회고

CV에서 classification에서 segmentation 분야로 확장하는 흐름을 익힐 수 있었고, 또 신기했다.

내용도 점차 어려워지지만 잘 정리하고 복습해서 내용을 익혀야겠다.