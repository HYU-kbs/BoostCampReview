# 9 / 13 (월)

### 1. 강의 복습

- Computer Vision
  - CNN은 하나의 black box이지만, visualization으로 내부를 파악할 수 있다.
  - Deconvolution을 통해 low level에서는 선과 같은 특징을, high level에서는 의미있는 모습을 학습한다는 것을 확인했다.
  - Filter weight를 초기 conv에서만 시각화를 진행한다. (그 이후는 차원이 높아 해석하기 힘듬)
  - Visualization은 모델의 행동을 분석하는 방법과 모델의 결정을 분석하는 방법이 있다.
  - **Model behavior Analysis**
  - Model behavior Analysis는 이미지의 feature vector를 DB의 많은 이미지들과의 Nearest Neighbor 방법으로 찾아낸다고 설명한다.
  - 고차원의 feature를 dimensionality reduction으로 해석가능하게 한다.(t-SNE)
  - 또, 중간의 layer activation을 masking하거나 patch들을 분석한다.
  - 클래스를 구분하는 layer에서 gradient ascent방법으로 visualization도 가능하다.
  - **Model decision Analysis**
  - Occulsion map(mask)을 이용해 이미지의 중요한 부분을 알아내 heatmap으로 관찰할 수 있다.
  - 또, backpropagation을 통해 saliency map을 얻을 수 있다.
  - ReLU를 사용했을때, 일반적인 backpropagation이 아닌 guided-backpropagation(forward에 ReLU를 통과할때 0이 된 부분은 backward도 0으로 하자)를 사용하면 조금더 clear한 saliency map을 얻을 수 있다.
  - Class Activation Mapping(CAM)은 CNN이후에 global average pooling을 거쳐 fc를 가진 모델에 사용한다.
  - classification task에 위치에 대한 정보가 없었지만, 어느정도 object detection도 수행할 수 있다.
  - 모델의 구조에 제약이 없는 Grad-CAM은 CNN만 있으면 가능하다. 또, guided backpropagation과 응용할 수도 있다.
  - 이외에도 classification의 이유도 볼 수 있는 SCOUTER도 있다.


---

### 2. 과제 수행 과정 / 결과물 정리

선택과제 1번은 CNN을 visualization 해보는 것으로 CNN model에 hook을 만들어 weight나 activation 값을 가져온 뒤 시각화 해보았다.

시각화 특성상 3개의 채널을 가진 첫번째 conv layer를 시각화했다.

또, 학습된 모델에 input을 넣었을 때, 중간 activation의 값을 가져와 시각화해보았다.

다만, Grad-CAM을 이용해 시각화해보려 했으나 어려움을 겪었다.

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

CAM, Grad-CAM의 유도 수식에 관해 이야기했다.

또, class visualization의 수식에 관해서도 이야기했다.

마지막으로 further question에 대한 답을 서로 이야기해 보았다.

- 왜 filter visualization에서 주로 첫번째 convolutional layer를 목표로할까요?
  - 첫번째 conv layer가 3개의 채널을 갖기 때문.
  - input image가 있다면 다른 layer의 activation을 가지고 Grad-CAM을 할 수 있다.

- Occlusion map에서 heatmap이 의미하는 바가 무엇인가요?
  - classification하는데 중요한 부분을 의미한다.

- Grad-CAM에서 linear combination의 결과를 ReLU layer를 거치는 이유가 무엇인가요?
  - 중요하지 않은 음수부분을 0으로 나타내기 위해서다.
  - 그렇다면 음수부분이 중요하지 않은 이유는?
  - 이 부분은 멘토님께 질문하기로 했다.

---

### 4. 학습 회고

Computer Vision은 모델을 시각화해보면서 학습을 진행할 수 있다는 점이 와닿는 시간이었다.

NLP와 다르게 시각화 해보면서 재미있었고 흥미로웠다.


# 9 / 14 (화)

### 1. 강의 복습

- Computer Vision
  - Instance Segmentation은 Semantic Segmentation에서 instance끼리의 구분이 가능하여야 한다.
  - 이러한 Instance segmentation의 첫 시작은 Object detection으로 구현했다.
  - Mask R-CNN은 interpolation을 이용하는 ROI Align을 이용했고, 기존의 Faster R-CNN에 있는 box regression, classification과 새로운 mask branch를 이용했다.
  - You Only Look At CoefficienTs(YOLACT)는 single-stage로 real time으로 가능한 모델이다. FPN구조를 사용하고, Mask R-CNN과 다르게 mask보다 훨씬 적은 수의 prototype을 만들어 이들의 weighted sum으로 mask를 만들어낸다.
  - Panoptic Segmentation은 배경도 고려하는 task이다.
  - UPSNet은 Semantic과 Instance head를 묶어 Panoptic head에서 최종적인 segmentation map을 만들어낸다.
  - VPSNet은 비디오에서 가능한 모델로 여러 frame에서의 ROI를 참고해서 사용한다.
  - Landmark Localization은 얼굴이나 사람의 포즈를 추정하고 이를 추적할 때 주로 사용된다.
  - Coordinate regression 방법보단 heatmap classification이 성능은 더 높지만 계산이 오래걸린다.
  - 사용되는 모델은 stacked hourglass network이다.
  - 사람의 모든 곳에 landmark를 만든다면, 3d로도 응용이 가능하다.
  - DensePose R-CNN은 Faster R-CNN에서 3d surface regression branch가 합쳐진 모델이다.
  - RetinaFace는 FPN구조에 classification, bounding box, 5 point regression, mesh regression 등 multi-task branch를 가진 모델이다.
  - KeyPoint를 이용한 dection model은 CornerNet과 CenterNet이 있다.


---

### 2. 과제 수행 과정 / 결과물 정리

Grad-CAM을 구현하는 부분에서 register_hook을 이용해 구현하는 것을 알게되었다.

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

- 어떤 방향으로 P stage에 임해야 할지 회의함
    - 리더보드 스코어보다는 여러 가지 실험을 해보기.
- 선택과제 1번을 같이 살펴봄
    - todo 5번에서 register_hook 을 사용하면 된다.
    - loss 없이 gradient를 어떻게 backprop? → 각 gradient는 해당 activation map이 결과 생성에 얼마나 영향을 미치는지를 나타낸다.

---

### 4. 학습 회고

오늘의 강의는 상당히 난이도가 있었다. 어려운 내용인만큼 추가로 검색해서 공부해야겠다고 생각했다.

