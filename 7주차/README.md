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



# 9 / 15 (수)

### 1. 강의 복습

- Conditional generative model은 조건에 맞게 이미지를 생성해준다.
- 일반적인 generative model과 다르게 조건에 맞게 랜덤 샘플을 만든다.
- 예시로 audio super resolution, machine translation, article generation with the title등이 있다.
- 또, image의 style transfer, super resolution, colorization등이 있다.
- Super resolution은 이전에 real과 fake를 구분할 때 Mean absolute error와 mean squared error를 사용했다. 이 방법은 실제 patch들의 평균으로 학습하게 되어 상당히 blurry한 image가 만들어진다. GAN loss는 blurry한 image가 discriminator에 의해 걸러지므로 더 좋은 성능을 내게된다.
- Image translation은 style이나 color등 image의 domain을 바꾸는 일이다.
- Pix2Pix는 GAN loss와 L1 loss를 함께 사용했다.
- CycleGAN은 unpaired data에 대해서도 동작한다. loss는 양방향으로의 GAN loss와 cycle-consistency loss를 이용한다.
- Perceptual loss는 pretrained model을 활용한 loss로, image transform network의 학습에 이용된다.
- Content target의 feature map과의 L2 loss와 Style target의 gram metric을 이용한 L2 loss를 활용한다.


---

### 2. 과제 수행 과정 / 결과물 정리

Conditional GAN을 quickdraw dataset을 이용하여 구현하는 문제이다.

generator가 noise와 class를 concatenate하여 생성한 이미지를 discriminator가 구분하면서 adversary하게 학습하는 구조이다.

다만 discriminator 부분에서 이미지 차원이 맞지않아 에러가 발생했다.

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

pix2pix 에서 L1 loss를 같이 사용하는 이유는 결국 generator가 똑같은 이미지만 생성하는 것을 막기 위함이다. 학습을 너무 쉽게 한다.

학습은 generator와 discriminator가 번갈아 가며 학습한다.

noise z는 task에 따라 있을 수도, 없을 수도 있다.

---

### 4. 학습 회고

GAN의 결과물들을 보면서 CV 과목에 더 흥미가 생겼다.

눈으로 중간 과정이나 결과물을 볼 수 있다는 것이 CV의 가장 큰 장점인것 같다.



# 9 / 16 (목)

### 1. 강의 복습

- Multi-modal learning은 다른 종류의 데이터로 학습하는 것을 말한다.
- 서로 다른 데이터끼리의 표현 방법의 차이가 있기 때문에 multi-modal learning에 어려움이 있다.
- 또, 데이터끼리의 matching이 unbalance한 문제도 있다.
- 쉬운 modality로만 학습하려는 경향이 있다.
- 많이 사용되는 architecture로 Matching, Translating, Referencing이 있다.
- Visual data & text의 matching task로 image tagging, image&food recipe가 있다.
- image tagging은 image에서 text로, text에서 image로 찾아주는 task이다.
- 각각의 matching된 데이터를 같은 embedding space로 바꾸고 거리를 줄이는 방향으로 학습된다. 서로 다른 데이터는 거리를 늘이는 방향으로 학습된다.
- image&food recipe는 image를 보고 recipe와 재료로, 반대로 recipe와 재료에서 image로 찾아주는 task이다.
- Translation task로는 image captioning과 text-to-image가 있다.
- Show, attend, and tell은 captioning의 단어에 따라 집중한 부분을 보여준다.
- text-to-image 모델은 generative model로 구현한다.
- Referencing task로는 Visual answering이 있다.
- Visual data & audio의 matching task로는 소리로 장소 구분하는 일이 있다(SoundNet).
- Translating task는 Speech2Face model, image to speech가 있다.
- Referencing task로는 Sound source localization이 있다.


---

### 2. 과제 수행 과정 / 결과물 정리

없음

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

필수과제 3번 quickdraw cGAN이 결과가 좋지 않았는데, 이에 대한 이유를 함께 고민했다.
- blur연산이나 모폴로지 팽창, 침식등을 수행하면 좋을것이다.(너무 sparse한 dataset image)
- epoch이 부족한 것 같다.
- 3개의 채널까지 필요 없을것이다.(gray나 binary)
- MNIST 같이 간단한 dataset으로 바꾸면 결과가 좋을 것이다.
- generator와 discriminator의 구조가 단순한 fc이다.

cGAN의 학습과정을 코드를 보면서 이해했다.

- 먼저 generator가 noise로 만든 이미지를 discriminator가 구분하여
- 그 다음 discriminator가 학습하는 구조이다.


---

### 4. 학습 회고

어려운 강의 내용이었지만, multi-modal이 어떤식으로 작동하는지에 대해 배웠다.

다 이해하지 못할만큼 어려웠기 때문에 나중에 꼭 복습해야겠다고 생각했다.



# 9 / 17 (금)

### 1. 강의 복습

- 3D는 우리가 사는 세상이며 AR/VR이나 3D printing, medical 분야에서 다루어진다.
- 이미지는 3D 세상을 2D에 projection한 것이다.
- 서로 다른 각도에서 찍은 2개의 사진이 있으면 3D point를 추출할 수 있다.
- 컴퓨터에서 3D는 multi-view image., volumetric, part assembly, point cloud, mesh, implicit shape한 방법으로 나타낼 수 있다.
- ShapeNet은 3D object들의 dataset이다. 또 PartNet도 part가 구분된 3D dataset이다.
- SceneNet은 Depth를 가진 실내 이미지 dataset이다. 또, ScanNet은 scan해서 찍은 실내 이미지이다.
- KITTI나 Waymo dataset은 자율주행을 위한 실외 이미지이다.
- 3D task로는 recognition, detection, semantic segmentation, generation등이 있다.


---

### 2. 과제 수행 과정 / 결과물 정리

3D focusing에 있어서 depth map이 중요하다는 것을 느꼈다.

---

### 3. 피어세션 정리

https://www.notion.so/HOME-771be0eeb7c846cb935860cfa7b143ea

새로운 팀 결성에 앞서 각종 TMI 방출과 팀회고를 했다.


---

### 4. 학습 회고

추석연휴동안 잘 쉬어서 다음 P stage를 대비하기로 했다.