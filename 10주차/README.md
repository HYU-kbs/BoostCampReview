# 10 / 5 (화)

### 1. 강의 복습

- Cascade RCNN은 기존 RCNN의 threshold IoU에 변화를 주었다.
- 단순히 IoU threshold만 높인다면 낮은 IoU에 대해 성능이 떨어진다.
- Cascade RCNN은 여러개의 RoI head를 학습시키되, 각각의 threshold를 다르게 설정하여 학습한다.
- 기존의 CNN은 고정된 정사각형 filter를 이미지 위로 sliding하여 학습하므로 Affine, Viewpoint, Pose 등 Geometric transformation에 한계가 있어 이미지에 Augmentation을 적용했다.
- Deformable Convolutional Network는 고정된 filter가 아닌 다양한 shape과 size를 가진 filter를 이용한다.
- filter에 offset을 참고해 conv 연산이 이루어진다.
- ViT는 transformer를 classification에 적용한 모델이다.
- 하지만 ViT는 굉장히 많은양의 data가 필요하고, backbone으로 사용하기 어렵다.
- DETR은 Detection에 transformer를 적용한 모델로 NMS같은 post processing이 필요없다.
- Swin Transformer는 window라는 개념을 도입해 embedding을 나누어 cost를 줄였다.

---

### 2. 과제 수행 과정 / 결과물 정리

mmdetection의 github docs를 보면서 학습을 시켜보았다.

---

### 3. 피어세션 정리

github discussion을 이용해 회의록 작성 및 질문이나 실험 공유등을 진행하기로 했다.

또, metric mAP가 bounding box가 늘어남에따라 같이 증가하는 이유에 대해 생각해 보았다.

---

### 4. 학습 회고

이번주는 여러가지 실험을 해보면서 성능을 높이려고 한다.

mmdetection 코드를 잘 분석하며 커스텀을 진행할 것이다.




# 10 / 6 (수)

### 1. 강의 복습

- Bag of Freebies는 inference 비용을 늘리지 않고 정확도를 향상시킨는 방법이다.
- Augmentation, Semantic Distribution bias, Label smoothing, Bbox regression, GIoU 등이 있다.
- Bag of Specials는 inference시에 시간이 조금 더 걸리지만, 성능에 도움이 된다.
- Enhancement of Receptive Field, Attention Module, Feature Integration, Activation Function, Post-processing method 등이 있다.
- YOLOv4에서 CSPNet을 사용해 경량화했다. 또, Augmentation으로 Mosaic와 Self-Adversarial Training을 이용했다.
- PAN과 SAM을 수정하여 사용했고, Cross mini-batch normalization을 이용했다.
- M2Det은 Multi-level, multi-scale feature pyramid를 제안했다.(MLFPN)
- CornerNet은 Anchor box가 없는 1 stage detector이다.

---

### 2. 과제 수행 과정 / 결과물 정리

Swin-T를 backbone으로 하여 학습을 진행시켰다.

---

### 3. 피어세션 정리

Augmentation에 대해 이야기해보았다.

---

### 4. 학습 회고

새로 배운 Mosaic Augmentation을 구현해 보아야겠다고 생각했다.