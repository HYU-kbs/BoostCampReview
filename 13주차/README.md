# 10 / 25 - 29

### 1. 강의 복습

HRNet
- Classification에선 고해상도의 이미지에서 conv와 pooling으로 저해상도의 feature map을 생성한다.
- 저해상도의 feature map은 classification을 하는데 효율적인 연산이 가능하지만, segmentation에서는 자세한 정보를 유지하지 못한다.
- DeconvNet, SegNet, U-Net은 feature map을 고해상도로 복원하는 방식으로 작동했다.
- DeepLab은 dilated conv 연산으로 해상도를 적게 줄이면서 dense한 feature map을 얻었다.
- classification based network들은 time complexity와 position sensitivity가 segementation에서 문제가 되어 새로운 구조의 모델이 필요하다.
- HRNet은 고, 중, 저 해상도의 특징을 모두 갖는 모델이다.
- 고해상도로는 원본의 1/4로 줄여 상대적으로 높은 해상도를 갖는다.
- 새로운 stream이 생성될때, 이전 해상도의 1/2로 갖는다.
- 마지막에서 다중해상도의 정보들을 융합한다.
- 각기 다른 해상도의 융합을 위해 Strided conv와 Bilinear upsampling, 1x1 conv 연산을 이용했다.
---

### 2. 과제 수행 과정 / 결과물 정리

DeeplabV3+ 모델로 학습을 진행시켰다.

또, mmsegmentation으로 HRNet을 학습시켰다.

---

### 3. 피어세션 정리

https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-14/discussions

- Unet계열에 backbone으로 efficientnet을 사용. 더 좋은 것을 쓰더라도 성능이 좋아지지 않았다.
- 최고 성능은 efficientnet-b4로 epoch 150에서 mIoU 0.652
- Unet++의 backbone으로 resnext 사용. epoch 80에서 best mIoU가 나왔다.
- 학습결과를 시각화하여 눈으로 살펴본 결과 :
- train set에서는 꽤 정확한 학습을 하였지만, valid set에선 아쉬운 부분들이 많았다.
- 얇은 끈을 예측하지 못하거나, 빛이나 밝기에 예민하여 학습하지 못한 부분이 있었다. (augmentation으로 해소하자)
- 또한 찌그러진 캔의 모습, 거울에 반사된 모습을 학습하지 못했다.
- 종이컵이나 비닐봉지 등 일관되지 않은 ground truth labeling도 영향이 있었을 것이다,
- focal loss를 사용하여 class imbalance 문제를 해결하고자 했다.
- wandb의 report 기능을 이용하면 다른 실험들 간의 그래프를 비교하여 글을 작성하기 좋다.
- 또한 augmentation을 이용하여 성능에 향상이 있었다.
- scheduler로 StepLR보다 CosineAnnealing을 이용해 성능에 향상이 있었다.
- smp 라이브러리에 HR-Net backbone이 있어 사용해 볼 것이다.
- 이미지 후처리로 성능을 높일 수 있을것이다.
- 학습 시 RAM에 이미지를 미리 불러와 병목현상을 줄여 학습시간을 단축시킬 수 있을것이다.

---

### 4. 학습 회고

pseduo labeling이나 후처리를 통해 성능향상에 집중할 것이다.