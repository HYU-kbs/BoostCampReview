# 9 / 27 (월)

### 1. 강의 복습

- Object Detection 성능의 평가로 mAP(Mean Average Precision)이 있다.
- PR Curve는 Precision과 Recall을 각각 y축, x축에 그린 것이다.
- PR Curve의 아래쪽의 면적은 Average Precision이다.
- IOU(Intersection Over Union)은 겹친 부분에 대한 수치이다.
- 이외에도 속도를 측정하는 방법으로 FPS와 FLOPs(Floating Point Operations)가 있다.
- Two-stage detector는 사람과 마찬가지로 물체의 위치를 먼저 찾고, 예측하는 2가지 단계를 거치는 모델이다.
- Region Proposal의 방법중 sliding window는 다양한 비율의 box를 sliding시키는 방법이다.
- Selective Search 방법은 무수히 많은 영역을 합쳐나가는 방식이다.
- 찾아낸 RoI(Region of Interest)들을 같은 크기로 warping한 후 classification하는 방식이다.
- 그 후 bounding box를 regression으로 미세하게 수정해 나간다.
- R-CNN은 초기의 연구인 만큼 강제 warping이나, end-to-end가 아니여서 성능에 한계가 있었다.
- SPPNet은 이미지의 feature map에서 RoI를 추출하고 spatial pyramid pooling을 진행한다.
- Fast R-CNN은 RoI projection과 RoI pooling으로 구한다.
- Faster R-CNN은 Region proposal network를 anchor box를 이용해 end-to-end로 구현했다.

---

### 2. 과제 수행 과정 / 결과물 정리

서버 할당과 baseline code를 실행해 보았다.

---

### 3. 피어세션 정리

조원들과 이번주는 학습강의와 EDA에 집중하기로 했다.

또, 개인별로 하루에 1번씩 제출하고, 추가로 제출하려면 메시지를 남기기로 했다.

이후 EDA에 관한 이야기를 해보았고, 시도해 볼만한 것은 다음과 같다.

- 바운딩 박스의 center 좌표가 어떻게 분포하는지? (box의 가로세로ratio 분포, size 분포도 확인해보기)
- 클래스별로 center 좌표의 분포가 다른지?
- 실제로는 하나의 객체인데 색이나 재질로 구분되어 annotation이 두가지로 되어있는 경우는 없을까?
- 한 이미지 안에 클래스 몇개씩 or 객체 몇개씩 존재하는지 분포?
- 플라스틱은 주로 종이와 동시에 존재하는 등의 특징은 없을까?
  -  추후 confidence값이 낮은 경우, 이런 규칙을 활용할 수 있을 것 같음
- RGB값 분포
- 클래스마다 분포가 차이가 있는지
- 더러운 플라스틱은 일반쓰레기로도 분류하기도 하던데?
- general trash를 제외하고 모델 학습을 한 뒤 confidence가 낮은 결과에 대해서 general trash로 분류하는 방식도 가능할 것 같음.

---

### 4. 학습 회고

첫날인 만큼 먼저 데이터와 강의에 집중해서 진행해야겠다고 생각했다.

또, EDA로 의미있는 결과를 보았으면 좋겠다.



# 9 / 28 (화)

### 1. 강의 복습

- Object Detection에서 많이 이용되는 library는 MMDetection과 Detectron2이다.
- MMDetection은 PyTorch 기반의 Object Detection 오픈소스 library이다.
- Config 파일을 통해 데이터셋, 모델, 스케쥴러, optimizer등을 정의한다.
- Detectron2는 PyTorch 기반의 library로 Object Detection외에도 Segmentation과 Pose prediction도 제공한다.
- Neck은 마지막 feature map에서만 RoI를 추출하지 말고 중간 단계의 feature map에서 정보를 얻자는 생각에서 출발했다.
- Neck은 다양한 크기의 detection을 수행하기 위해 필요하다.
- 또, 하위 level과 상위 level의 정보를 합쳐서 사용한다.
- Feature Pyramid Network(FPN)는 Top-down path를 만들었다.
- 이 path에서 1x1 conv 연산과 Nearest neighbor upsampling이 일어난다.
- FPN의 문제점은 backbone이 너무 길어 정보전달이 어렵다는 점인데, Path Aggregation Network(PANet)은 이를 bottom-up path augmentation을 활용했다.
- DetectoRS는 RPN에서 영감을 받아 Recursive Feature Pyramid(RFP)를 제안했다.
- RFP는 FPN과 비슷하지만 top-down path에서 나온 feature map을 다시 backbone으로 넣는 방식으로, FLOPs가 많다.
- Bi-directional Feature Pyramid는 EfficientDet에서 제안된 방식으로 FPN처럼 단순 summation이 아닌 Weighted Feature Fusion을 이용했다.
- NASFPN은 사람의 heuristic한 방법으로 모델링하지 않고, Neural architecture search로 찾는 방법이다.
- 이 모델은 COCO dataset과 ResNet 기준으로 찾은 것이기 때문에 범용적이지 못하다.
- AugFPN은 high level에서의 정보손실에 주목해 Residual Feature Augmentation과 Soft RoI Selection을 제안했다.


---

### 2. 과제 수행 과정 / 결과물 정리

COCO format에 대해 찾아보던 중 fiftyone이라는 tool을 이용해 데이터를 한눈에 볼 수 있는 것을 확인했다.

원래 서버에 있는 데이터를 로컬 환경에서 볼 수 있는 튜토리얼대로 따라해 보았지만, 포트 관련해 어려움을 겪어 데이터를 로컬로 다운받아 진행했다.

그리고 간단한 EDA를 시도해 보았지만, 원하는 결과를 얻지 못했다.

---

### 3. 피어세션 정리

fiftyone tool을 피어분들께 소개했다.

또, 앞으로 프로젝트 진행을 할때 네이티브 코드를 이용할지, 라이브러리를 이용할지 고민했다.

베이스라인 코드를 실행하면서 생긴 오류에 관해서도 이야기했다.

---

### 4. 학습 회고

EDA에 관해 원하는 결과를 얻지 못해 아쉬웠다.

다른 EDA 방법을 고민해보고 의미있는 결과를 얻으면 좋겠다.




# 9 / 29 (수)

### 1. 강의 복습
- One stage detector는 Two stage detector가 속도가 느리다는 단점을 보완한 모델이다.
- localization과 classification이 동시에 진행되며 객체에 대한 맥락적 이해가 높은 특징이 있다.
- You Only Look Once(YOLO)는 region proposal이 없어 bbox 예측과 classification이 동시에 진행된다.
- feature map에 grid로 나누고 각각의 grid에 대해 bbox를 만들어 예측한다.
- loss로는 localization term(중심좌표와 높이, 너비), confidence term(object가 있거나 없거나), classification term이 있다.
- 하지만 YOLO는 grid보다 작은 물체를 찾을 수 없고 마지막 feature만 사용하는 단점이 있다.
- SSD는 서로다른 크기의 feature map을 모두 사용하고, fc layer 대신 conv layer를 사용하고 anchor box를 이용한다.
- YOLO v2는 기존에 비해 정확도, 속도, 많은 클래스를 예측하도록 개선되었다.
- YOLO v1에서 fc layer를 제거하고 bounding box 대신 anchor box로 바꾸고 offset 예측 문제로 바꾸어 성능을 향상시켰다.
- 또, 기존의 GoogleNet에서 DarkNet으로 바꾸고 Fine-grained feature와 Multi-scale training을 이용했다.
- YOLO v3는 skip-connection과 Multi-scale Feature map을 이용했다.
- RetinaNet은 One stage detector가 가진 문제점인 class imbalance 문제(객체 영역보다 배경 영역이 많은 것)를 Focal Loss를 도입해 다루었다.

---

### 2. 과제 수행 과정 / 결과물 정리

label의 분포, bbox의 위치와 크기등을 간단하게 EDA 해 보았다.

label의 경우에는 불균형이 있었고, bbox는 여러곳에 위치해 있지만, 비교적 가운데에 모여있었다.

bbox의 비율도 어느정도 경계가 있었고, 크기역시 주로 작은 크기를 가졌다.

---

### 3. 피어세션 정리

Train과 Valid set을 어떻게 나누어야 할지 이야기했다.

또, 모델 라이브러리가 어디까지 커스텀이 가능한지 알아보자고 했다.

---

### 4. 학습 회고

이번주는 EDA를 수행하느라 베이스라인 코드를 이해하지 못했는데, 다음주부터 실험을 하기 위해 베이스라인 코드를 이해하고 넘어가야겠다고 생각했다.
