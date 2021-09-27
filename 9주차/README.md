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
- 실제로는 하나의 객체인데 색이나 재질로 구분되어 annotation이 두가지로 되어있는 경우는 없을까?
- RGB값 분포
- 클래스마다 분포가 차이가 있는지
- 더러운 플라스틱은 일반쓰레기로도 분류하기도 하던데?
- general trash를 제외하고 모델 학습을 한 뒤 confidence가 낮은 결과에 대해서 general trash로 분류하는 방식도 가능할 것 같음.

---

### 4. 학습 회고

첫날인 만큼 먼저 데이터와 강의에 집중해서 진행해야겠다고 생각했다.

또, EDA로 의미있는 결과를 보았으면 좋겠다.