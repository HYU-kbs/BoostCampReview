# 8 / 23 (월)

### 1. 강의 복습

* 대회에서 Overview의 중요성 - Overview에는 방향성이 드러난다.
* Overview에서 풀어야 할 문제를 정의한다.
* Data Description은 다루는 Data를 소개하고 설명하는 요약본이다.
* Exploratory Data Analysis (EDA) - 데이터를 이해하려는 노력이다. 데이터에 대해 궁금한 것들을 알아가는 것이다.


---

### 2. 과제 수행 과정 / 결과물 정리

EDA를 통해 train data를 알아보았다.

직접 파일을 열어보기도 하고, U 스테이지에서 배운 시각화를 진행했다.

알아낸 사실은 train dataset에 여성이 더 많고, 나이의 분포가 불균형이 있다는 것이었다.

또, 나만의 dataset과 dataloader를 구현해보았다.

학습시키는 코드는 이전의 실습자료를 참고해서 구현해보았고, train을 시켜보았으나 어딘가에서 이상이 생겨 더 시도해볼것이다.

---

### 3. 피어세션 정리

데이터 전처리를 한다면 성능향상에 도움이 될 것 같다.

데이터셋의 확장자가 다양해 glob 모듈을 이용하면 편하다.

age를 categorical로 생각할 지, regression으로 생각할 지 이야기했다.

---

### 4. 학습 회고

코딩이 익숙하지 않아 시간을 많이 소모한 것 같다.

layer 변경을 하지않고 학습을 시켜보았는데, 성능이 낮았다.

model을 구성하는 layer를 바꾸어가며 모델을 학습시켜야겠다.

# 8 / 24 (화)

### 1. 강의 복습

* Data Pre-processing은 좋은 데이터를 만들기 위한 중요한 작업이다.
* 사진에서 필요없는 부분을 잘라내는 bounding box를 활용하자.
* 큰 사진보다 작은 사진으로도 충분히 좋은 성능이 나올 수 있다. (Resize 활용)
* 도메인, 데이터 형식에 따라 각기 다른 전처리 과정이 필요하다.
* Underfitting과 Overfitting의 중간에서 모델학습을 멈추자.
* Train, Validation, Test set으로 나누어 검증에 사용하자.
* 주어진 데이터가 가질 수 있는 여러가지 상황을 만드는 augmentation을 활용하자.
* 다만 이러한 기법들은 항상 좋은 결과를 내진 않을 수 있다.
* Data feeding에서 model의 처리량을 감당할 data generator를 고려해야한다.(병목현상)
* PyTorch의 custom dataset은 init, len, getitem을 구현해줘야한다.
* DataLoader는 dataset을 효율적으로 사용하기 위한 기능이 담긴 class이다.

* Seaborn
    - counterplot으로 개수를 세주는 bar plot을 만든다.
    - 데이터의 분포를 살피는 plot으로 box, violin, boxen, swarm, strip등이 있다.
    - histplot, kdeplot, ecdfplot, rugplot으로 분포를 살필 수 있다.
    - scatterplot, lineplot, regplot으로 관계를 살펴볼 수 있다.
    - heatmap으로 상관관계를 살펴볼 수 있다.


---

### 2. 과제 수행 과정 / 결과물 정리

* 수업에서 들은 내용을 바탕으로 여러가지를 적용해 봐야겠다고 생각했다.
    - bounding box를 활용하여 전처리를 해야겠다.
    - 또, 그에따른 RGB 채널의 변화를 살펴봐야겠다.
    
* K-fold validation을 이용해 validation set을 만들어야겠다.

* Data Transform을 진행할 때, 순서에따라 성능이 차이나기도 한다.

* 서로다른 task의 모델을 3개 만들어 진행하는 것은 어떨까 생각했다.
---

### 3. 피어세션 정리

다른 피어분들은 여러가지 pretrained된 모델을 이용해 학습을 진행하고 있었다.

dataloader를 사용할 때, ImageFolder를 이용할 수 있다.

이미지 normalization을 통해 학습을 더 빨리 진행할 수 있다.

---

### 4. 학습 회고

학습을 하는데 있어 데이터의 전처리가 매우 중요하다고 생각했다.

EDA 예제파일을 보면서 여러가지 방법을 떠올렸고, 잘못 labeling된 파일에 대해서 처리할 방법을 생각했다.

# 8 / 25 (수)

### 1. 강의 복습

* Model
  - 모델은 input을 원하는 output으로 출력하는 하나의 시스템을 말한다.(이미지 분류)
  - 커스텀 모델은 nn.Module을 상속받아 정의하고, init과 forward를 구현해준다.
  - PyTorch의 Pythonic한 장점때문에 형식과 구조를 알게되어 여러가지 응용이 가능하다.
  - ImageNet이라는 거대한 고품질 데이터셋의 발전으로 CV도 함께 발전했다.
  - 좋은 품질의 대용량의 데이터로 학습한 PreTrained 모델을 바탕으로 내 목적에 맞게 사용하자.
  - Transfer learning을 수행할 때, 현재 문제와의 유사성을 고려해서 어떤 부분을 freezing할지 선택하자.

---

### 2. 과제 수행 과정 / 결과물 정리

이미지 crop을 하기위해 face detect를 할 수 있는 cvlib의 detect_face를 이용했다.

하지만 cvlib이 haar cascade classifier보다 정밀했지만 문제가 있었다.

바로 normal의 얼굴은 잘 찾아내지만, 마스크 쓴 얼굴은 낮은 confidence로 덜 정확하게 찾아냈다.

그래서 낮은 confidence 중에서 bounding box가 중앙에서 크게 벗어나지 않은 것만 취했고,

결과가 좋은지 확인하기위해 visualization을 사용했다.


---

### 3. 피어세션 정리

이번에도 여러가지 고민거리들을 피어세션때 이야기했다.

서버가 꺼지지 않게 하려면 tmux나 nohup을 이용한다.

모델은 어느것을 pretraining으로 써야할지 조금더 시간이 필요하다.

모델에 집중해서 코딩할지, 전처리에 신경쓸지 고민했다.

---

### 4. 학습 회고

나는 전처리에 힘써서 EDA와 함께 진행했는데, 뚜렷한 성과가 없는것 같았다.

팀을 맺기 전에 모델을 만들고, 정확도를 확인해보고 싶다.