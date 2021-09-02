# 8 / 30 (월)

### 1. 강의 복습

없음

---

### 2. 과제 수행 과정 / 결과물 정리

baseline 코드를 수정해보았다.

먼저 custom model을 ResNext50으로 바꾸고, 마지막 classification layer을 18개의 class가 나오도록 수정했다.

또, train data의 bbox를 미리 계산한 csv파일을 이용해 crop을 수행했다.

그리고 test data의 bbox는 이미지가 들어오면 계산해 crop에 이용했다.

마지막으로 f1 score를 validation set으로 계산한 뒤 best model 판단에 사용했다.

---

### 3. 피어세션 정리

각자 주말동안 학습한 내용을 공유하며 모델의 성능을 높일 방법을 고민했다.

특히 앞으로 multiple label classification과 mixup, oversampling을 구현해야겠다고 생각했다.


---

### 4. 학습 회고

오늘 스페셜미션에서 나온 multiple class classification 방식에 대해 수정할 필요를 느꼈다.

그리고 이를 성별/나이/마스크를 구분하는 3개의 다른 classifier를 구현해야겠다고 생각했다.


# 8 / 31 (화)

### 1. 강의 복습

없음

---

### 2. 과제 수행 과정 / 결과물 정리

Pretrained model로 resnext를 쓰고, 마스크/성별/나이 분류를 하는 classifier를 3개 달았다.

그리고 bbox를 facenet을 이용한 것으로 바꾸어야겠다고 생각했다.

label을 smoothing하게 하고 mixup을 활용하기 위해선 label의 출력을 바꾸어줄 필요를 느꼈다.

---

### 3. 피어세션 정리

각자 구현한 부분의 코드를 보며 리뷰를 진행했다.

oversampling을 위해 클래스의 비율에따라 늘려주는 방법을 이용해야겠다.

---

### 4. 학습 회고

해야할 기능들은 많지만, 코드로 구현하는 것이 쉽지않았다.

코딩실력에 대한 부족함을 많이 느꼈다.

# 9 / 1 (수)

### 1. 강의 복습

없음

---

### 2. 과제 수행 과정 / 결과물 정리

bbox를 facenet으로 전처리한 csv파일을 만들었다. 그리고 train model에서도 facenet으로 바꾸었다.

출력 label을 one-hot encoding이 아닌 다른 벡터로 바꾸어주었고, 그에따른 loss도 바꾸어주었다.

하지만 class imbalance 문제는 해결되지 않아 학습 성능이 오히려 더 떨어졌다.

---

### 3. 피어세션 정리

focal loss에서 weight의 역할은 적은 클래스에 대해서는 가중치를 높게주어 loss를 크게 만드는 역할을 한다.

cutmix를 활용한 학습도 진행해 보았다.

loss의 backward에서 retain_graph에 대해 이야기했다.

---

### 4. 학습 회고

class imbalance에 대한 문제를 focal loss로만 해결하는 것은 별로 좋지 않은 것 같다.

oversampling을 구현해 imbalance 문제를 완화해야겠다고 생각했다.

