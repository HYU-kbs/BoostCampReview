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