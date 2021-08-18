# 8 / 17 (화)

### 1. 강의 복습

* Tensor
    - Array를 표현하는 PyTorch의 클래스다.
    - numpy의 ndarray와 동일하다.
    - numpy의 대부분의 사용법과 같다.
    - device를 cpu와 gpu중 선택해서 사용할 수 있다.
    - view는 copy가 일어나지 않고, reshape은 copy가 일어난다.
    
* Squeeze와 Unsqueeze
    - Squeeze는 1인 차원을 없애준다.
    - Unsqueeze는 axis를 사용해 1인 차원을 만들어준다.
    
* Tensor의 operation도 numpy와 동일하다.
* 벡터의 내적에 dot을 사용하고, 행렬의 곱셈에는 mm과 matmmul을 사용한다.
* mm은 broadcasting이 일어나지 않고, matmul은 일어난다.
* nn.functional에는 softmax나 onehot 등이 있다.
* backward를 통해 자동미분이 가능하다.

* PyTorch Template을 이용해 프로젝트의 구성을 쉽게 할 수 있다.
    - train.py는 각각의 argument로 지정이 가능하다.
        - config.json에는 hyperparameter의 값을 미리 지정해놓았다.
    - test.py는 학습 후 성능을 테스트할 때 사용한다.

---

### 2. 과제 수행 과정 / 결과물 정리

torch의 기본 문법을 활용한 과제로 양이 생각보다 많아 다하지 못했다.

특히 gather함수의 사용법이 많이 헷갈렸다.


---

### 3. 피어세션 정리

다른 피어분들도 gather의 사용법이 익숙하지 않아 어려움을 겪었다.

또, 다른 피어분의 발표로 git의 사용법도 다시한번 익혔다.

---

### 4. 학습 회고

예상대로 torch 코딩을 하는데 어려움을 겪었다.

그리고 template 강의가 인상깊었다.

각각의 파일들을 더 자세히 살펴봐야겠다고 생각했다.

# 8 / 18 (수)

### 1. 강의 복습

* torch.nn.Module에는 각각의 layer의 base class가 있다.
    - input, output, forward, backward, parameter를 정의한다.
    
* torch.nn.Parameter는 nn.Module내의 attribute로 하게된다면 required_grad=True로 해준다.

* forward는 예측값을 만들어주고, backward는 loss를 미분하여 parameter를 업데이트한다.

* Dataset과 DataLoader
    - Dataset class는 init, len, getitem등이 있다.
    - transforms는 데이터를 전처리하거나 augmentation 할 때 사용된다.
    - DataLoader는 batch를 만들거나 shuffle하는 기능이 있다.
    
* 데이터 형태에 따라 dataset class를 다르게 정의한다.
    - 이미지의 Tensor로의 변환은 학습에 필요한 시점에 변환한다.
    
* DataLoader parameter중 collate_fn은 variable 정의나 padding에 이용된다.

* Data Visualization - Text
    - 부족한 설명을 보충하는 역할을 한다. 또, 오해를 방지할 수 있다.
    - 과도한 text 사용은 이해를 방해할 수 있다.
    - Title은 가장 큰 주제를 설명하는 제목이다.
    - Label은 축에 대한 정보를 제공한다.
    - Tick label은 축에 눈금을 사용하여 정보를 제공한다.
    - Legend는 서로다른 데이터를 구분하기 위해 사용된다.
    - Annotation은 그 외의 설명을 추가할 때 사용된다.
    

---

### 2. 과제 수행 과정 / 결과물 정리

어제에 이어서 오늘도 Custom Model 과제를 수행했다.

torch의 hook의 개념적인 이해는 되었지만, 코드로는 어떤식으로 짜야할 지 헷갈렸다.

또, class의 메소드를 partial을 이용해 구현할 수 있다는 것을 알게되었다.


---

### 3. 피어세션 정리

과제가 분량이 많아 대부분 과제에 관한 질문들로 진행되었다.

hook은 CNN 모델에서 중간의 output을 확인할 때 사용되기도 한다.

또, torch.swapdim은 차원을 바꾸어주는 함수이다.

---

### 4. 학습 회고

과제가 상당히 양도 많고 시간도 오래걸렸지만, 차근차근 하면 할 수 있는 문제들이었다.

점점 pytorch에 익숙해져가는 느낌이 들었다.