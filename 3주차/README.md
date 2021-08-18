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