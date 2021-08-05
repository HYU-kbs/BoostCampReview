# 8 / 2 (월)

### 1. 강의 복습 

PreCourse에서 배운 내용이 대부분이어서 중요하거나 헷갈리다고 생각되는 부분을 복습했다.

* 파이썬 기본 자료구조
    - **Stack**
        - Last In First Out
        - 리스트의 append()와 pop()으로 구현 가능
    - **Queue**
        - First In First Out
        - 리스트의 append()와 pop(0)로 구현 가능
    - **Tuple**
        - 값의 변경이 불가능한 리스트
        - 실수로 인해 값이 변경되는 것을 방지
    - **Set**
        - 순서없이 값을 중복되지 않게 저장하는 자료구조
        - add(), remove(), 수학에서의 집합연산을 지원
    - **Dict**
        - Key 값을 활용하여 Value 값을 저장
    - **Deque**
        - Stack과 Queue를 지원하는 모듈
        - rotate나 reverse같은 linked list의 특성을 갖음
    - **Counter**
        - element의 갯수를 세어 dict 형태로 반환
        - Set 연산을 지원
    
* Pythonic code
    - **Generator**
        - iterable object를 특수한 형태로 사용해주는 함수
        - yield 키워드를 사용하거나 ()를 이용한 generator comprehension을 사용
        - 값이 사용될때만 호출이 되어 큰 데이터를 처리할 때 메모리 공간을 효율적으로 사용
    - **가변인자 (Variable-length)**
        - parameter로 *args 변수명을 사용
        - tuple로 인식  
        - ```python
          def summation(*args):
              return sum(args)
          ```
    - **키워드 가변인자 (Keyword variable-length)**
        - parameter로 **kwargs 변수명을 사용
        - dict로 인식
        - ```python
          def kwargs_test(one, two=3, *args, **kwargs):
              pass
          
          kwargs_test(10, 300, first=3, second=2, third=5)
          ```
    
* 벡터와 행렬
    - **벡터**
        - 공간에서의 한 점을 나타낸다
        - 원점으로부터 상대적 위치를 나타낸다
    - **벡터의 노름**
        - 원점에서부터의 거리를 나타내는 값
        - L1-노름은 각 성분의 변화량의 절대값을 더한 값
        - L2-노름은 유클리드 거리 값
    - **행렬**
        - 벡터를 원소로 가지는 2차원 배열
        - 공간에서의 여러 점을 나타낸다
        - 벡터 공간에서 사용되는 연산자로 이해할 수 있다 (벡터를 다른 차원으로 보낼 수 있다)
    - **역행렬**
        - 행렬의 연산을 되돌리는 행렬
        - 행과 열이 같고 determinant가 0이 아니어야 계산 가능하다
        - 역행렬을 계산할 수 없을 때 유사 역행렬인 **무어-펜로즈 역행렬**을 이용한다
   
* 경사하강법
    - **변수가 1개일 때**
        - 한 점에서의 미분값을 빼면서 함수의 극소값을 구할 수 있다
    - **변수가 여러개 일 때 (벡터)**
        - 편미분을 이용하여 gradient vector를 구한다
        - 한 점에서의 미분값을 빼면서 함수의 극소값을 구할 수 있다
    - **경사하강법을 이용한 선형회귀 분석**
        - 여러 점들을 행렬 X로 표현, coefficient vector를 b, 정답이 되는 벡터를 y라고 할때,
        - 선형회귀의 목적식은 y-Xb의 L2-노름이다
        - 목적식을 최소화하는 과정에서 b에 대해 편미분이 이용된다
        - 비선형회귀에서는 경사하강법 수렴이 보장되지 않는다
    
* 확률적 경사하강법
    - 모든 데이터가 아닌 일부만을 사용하여 업데이트한다
    - 연산량이 줄어드는 효과가 있다
    - 목적식이 계속해서 변화하므로 non-convex 계산식에도 적용가능하다
    
---
### 2. 과제 수행 과정 / 결과물 정리
처음으로 수행한 필수과제는 별로 어렵지 않았다

정규표현식 regex를 이용하면 더욱 깔끔한 코딩을 할 수 있을 것 같다

* Assignment1_Basic_Math
    - numpy를 import하여 각각 np.max, np.min, np.mean, np.median을 이용하였다
    
* Assignment2_Text_Processing_1
    - normalize 함수를 구현할 때 split()과 join()을 이용하였다
    - no_vowel_string 함수를 구현할 때 for loop를 이용하여 모음이 아닐 때만 return값에 더해주었다
    
* Assignment3_Text_Processing_II
    - digits_to_words 함수를 구현할 때, 숫자에 대응하는 영어 단어를 저장하는 list를 만들었다

---
### 3. 피어세션 정리

처음 한 피어세션에서 여러 캠퍼들을 만날 수 있었고, 앞으로 피어세션을 어떻게 진행할지 정하였다

모더레이터는 매일 돌아가며 맡기로 하였고, 피어세션을 진행하기로 하였다

피어세션에서 서로 강의중에 생긴 질문을 물어보며 답하는 시간과 코드 리뷰와 피드백을 진행하기로 하였다

---
### 4. 학습 회고

첫 날부터 많은일이 진행된것 같아 정신없었지만, 앞으로 많은 캠퍼들과 협업하여 프로젝트를 진행해보고 싶다

당장은 알고있는 내용 위주로 공부하고 있지만, 잘 정리해서 온전히 내 것으로 만들어야겠다고 생각했다


# 8 / 3 (화)

### 1. 강의 복습 

* Object Oriented Programming
    - OOP는 설계도인 class와 구현체인 instance로 나눌 수 있다.
    - 파이썬에서 class는 다음과 같이 정의한다
        ```python
      class SoccerPlayer():
            def __init__(self, name, position, back_number):
                self.name = name
                self.position = position
                self.back_number = back_number
        ```
    - OOP의 특성
        - **상속 (Inheritance)** - 부모 클래스로부터 속성과 method를 물려받는다.
        - **다형성 (Polymorphism)** - 같은 이름의 method를 다르게 작성할 수 있다.
        - **가시성 (Visibility)** - 객체의 정보를 볼 수 있는 레벨을 조정하는 것이다.
    
* Decorator
    - 함수의 parameter로 다른 함수를 쓸 수 있다.
    - 함수 내에서 다른 함수를 정의할 수 있다.
    - 함수 내에서 다른 함수를 반환하는 것을 클로저(closure)라고 하고, 이를 간단하게 사용하게 하려면 decorator를 이용한다.
    
* Module and Project
    - Module은 .py 파일이며 import문을 사용하여 호출한다.
    - 모듈의 특정 함수만 불러오려면 from ... import ... 와 같이 사용하거나 as로 별칭을 이용한다.
    - Package는 다양한 모듈의 모음이다.
    - 패키지를 구성할 때에는 \_\_init__.py를 만들고, 하위폴더와 py파일을 포함한다. 또, \_\_all__키워드를 사용한다.
    
* 딥러닝 학습방법
    - 분류 문제에선 모델의 출력을 확률로 변환하는 함수인 softmax함수가 이용된다.
    - 비선형 모델을 학습시키려면 활성화 함수가 필요하다. (sigmoid, tanh, ReLU)
    - 층을 여러개 쌓게되면 목적함수를 근사하는데 더 적은 뉴런만 학습시키면 된다.
    - chain-rule을 기반으로 back propagation이 이루어진다.
    
* 딥러닝과 확률론
    - 딥러닝은 확률론 기반의 기계학습 이론에 바탕을 두고 있다.
    - 목적식인 loss function들의 작동원리는 데이터 공간의 통계학적 해석으로 유도된다.
    - L2-norm은 예측오차의 분산을 최소화하는 방향으로 학습된다.
    - cross-entropy는 모델 예측의 불확실성을 최소화하는 방향으로 학습된다.
    - 확률분포를 알면 데이터를 알 수 있다.
    - 하지만 데이터가 주어지더라도 확률분포를 알 수 없기때문에 **몬테카를로 샘플링** 방법을 이용하여 기대값을 구한다.

---

### 2. 과제 수행 과정 / 결과물 정리

정규표현식 regex를 사용한다면 쉽게 text 전처리가 가능할 것 같았다.

하지만 regex가 익숙하지 않아 익숙한 방법으로 코딩을 진행해나갔다.

과제가 각각의 함수들로 잘 구분되어있어서 수행해나가는데 큰 어려움은 없었다.

* Assignment4_Baseball
    - 전체적으로 어렵진 않았지만 프로그램의 입출력 설명이 애매해 잘못 이해해서 많이 헤맸던 것 같다.

* Assignment5_Morsecode
    - dict를 이용하여 알파벳을 모스부호로 쉽게 바꿀 수 있었다.
    - 하지만 모스부호를 알파벳으로 바꿀 때에는 key와 value가 뒤바뀐 dict가 필요하여 dict comprehension을 이용하였다.
    - join과 split을 이용하여 문자열을 다루었다.

---

### 3. 피어세션 정리 
* 강의중에 배운 무어-펜로즈 역행렬의 이해방법과 유도방법을 이야기하였다.
    - 무어-펜로즈 역행렬은 SVD를 활용하여 유도할 수 있다.
    
* 강의중에 배운 경사 하강법 목적식에 대해 이야기하였다.
    
* 미니배치를 활용하여 학습할 때, 왜 연산량이 줄어들지에 대해 고민해보았다.
    - 빠른 GPU의 연산을 활용하여 시간을 줄인다고 생각했다.
    
* 멘토님과 5일 20시에 모임을 갖기로 하였다.

* 오늘 한 과제에 대해 막히는 부분에 대해 서로 질문했다.
    - 단순히 과제가 어렵기보단, 나처럼 입출력 설명을 잘못 이해했다.
    
* 매주 금요일에 AI 관련 기술, 이론, 논문 등을 발표하기로 하였다.
    - 앞으로 학습 난이도가 증가해서 시간이 부족해질 수도 있다.


---

### 4. 학습 회고

precourse에서 이해되지않았던 확률론 부분을 다시 들어보니 전보다 훨씬 이해되는 것 같았다.

과제 자체는 어렵지 않았지만 많이 헤매었던 것이 아쉬웠다.

피어세션에서 나도 모르고 지나갈 뻔한 질문들을 함께 고민하는 과정이 정말 좋았다.

# 8 / 4 (수)

### 1. 강의 복습

* Exception handling
    - 모든 잘못된 상황에서의 대처가 필요하다.
    - try ~ except문으로 사용한다.
    - 예외가 발생하지 않으면 else, 예외에 상관없이 수행하려면 finally 구문을 이용한다.
    - raise를 이용해 예외를 발생시킨다.
    - assert를 이용해 조건이 맞지 않으면 예외를 발생시킨다.
    
* File handling
    - open과 close를 이용하여 파일을 열고 닫을 수 있다.
    - with구문을 이용해도 파일을 읽고 쓸 수 있다.
    - pickle - 파이썬의 객체를 영속화(persistence)하는 객체로 실행중의 정보를 저장할 때 사용된다.
    - pickle.dump로 저장, pickle.load로 불러올 수 있다.
    
* Logging
    - logging 모듈을 import하여 사용한다.
    - 레벨에따라 debug, info, warning, error, critical로 나뉜다.
    - configparser - config file을 만들어 section, key, value 형태로 설정파일을 저장하여 dict 형태로 사용한다.
    - argparser - add_argument로 실행시에 argument를 설정할 수 있다.
    
* Data handling
    - **CSV(Comma Separate Values)**
        - 엑셀 양식의 데이터로 프로그램에 상관없이 쓸 수 있다.
        - csv 모듈의 reader를 이용하면 쉽게 데이터를 읽을 수 있다.
    - **HTML(Hyper Text Markup Language)**
        - 웹 상의 정보를 구조적으로 표현하기 위한 언어이다.
        - 트리 구조로 <>를 이용하여 나타낸다.
    - **정규식(Regular Expression)**
        - HTML의 표현식에서 원하는 정보만 추출할 수 있다.
        - re 모듈을 import하여 re.search, re.findall로 찾아낼 수 있다.
    - **XML(eXtensible Markup Language)**
        - 데이터의 구조와 의미를 설명하는 tag를 사용하여 표시하는 언어이다.
        - HTML과 구조가 비슷하다.
        - Markup 언어를 parsing하는 BeatifulSoup이라는 모듈이 있다.
    - **JSON(JavaScript Object Notation)**
        - JavaScript의 객체 표현 방식언어다.
        - 간결하고 용량이 적고 코드로의 전환이 쉬워 XML의 대체제로 활용된다.
        - json모듈을 호출하고 dump로 저장, loads로 불러와 사용할 수 있다.
        
* 통계학과 머신러닝
    - 데이터가 주어지더라도 모집단의 확률분포를 알 수 없기 때문에 모델로 추정하여 그 불확실성을 최소화한다.
    - 모델을 결정하는 모수에는 평균과 분산이 있다.
    - 모집단이 정규분포를 따르지 않더라도 많은 표본의 평균은 정규분포를 따르게된다.
    - 주어진 데이터에서 가장 가능성이 높은 모수를 추정하는 방법을 **Maximum Likelihood Estimation**이라 한다.
    - 주어진 확률분포 P, Q의 거리를 계산할 때, **Kullback-Leibler Divergence** 방법을 이용한다.
    
* 베이즈 정리
    - 어떤 병의 발병률이 알려져있다 : prior(사전확률)
    - 실제 걸렸을 때 검진될 확률, 실제 걸리지 않았을 때 오검진될 확률 : likelihood(가능도)
    - 질병에 걸렸다고 검진되었을 때, 실제로 걸렸을 확률은 우리가 구하고자하는 posterior(사후확률)이다.
    - True Positive, False Negative(1종 오류), True Negative, False Negative(2종 오류)가 있다.
    - 새로운 데이터가 들어왔을 때 먼저 계산한 사후확률을 사전확률로 사용하여 갱신된 사후확률을 구할 수 있다.
    - 조건부 확률로 인과관계를 해석하는 것은 위험하며 인과관계를 알아내기 위해서는 중첩요인을 제거해야한다.
    

---
### 2. 과제 수행 과정 / 결과물 정리

선택과제 1, 3번을 하려 했으나 생각보다 오래걸려 1번만 수행했다.

* Assignment1__Gradient_Descent
    - 강의에서 볼때와 직접 코딩하는 것은 정말 많이 다르다고 느꼈다.
    - 강의노트와 검색을 병행하여 풀기도 하였고, 손실함수를 직접 w와 b에 대해 미분하여 선형회귀분석을 경사하강법으로 해결했다.
    - 그래프를 그려 일반 경사하강법과 확률적 경사하강법이 어떤식으로 손실이 줄어드는지 눈으로 확인할 수 있어 좋았다.
    - 확률적 경사하강법을 구현할 때, mini-batch를 사용하는 부분에서 내가 코딩한 방법이 옳은건지 의심이 들었다.
    
---

### 3. 피어세션 정리

* 어제 배운 경사하강법인 SGD에 대해 개념을 정리해보았다.
    - 한번의 iteration에서 가중치 w가 학습된다.
    - mini-batch size는 주로 2의 지수의 크기만큼 설정한다.
    - 모든 데이터셋이 학습에 참여되면 한번의 epoch이라고 한다.
    
* 활성화함수 sigmoid와 ReLU를 비교했다.
    - sigmoid는 신경망이 깊어질수록 gradient vanishing이 일어나 학습반영이 잘 안될 수 있다.
    - 반면 ReLU는 미분값이 1이어서 계산도 쉽고 신경망이 깊어지더라도 학습반영이 잘 된다.
    
* 내일 있을 멘토분과의 만남에서 미리 질문을 생각했다.

---

### 4. 학습 회고

피어세션때 모더레이터분이 질문에 대한 답을 발표처럼 진행했는데, 굉장히 준비를 많이 하신 것 같아 나에게 자극이 되었다.

또, 선택학습이 난이도가 있어 고민하고 검색하는데 시간을 많이 쓴 것같다.

강의의 내용도 점점 어려워지고 있어 걱정이되지만 다른분들과 이야기하며 해결해 나갈 것이다.

# 8 / 5 (목)

### 1. 강의 복습

* Numpy(Numerical Python)
    - 파이썬의 고성능 과학 계산용 패키지이다.
    - 일반 list보다 빠르고 메모리 효율적이며 반복문없이 배열을 처리할 수 있다.
    - numpy의 array는 ndarray 타입으로 한가지 type만 들어갈 수 있다. (Dynamic typing not supported)
    
* ndarray
    - array의 데이터 타입인 **dtype**과 dimension인 **shape**가 있다.
    - 차원의 개수인 **ndim**과 element의 개수인 **size**, array의 byte크기인 **nbytes**가 있다.
      
* Handling shape
    - array의 shape을 변경하는 **reshape**가 있다.
    - **flatten**은 다차원의 array를 1차원으로 바꿔준다.
    ```python
    # a의 shape은 (2, 4)
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
  
    # -1은 size를 기반으로 저절로 계산해준다. b의 shape은 (4, 2)
    b = np.array(a).reshape(-1, 2)
    ```

* Indexing과 Slicing
    - numpy의 **indexing**은 a[0][0]과 a[0, 0] 모두 지원한다.
    - numpy는 **slicing**을 지원한다.
    ```python
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # a[:2,:2], a[:, 3], a[:,::2]과 같은 연산이 가능하다.
    ```
  
* Array creation
    - **arange**는 array의 범위를 설정하여 array를 만들어준다.
    - **zeros**와 **ones**는 각각 0과 1로 가득찬 array를 만들어준다.
    - **empty**는 비어있는 array를 만들어준다. (memory initialization이 안 되어있음)
    - **zeros_like**와 **ones_like**, **empty_like**는 기존의 array와 같은크기를 가지면서 각각 0과 1, 비어있는 값으로 가득찬 array를 만들어준다.
    ```python
    a = np.arange(0, 10, 0.5)
    b = np.zeros(shape=(2, 2))
    c = np.ones_like(b)
    ```
    - **identity**는 단위행렬을 만들어주고, **eye**는 대각선이 1인 행렬을 만들어준다.
    - **diag**는 행렬의 대각성분만 추출할 때 사용된다.
    ```python
    a = np.identity(3)
    b = np.eye(N=3, M=5, k=1) # k는 시작점
    c = np.diag(b, k=1)
    ```
    - **random.uniform**과 **normal, binomial, exponential**은 각각의 확률분포에서 sampling을 해준다.
    
* Operation function
    - **axis**는 operation을 수행할 때 기준이되는 dimension 축이다.
    - 새롭게 추가된 dimension에 axis 0번을 부여한다. **np.newaxis**를 이용해 새로운 축을 추가할 수 있다.
    - **sum**은 ndarray의 합을 구해준다. 이때 axis별로 계산을 다르게 할 수 있다.
    - 그외에도 **mean**, **std** 등이 있다.
    - **concatenate** array를 서로 붙이는 함수로 위아래로 붙이는 **vstack**과 좌우로 붙이는 **hstack**이 있다.
    - **concatenate**의 axis는 붙이고 난 뒤 생기는 axis의 결과라고 생각하면 이해하기 편하다.
    
* Array operation
    - array끼리 shape이 같을 때 element끼리 +, -, *가 가능하다. (Element-wise operation)
    - **np.dot**을 이용해 행렬끼리 곱셈, **.T**또는 **transpose**를 이용해 transpose가 가능하다.
    - array와 scalar, 또는 shape이 다른 array연산에서 **broadcasting**이 일어난다.
    - jupyter에서 **%timeit**을 이용하여 성능을 비교할 수 있다.
    
* Comparison
    - **all**과 **any**는 boolean array에 대해 연산을 지원한다.
    - shape이 같은 array끼리의 비교도 broadcasting이 일어난다.
    - **logical_and, or, not**은 boolean array에 대해 연산을 지원한다.
    - **where**(boolean_array, True_value, False_value)로 boolean_array에 따라 값을 다르게 넣을 수 있다.
    - **where**(boolean_array)는 True인 index array를 Tuple로 반환한다.
    - **isnan, isfinite**로 숫자인지, 유한한지 확인할 수 있다.
    - **argmax, argmin**으로 axis를 활용하여 최대최소인 index를 알 수 있다.
    - **argsort**는 작은값을 index로 반환한다.
    
* Boolean index와 Fancy index
    - **boolean index**는 a[a > 3]과 같이 사용한다. 이때, shape이 같아야한다.
    - **fancy index**는 a[b]와 같이 사용한다. 이때, b는 int자료형의 index array이다.
    - matrix array에 대해서도 a[b, c]와 같이 사용한다. 이때, b, c는 int자료형의 index array이다.

* Numpy I/O
    - **loadtxt, savetxt**로 저장과 불러오기가 가능하다.
    - **load, save**는 pickle형태로 저장되며 .npy로 저장된다.
    
* Convolutional Neural Network (CNN)
    - MLP에서는 입력벡터 x가 가중치 행렬 W와 곱해지는 형태였다.
    - 하지만 CNN에서는 커널을 입력벡터상에서 움직이면서 곱해지는 형태이다.
    - 수학적인 의미로 보면 신호(signal)를 커널을 이용해 증폭, 감소시켜서 정보를 추출하는 의미이다.
    - CNN은 입력 신호에따라 다양한 차원을 갖게된다.
    - 입력신호의 크기를 (H, W), 커널의 크기 (K_H, K_W), 출력 크기를 (O_H, O_W) 라고 할때, 이들 사이에는 다음과 같은 식을 만족한다.
    
            O_H = H - K_H + 1
            O_W = W - K_W + 1
    - CNN의 back propagation 역시 CNN의 연산과 동일하게 gradient를 곱해준다.

---

### 2. 과제 수행 과정 / 결과물 정리

* Assignment3_Maximum_Likelihood_Estimation(MLE)

수식을 유도하는 것은 강의의 내용과 다르지 않아 어렵지 않았다.

하지만 pyplot을 써본 적이 없어 pyplot을 구현하지 못했다.

그리고 처음으로 수식을 마크다운언어로 작성했다. 

솔루션이 나오고나서 비교해봐야겠다고 생각했다.

---

### 3. 피어세션 정리

서로 과제를 수행하면서 생긴 질문들을 주로 이야기했다.

- mini-batch를 이용하여 데이터를 추출할 때, 중복되지 않게 뽑기위해 비복원추출을 이용한다.

- RNN에서 시퀀스가 너무 길면 적당히 잘라서 back-propagation한다.

- 미분식을 코딩으로 구현할 때, 직접 미분하여 사용하는 것과 근사해 사용하는 것의 차이에 대해 이야기했다.

---

### 4. 학습 회고

피어세션이 서로 질문을 답해주는것만으로도 시간이 모자르다고 생각했다.

그리고 내가 쉽게 넘어갔던것도 더 자세히 질문하고 배워가는 과정이 정말 좋다고 생각했다.

마스터클래스에서 교수님이 저희의 궁금한점들을 너무 잘 답해주신것 같아 1시간이 짧다고 느꼈다.

앞으로 공부해나가야 할 길을 알게 된 것 같다.