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
