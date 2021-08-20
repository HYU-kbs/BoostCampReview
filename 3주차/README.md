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

# 8 / 19 (목)

### 1. 강의 복습

* 학습 모델을 저장하려면 torch.save()를 이용한다. 모델의 형태와 파라미터를 저장한다.
* EarlyStopping 기법을 사용하려면 중간 결과를 저장하는 checkpoint가 필요하다.
* Transfer Learning은 이미 대용량의 데이터셋으로 만들어진 모델을 현재의 데이터셋에 적용하는 것이다.
* Freezing을 이용해 pretrained model의 parameter를 학습시키지 않을 수 있다.
* 학습 중간에 monitoring하는 도구중에는 Tensorboard와 Weight&Biases가 있다.


* Data_Visualization - Color
    - 위치와 색은 가장 효과적으로 구분되는 속성이다.
    - 하지만 화려함은 시각화의 일부분으로 전하는 메세지가 드러나야한다.
    - 독립된 색상으로 구성된 컬러맵은 범주형 변수에 사용하기 적합하다.
    - 연속된 색상으로 구성된 컬러맵은 연속형 변수에 사용하기 적합하다.
    - 발산형 색상으로 구성된 컬러맵은 상반된 값을 표현하기에 적합하다.
    - 강조를 위해 명도, 색상, 채도, 보색대비를 이용한다.
    - 색각 이상을 고려한 컬러맵을 사용한다.

* Data_Visualization - Facet
    - 화면을 분할해 여러 관점으로 데이터셋을 보여준다.
    - matplotlib에서는 figure는 1개, Ax는 여러 개 사용한다.

---

### 2. 과제 수행 과정 / 결과물 정리

Dataset을 스스로 만들고, DataLoader를 스스로 만드는 방법을 과제를 수행하면서 감을 익힌 것 같다.

하지만 사용한 Dataset중 AG News는 이해하기가 어려웠다.


---

### 3. 피어세션 정리

과제에 대한 질문들을 주로 이야기했다.

과제의 양이 많아서 많은 질문들이 있었고, 혼자서 해결하지 못한 문제들을 다같이 모여 이야기해서 해결할 수 있었다.

zero padding을 하기 위해 tensor에 concat이나 hstack, functional.pad를 이용할 수 있었다.

또, 데이터셋에서 구현하는 getitem 함수는 train, test 여부에따라 label이 주어져있지 않을 수 있기 때문에 X만 return하도록 구현하였다.


---

### 4. 학습 회고

필수과제가 상당히 까다로워 선택과제를 전혀 하지 못했다.

하지만 과제 하나하나를 성실히 하면서 PyTorch를 배워가는 것 같아 다음엔 더 능숙하게 사용했으면 좋겠다.

# 8 / 20 (금)

### 1. 강의 복습

* Multi-GPU
    - Node는 하나의 시스템을 말한다.
    - Model이나 Dataset을 parallel하게 나누어 학습을 분산시킬 수 있다.
    - Model parallel에서는 pipeline이 되도록 하는 것이 중요하다.
    - Dataset parallel은 mini-batch와 비슷하지만, 한번에 여러 GPU에서 수행한다.
    - 한 GPU에만 연산 불균형이 일어날 수 있다. 이때 DistributedDataParallel을 이용한다.
    
* HyperParameter Tuning
    - learning rate, 모델의 크기, optimizer는 학습하지 않는 값으로 사람이 지정해 주어야한다.
    - grid search는 일정한 간격으로 찾는 반면, random search는 그렇지 않다.
    - 최근에는 베이지안 기반의 기법들이 많이 이용된다.
    - Ray는 hyperparamter tuning을 위한 많은 모듈을 제공한다.
    
* PyTorch TroubleShooting
    - GPUUtil을 이용해 GPU의 상태를 확인할 수 있다.
    - torch.cuda.empty_cache()를 사용해 cache를 정리할 수 있다.
    - loop에 tensor로 축적되는 변수를 확인한다.
    - del 명령어를 이용해 필요없어진 변수는 삭제해준다.
    - batch 사이즈를 줄여 실행해본다.
    - Inference 시점에는 torch.no_grad()를 사용한다.
    
* Data Visualization - 그외의 팁들
    - Grid는 무채색으로, 맨 밑에 오도록 조정된다.
    - 또, major, minor, x축, y축 등을 선택할 수 있다.
    - X + Y = C의 형태는 Feature의 절대적 합이 중요한 경우 사용된다.
    - Y = CX의 형태는 Feature의 비율이 중요한 경우 사용된다.
    - 동심원의 형태는 특정 지점에서 거리를 살펴볼 경우 사용된다.
    - 선, 면을 추가하여 가독성을 높일 수 있다.
    - 테마를 변경하여 한번에 변경할 수 있다.

---

### 2. 과제 수행 과정 / 결과물 정리

드디어 선택과제를 읽어보면서 과제를 진행했다.

Pre-trained된 모델인 ResNet을 이용해 Mnist와 FashionMnist를 학습시켰다.

맨 앞쪽 layer와 맨 뒷쪽 layer의 input_feature과 output_feature를 수정해도 꽤 결과가 잘나오는 것을 확인했다.

다만, 시간이 부족해 다른 모델을 사용했을 때 성능이 좋아지는지에 대해 더 알아보지 못해서 아쉬웠다.

---

### 3. 피어세션 정리

랜덤 피어세션 이후 피어세션에서는 팀 회고록을 작성해 각자 발표하는 시간을 가졌다.

또, 다음주에는 ViT논문을 가지고 리뷰를 진행할 예정이다.


---

### 4. 학습 회고

이번주는 PyTorch위주로 학습을 해서 나의 부족한 점을 잘 알게된 한 주였다.

다음주에 있을 대회에 앞서 복습과 PyTorch코드도 연습해야겠다고 생각했다. 