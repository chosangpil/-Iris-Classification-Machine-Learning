from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# Stratified K 폴드
# 불균형한 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K 폴드 방식
# Stratified K 폴드는 K 폴드가 레이블 데이터 집합이 원본 데이터 집합의 레이블 분포를
# 학습 및 테스트 세트에 제대로 분배하지 못하는 경우의 문제를 해결해줌

iris = load_iris()
features = iris.data
label = iris.target
df_clf = DecisionTreeClassifier(random_state=156)
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df['label'].value_counts()

print(iris_df)
print(iris_df['label'].value_counts())

kfold = KFold(n_splits=3)
n_iter = 0

# 각 교차 검증 시마다 생성되는 학습/검증 레이블 데이터 값의 분포도 확인
for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(' ## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print('검증 레이블 데이터 분포 : \n', label_test.value_counts())
    # 교차 검증시마다 3개의 폴드 세트로 만들어지는 학습레이블과 검증 레이블이 완전히 다른 값으로 추출 되는것을 확인
    # ex) 첫번째 교차 검증시 학습레이블의 1, 2 값이 각각 50개 추출, 검증 레이블 값 0이 50개 추출
    # ==> 학습 모델은 0의 경우를 학습하지 못한다 따라서 검증 레이블은 0밖에 없으므로 학습모델은 0을 예측하지 못함

# KFold로 분할된 레이블 데이터 세트가 전체 레이블 값의 분포도를 반영하지 못하는 문제

# 동일한 데이터 분할을 위해 데이터 분할을 StratifiedKFold로 수행
skfold = StratifiedKFold(n_splits=3)
n_iter = 0

for train_index, test_index in skfold.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(' ## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print('검증 레이블 데이터 분포 : \n', label_test.value_counts())
    # 학습 레이블과 검증 레이블 데이터 값의 분포도가 거의 동일하게 할당되는 것을 확인
    
n_iter = 0
cv_accuracy = []
   
# StratifiedKFold의 split() 호출 시 반드시 레이블 데이터 세트도 추가 입력 필요 
for train_index, test_index in skfold.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # 학습 및 예측
    df_clf.fit(X_train, y_train)
    pred = df_clf.predict(X_test)
    
    # 반복 마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'.format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스 : {1}'.format(n_iter, test_index))

# 교차 검증별 정확도 및 평균 정확도 계산
print('\n ## 교차 검증별 정확도 : ', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도 : ', np.round(np.mean(cv_accuracy), 4))
