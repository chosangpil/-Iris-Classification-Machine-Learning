from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# K 폴드 교차 검증
# K개의 데이터 폴드 세트를 만들어 K번만큼 각 폴더 세트에 학습과 검증평가를 반복적으로 수행
# 과적합(Overfitting)방지, 별도의 여러 세트로 구성된 학습데이터 세트와 검증데이터 세트에서 평가 수행 - 데이터 편중을 막음

kfold = KFold(n_splits=5)   # 5개의 폴드 세트로 분리, KFold 객체
cv_accuracy = []    # 폴드 세트별 정확도를 담을 리스트
print('붓꽃 데이터 세트 크기 : ', features.shape[0])
n_iter = 0

# KFold 객체의 split() 호출 시 폴드별 학습용, 검증요 테스트의 로우 인덱스를 array로 변환
for train_index, test_index in kfold.split(features):
    # kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1
    
    # 반복마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'.format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스 " {1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)

# 개별 iteration별 정확도의 평균 정확도 계산
print('\n## 평균 검증 정확도 : ', np.mean(cv_accuracy))
