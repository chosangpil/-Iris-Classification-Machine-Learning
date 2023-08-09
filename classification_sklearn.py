from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 붓꽃 데이터 세트 로딩
iris = load_iris()
iris_data = iris.data   # iris.data는 Iris 데이터 세트에서 feature만으로 된 데이터를 numpy로 가짐
iris_label = iris.target    # iris.target은 붓꽃 데이터 세트에서 레이블(결정값) 데이터를 numpy로 가짐

print(iris_label)
print(iris.target_names)    # 레이블은 0, 1, 2 세가지 값, 0은 setosa, 1은 versicolor, 2는 virginica 품종을 의미

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target

print(iris_df.head(3))  # 붓꽃 데이터 세트를 DataFrame으로 변환하여 확인

# 학습 데이터와 테스트 데이터를 test_size 파라미터 입력값의 비율로 분할 (학습 데이터 80%, 테스트 데이터 20%)
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
dt_clf.fit(X_train, y_train)

# 테스트 데이터 세트로 예측 수행
pred = dt_clf.predict(X_test)

print('예측 정확도 : {0:.4f}'.format(accuracy_score(y_test, pred)))
