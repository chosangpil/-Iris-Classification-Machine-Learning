from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# GridSearchCV - 교차검증 및 최적 하이퍼파라미터 튜닝
# GridSearchCV API : Classifier나 Regressor와 같은 알고리즘에 사용되는 하이퍼파라미터를 순차적으로 입력하며 편리하게 최적의 파라미터 도출
# 교차 검증을 기반으로 하이퍼 파라미터의 최적값을 찾음

# 주요 파라미터 : estimator, param_grid, scoring, cv, refit


# 데이터 로딩 및 학습데이터와 테스트데이터 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)

dtree = DecisionTreeClassifier()

# 파라미터를 딕셔너리 형태로 설정
parameters = {'max_depth':[1, 2, 3], 'min_samples_split':[2, 3]}

# param_grid의 하이퍼 파라미터를 3개의 train, test set fold로 나누어 테스트 수행 설정
# 순차적으로 6회에 걸쳐 하이퍼 파라미터를 변경하며 교차검증 데이터 세트에 수행성능 측정
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)   # cv 3회 X 6개 파라미터 조합

# param_grid의 하이퍼 파라미터를 순차적으로 학습/평가
grid_dtree.fit(X_train, y_train)

# GridSearchCV 객체의 fit 메서드 수행 후 결과는 cv_result_ 속성에 기록됨, 결과를 추출해 DataFrame으로 변환하여 확인
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']])

print('\nGridSearchCV 최적 파라미터 : ', grid_dtree.best_params_)
print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_dtree.best_score_))

# refit=True 이면 GridSearchCV가 최적 성능을 나타내는 하이퍼 파라미터로 Estimator를 학습해 best_estimator_로 저장함
estimator = grid_dtree.best_estimator_  # GridSearchCV의 refit으로 이미 학습된 estimator 반환

# GridSearchCV의 best_estimator_는 이미 최적학습이 완료된 상태, 별도 학습 필요없음
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도 : {0:.4f}'.format(accuracy_score(y_test, pred)))
