#
# 마크다운은 그냥 여기서 부터 마크다운이고 여기서 끝난다
"""
🗂 데이터셋: Boston Housing Dataset
설명: 지역별 범죄율, 방 개수, 교통 접근성 등을 이용해 주택 가격을 예측하는 회귀 문제 분석 과정

## 1. 데이터 전처리 (Data Preprocessing)
### 1.1 결측치 처리 (Missing Value Handling)
- Boston Housing 데이터셋은 전통적으로 결측치가 없으나, 실제 운영환경에서는 누락된 값이 발생할 수 있습니다.
- **이유**: 결측치는 모델 학습 시 정보 손실을 야기하고 예측 정확도를 저하시킬 수 있습니다.
- **방법**:
   1. 결측치 비율이 극히 낮으면 해당 레코드 삭제
   2. 비율이 높거나 중요한 변수인 경우, 평균/중앙값 대체, 회귀 기반 대체, KNN 대체 등을 활용
- **다음 단계**: 결측치 처리 직후 데이터 분포와 모델 성능 변화를 시각화하여 적절성을 평가합니다.

### 1.2 이상치 탐지 및 제거 (Outlier Detection and Removal)
- 회귀 계수에 큰 영향을 주는 이상치는 모델의 일반화 성능을 저해할 수 있습니다.
- **주요 방법**:
   1. IQR 기반 Tukey 방법: Q1 - 1.5*IQR, Q3 + 1.5*IQR 범위를 벗어난 값 제거
   2. Z-score 기반: 절대값이 특정 임계치(예: 3)를 초과하는 값 검출
   3. 시각화(Boxplot, Scatter)로 잠재적 이상치 탐색
- **실습**: IQR 방법으로 이상치를 제거한 뒤, 제거 전/후 모델 성능을 비교하세요.
- **다음 단계**: 이상치 제거 후 변수 간 상관관계와 분포 변화를 검토합니다.

## 마크다운은 그냥 여기서 끝난다
"""
#############################
# 3. 환경 설정
"""  
## 3.1 라이브러리 불러오기
- 분석에 필요한 기본 라이브러리를 불러옵니다.
"""
#############################
# 3 Set-up
## 3. 1 라이브러리 불러오기
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from collections import Counter
## 3.2 회귀 지표 함수 정의
def Reg_Models_Evaluation_Metrics (model,X_train,y_train,X_test,y_test,y_pred):
    cv_score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
    
    # Adjusted R-squared 계산
    r2 = model.score(X_test, y_test)
    # 관측치 수
    n = X_test.shape[0]
    # 특성(독립변수) 수
    p = X_test.shape[1]
    # Adjusted R-squared 공식
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    R2 = model.score(X_test, y_test)
    CV_R2 = cv_score.mean()

    return R2, adjusted_r2, CV_R2, RMSE
## 3.3 데이터셋 특성

### Boston House Prices

https://www.kaggle.com/datasets/vikrishnan/boston-house-prices

각 레코드는 보스턴 교외 또는 도시를 설명합니다. 데이터는 1970년 보스턴 표준 대도시 통계 지역(SMSA)에서 수집되었습니다. 특성들은 다음과 같습니다 (UCI ML Repository 참고):

  - CRIM     도시별 1인당 범죄율
  - ZN       25,000 평방피트 이상 주거용 토지 비율
  - INDUS    도시별 비소매업 비즈니스 면적 비율
  - CHAS     찰스강 더미 변수(강과 접하면 1, 아니면 0)
  - NOX      질소산화물 농도(1천만분의 1)
  - RM       주택당 평균 방 개수
  - AGE      1940년 이전에 지어진 소유주택 비율
  - DIS      5개 보스턴 고용센터까지의 가중 거리
  - RAD      방사형 고속도로 접근성 지수
  - TAX      1만 달러당 재산세율
  - PTRATIO  도시별 학생-교사 비율
  - B        1000(Bk - 0.63)^2 (Bk: 흑인 비율)
  - LSTAT    저소득층 인구 비율(%)
  - MEDV     소유주택의 중앙값(천 달러 단위)

결측치: 없음

중복 데이터: 없음

UCI ML housing 데이터셋 복사본입니다.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
## 3.4 데이터 불러오기 (Boston Housing)
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df = pd.read_csv('housing.csv', header=None, delim_whitespace=True, names=column_names)
# 4. Some visualisations


# 4. 데이터 전처리 (Data Preprocessing)
"""
## 4.1 결측치 처리 (Missing Value Handling)
- Boston Housing 데이터셋에는 결측치가 없으나, 실제 운영 환경에서는 누락된 값이 있을 수 있습니다.
- **이유**: 결측치는 학습 시 정보 손실과 예측 성능 저하를 유발하므로 반드시 검증합니다.
"""
# 결측치 확인
print(df.isnull().sum())

"""
## 4.2 이상치 탐지 및 제거 (Outlier Detection and Removal)
- 회귀 모델의 안정성을 위해 IQR 기반 이상치 제거를 수행합니다.
- **코드 흐름**:
  1. 각 특성의 Q1, Q3 계산
  2. IQR = Q3 - Q1
  3. 하한 = Q1 - 1.5*IQR, 상한 = Q3 + 1.5*IQR
  4. 해당 범위를 벗어나는 샘플 인덱스 수집 및 제거
"""
def detect_outliers_iqr(data, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_list_col = data[(data[col] < lower) | (data[col] > upper)].index
        outlier_indices.extend(outlier_list_col)
    return list(set(outlier_indices))

numeric_features = df.columns[:-1]
outliers = detect_outliers_iqr(df, numeric_features)
print(f"제거 전 데이터 크기: {df.shape}")
df = df.drop(outliers).reset_index(drop=True)
print(f"제거 후 데이터 크기: {df.shape}")

"""
## 4.3 데이터 분할 (Train-Test Split)
"""
from sklearn.model_selection import train_test_split
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
## 4.4 특성 스케일링 (Feature Scaling)
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 모델 학습 및 평가

# (이후 시각화 및 데이터 전처리, 모델링 코드는 Boston Housing 데이터셋에 대해서만 진행)
# 6. Comparing different models
## 6.1 Linear Regression
from sklearn.linear_model import LinearRegression

# Creating and training model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Model making a prediction on test data
y_pred = lm.predict(X_test)
### Linear Regression performance for Avocado dataset
ndf = [Reg_Models_Evaluation_Metrics(lm,X_train,y_train,X_test,y_test,y_pred)]

lm_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
lm_score.insert(0, 'Model', 'Linear Regression')
lm_score
plt.figure(figsize = (10,5))
sns.regplot(x=y_test,y=y_pred)
plt.title('Linear regression for Avocado dataset', fontsize = 20)
### Linear Regression performance for Boston dataset
lm.fit(X_train2, y_train2)
y_pred = lm.predict(X_test2)
ndf = [Reg_Models_Evaluation_Metrics(lm,X_train2,y_train2,X_test2,y_test2,y_pred)]

lm_score2 = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
lm_score2.insert(0, 'Model', 'Linear Regression')
lm_score2
## 6.2 Random Forest
from sklearn.ensemble import RandomForestRegressor

# Creating and training model
RandomForest_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
### Random Forest performance for Avocado dataset
RandomForest_reg.fit(X_train, y_train)
# Model making a prediction on test data
y_pred = RandomForest_reg.predict(X_test)
ndf = [Reg_Models_Evaluation_Metrics(RandomForest_reg,X_train,y_train,X_test,y_test,y_pred)]

rf_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
rf_score.insert(0, 'Model', 'Random Forest')
rf_score
### Random Forest performance for Boston dataset
RandomForest_reg.fit(X_train2, y_train2)
# Model making a prediction on test data
y_pred = RandomForest_reg.predict(X_test2)
ndf = [Reg_Models_Evaluation_Metrics(RandomForest_reg,X_train2,y_train2,X_test2,y_test2,y_pred)]

rf_score2 = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
rf_score2.insert(0, 'Model', 'Random Forest')
rf_score2
## 6.3 Ridge Regression
from sklearn.linear_model import Ridge

# Creating and training model
ridge_reg = Ridge(alpha=3, solver="cholesky")
### Ridge Regression performance for Avocado dataset
ridge_reg.fit(X_train, y_train)
# Model making a prediction on test data
y_pred = ridge_reg.predict(X_test)
ndf = [Reg_Models_Evaluation_Metrics(ridge_reg,X_train,y_train,X_test,y_test,y_pred)]

rr_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
rr_score.insert(0, 'Model', 'Ridge Regression')
rr_score
### Ridge Regression performance for Boston dataset
ridge_reg.fit(X_train2, y_train2)
# Model making a prediction on test data
y_pred = ridge_reg.predict(X_test2)
ndf = [Reg_Models_Evaluation_Metrics(ridge_reg,X_train2,y_train2,X_test2,y_test2,y_pred)]

rr_score2 = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
rr_score2.insert(0, 'Model', 'Ridge Regression')
rr_score2
## 6.4 XGBoost
from xgboost import XGBRegressor
# create an xgboost regression model
XGBR = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.8, colsample_bytree=0.8)
### XGBoost performance for Avocado dataset
XGBR.fit(X_train, y_train)
# Model making a prediction on test data
y_pred = XGBR.predict(X_test)
ndf = [Reg_Models_Evaluation_Metrics(XGBR,X_train,y_train,X_test,y_test,y_pred)]

XGBR_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
XGBR_score.insert(0, 'Model', 'XGBoost')
XGBR_score
### XGBoost performance for Boston dataset
XGBR.fit(X_train2, y_train2)
# Model making a prediction on test data
y_pred = XGBR.predict(X_test2)
ndf = [Reg_Models_Evaluation_Metrics(XGBR,X_train2,y_train2,X_test2,y_test2,y_pred)]

XGBR_score2 = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
XGBR_score2.insert(0, 'Model', 'XGBoost')
XGBR_score2
## 6.5 Recursive Feature Elimination (RFE)
RFE is a wrapper-type feature selection algorithm. This means that a different machine learning algorithm is given and used in the core of the method, is wrapped by RFE, and used to help select features.

Random Forest has usually good performance combining with RFE
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

# create pipeline
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=60)
model = RandomForestRegressor()
rf_pipeline = Pipeline(steps=[('s',rfe),('m',model)])
### Random Forest RFE performance for Avocado dataset
rf_pipeline.fit(X_train, y_train)
# Model making a prediction on test data
y_pred = rf_pipeline.predict(X_test)
ndf = [Reg_Models_Evaluation_Metrics(rf_pipeline,X_train,y_train,X_test,y_test,y_pred)]

rfe_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
rfe_score.insert(0, 'Model', 'Random Forest with RFE')
rfe_score
### Random Forest RFE performance for Boston dataset
# create pipeline
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=8)
model = RandomForestRegressor()
rf_pipeline = Pipeline(steps=[('s',rfe),('m',model)])

rf_pipeline.fit(X_train2, y_train2)
# Model making a prediction on test data
y_pred = rf_pipeline.predict(X_test2)
ndf = [Reg_Models_Evaluation_Metrics(rf_pipeline,X_train2,y_train2,X_test2,y_test2,y_pred)]

rfe_score2 = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
rfe_score2.insert(0, 'Model', 'Random Forest with RFE')
rfe_score2
# 7. Final Model Evaluation
## 7.1 Avocado dataset
predictions = pd.concat([rfe_score, XGBR_score, rr_score, rf_score, lm_score], ignore_index=True, sort=False)
predictions
## 7.2 Boston dataset
predictions2 = pd.concat([rfe_score2, XGBR_score2, rr_score2, rf_score2, lm_score2], ignore_index=True, sort=False)
predictions2
## 7.3 Visualizing Model Performance
f, axe = plt.subplots(1,1, figsize=(18,6))

predictions.sort_values(by=['Cross Validated R2 Score'], ascending=False, inplace=True)

sns.barplot(x='Cross Validated R2 Score', y='Model', data = predictions, ax = axe)
axe.set_xlabel('Cross Validated R2 Score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)

axe.set(title='Model Performance for Avocado dataset')

plt.show()
f, axe = plt.subplots(1,1, figsize=(18,6))

predictions2.sort_values(by=['Cross Validated R2 Score'], ascending=False, inplace=True)

sns.barplot(x='Cross Validated R2 Score', y='Model', data = predictions2, ax = axe)
axe.set_xlabel('Cross Validated R2 Score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)

axe.set(title='Model Performance for Boston dataset')
plt.show()
# 8. Bonus: hyperparameter Tuning Using GridSearchCV
Hyperparameter tuning is the process of tuning the parameters present as the tuples while we build machine learning models. These parameters are defined by us. Machine learning algorithms never learn these parameters. These can be tuned in different step.

GridSearchCV is a technique for finding the optimal hyperparameter values from a given set of parameters in a grid. It's essentially a cross-validation technique. The model as well as the parameters must be entered. After extracting the best parameter values, predictions are made.

The “best” parameters that GridSearchCV identifies are technically the best that could be produced, but only by the parameters that you included in your parameter grid.
## 8.1 Tuned Ridge Regression
from sklearn.preprocessing import PolynomialFeatures

# Polynomial features are those features created by raising existing features to an exponent. 
# For example, if a dataset had one input feature X, 
# then a polynomial feature would be the addition of a new feature (column) where values were calculated by squaring the values in X, e.g. X^2.

steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=3.8, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)

# Model making a prediction on test data
y_pred = ridge_pipe.predict(X_test)
from sklearn.model_selection import GridSearchCV

alpha_params = [{'model__alpha': list(range(1, 15))}]

clf = GridSearchCV(ridge_pipe, alpha_params, cv = 10)
### Tuned Ridge Regression performance for Avocado dataset
# Fit and tune model
clf.fit(X_train, y_train)
# Model making a prediction on test data
y_pred = ridge_pipe.predict(X_test)
# The combination of hyperparameters along with values that give the best performance of our estimate specified
print(clf.best_params_)
ndf = [Reg_Models_Evaluation_Metrics(clf,X_train,y_train,X_test,y_test,y_pred)]

clf_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
clf_score.insert(0, 'Model', 'Tuned Ridge Regression')
clf_score
### Tuned Ridge Regression performance for Boston dataset
steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=3.8, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train2, y_train2)

# Model making a prediction on test data
y_pred = ridge_pipe.predict(X_test2)

alpha_params = [{'model__alpha': list(range(1, 15))}]

clf = GridSearchCV(ridge_pipe, alpha_params, cv = 10)
# Fit and tune model
clf.fit(X_train2, y_train2)
# Model making a prediction on test data
y_pred = ridge_pipe.predict(X_test2)
# The combination of hyperparameters along with values that give the best performance of our estimate specified
print(clf.best_params_)
ndf = [Reg_Models_Evaluation_Metrics(clf,X_train2,y_train2,X_test2,y_test2,y_pred)]

clf_score2 = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
clf_score2.insert(0, 'Model', 'Tuned Ridge Regression')
clf_score2
# 9. Final performance comparison
## 9.1 Avocado data set
result = pd.concat([clf_score, predictions], ignore_index=True, sort=False)
result
f, axe = plt.subplots(1,1, figsize=(18,6))

result.sort_values(by=['Cross Validated R2 Score'], ascending=False, inplace=True)

sns.barplot(x='Cross Validated R2 Score', y='Model', data = result, ax = axe)
#axes[0].set(xlabel='Region', ylabel='Charges')
axe.set_xlabel('Cross Validated R2 Score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)
axe.set(title='Model Performance for Avocado dataset')

plt.show()
## 9.2 Boston data set
result = pd.concat([clf_score2, predictions2], ignore_index=True, sort=False)
result
f, axe = plt.subplots(1,1, figsize=(18,6))

result.sort_values(by=['Cross Validated R2 Score'], ascending=False, inplace=True)

sns.barplot(x='Cross Validated R2 Score', y='Model', data = result, ax = axe)
#axes[0].set(xlabel='Region', ylabel='Charges')
axe.set_xlabel('Cross Validated R2 Score', size=16)
axe.set_ylabel('Model')
axe.set_xlim(0,1.0)
axe.set(title='Model Performance for Boston dataset')

plt.show()
# 10. Other notebooks
Clustering methods - comprehensive study

https://www.kaggle.com/code/marcinrutecki/clustering-methods-comprehensive-study

Outlier detection methods!

https://www.kaggle.com/code/marcinrutecki/outlier-detection-methods

Multicollinearity - detection and remedies

https://www.kaggle.com/code/marcinrutecki/multicollinearity-detection-and-remedies
# 11. Some referrals
https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html
