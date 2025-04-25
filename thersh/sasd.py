# # Pima Indians Diabetes - Binary Classification

# >목표 : 환자가 당뇨병(diabetes)인지 아닌지를 정확하게 예측할 수 있는 모델 만들기  

# 데이터셋은 여러 개의 `의료 관련 예측 변수(predictor variables)`와 하나의 목표 변수(target variable)인 `Outcome`으로 구성

# 예측 변수에는 환자가 임신한 횟수(Pregnancies), 체질량지수(BMI), 인슐린 수치(Insulin), 나이(Age) 등이 포함
# ### Dataset Description (변수 설명)

# `Pregnancies`: 지금까지 임신한 횟수

# `Glucose`: 경구 포도당 내성 검사(OGTT) 중 2시간 동안 측정된 혈장 포도당 농도

# `BloodPressure`: 이완기 혈압 (단위: mm Hg)

# `SkinThickness`: 상완 삼두근 피부두께 (단위: mm) — 체지방률을 간접적으로 측정

# `Insulin`: 2시간 혈청 인슐린 수치 (단위: mu U/ml)

# `BMI`: 체질량지수 (공식: 체중(kg) / 키(m)^2) — 비만도를 나타냄

# `Pedigree`: Diabetes Pedigree Function — 가족력 기반으로 당뇨병 가능성을 수치화한 값

# `Age`: 나이 (단위: 세)

# `Outcome`: 클래스 변수 (0이면 당뇨병이 아님, 1이면 당뇨병임)
### Data Load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/diabetes.csv")

df.head()
df.shape
### Data Processing
# data 정보 확인
df.info()
df.describe()
# `Pregnancies` 값은 현실적인 범위인 0에서 17 사이로 나타남

# `DiabetesPedigreeFunction`은 가족력을 바탕으로 당뇨병 발생 가능성을 점수화한 함수로, 값의 범위는 0.08에서 2.42

# `Age` 변수는 21세부터 81세까지의 값을 가짐

# `Outcome`은 목표 변수(target variable)로 값이 0이면 건강한 사람(당뇨병이 아님),1이면 당뇨병이 있는 사람을 의미!

# 그런데 `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`와 같은 변수들에는 실제로는 존재할 수 없는 값인 0이 포함되어 있음
# 이러한 비현실적인 값(0)은 전처리 단계에서 수정할 예정 → mean / meadian 값으로 대체

# 여기서는 `median`을 사용!
df.isnull().sum().sum()
df.duplicated().sum()
# 다행히 중복된 데이터는 0개로 존재하지 않음!
#### median으로 대체할 비현실적인 값들은 모두 확인
# 데이터셋에서 0 값이 있는지 확인 (아까 봤던 column들을 5개만 보기)
print("Blood Pressure column에서 0인 값들의 수  : ", df[df['BloodPressure']==0].shape[0])
print("Glucose column에서 0인 값들의 수         : ", df[df['Glucose']==0].shape[0])
print("Skin Thickness column에서 0인 값들의 수  : ", df[df['SkinThickness']==0].shape[0])
print("Insulin column에서 0인 값들의 수         : ", df[df['Insulin']==0].shape[0])
print("BMI column에서 0인 값들의 수             : ", df[df['BMI']==0].shape[0])
### EDA

# Outlier 찾기
plt.figure(figsize = (12,12))
for i,col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(3,3, i+1)
    sns.boxplot(x = col, data = df)
plt.show()
#### Visualization of Target Variable
sns.set_theme(style="darkgrid")
labels = ["Healthy", "Diabetic"]
plt.figure(figsize=(10,7))
plt.title('Pie Chart', fontsize=20)
df['Outcome'].value_counts().plot(kind='pie',labels=labels, subplots=True,autopct='%1.0f%%', labeldistance=1.2, figsize=(9,9))
from matplotlib.pyplot import figure, show
figure(figsize=(8,6))
ax = sns.countplot(x=df['Outcome'], data=df,palette="husl", hue="Outcome")
ax.set_xticklabels(["Healthy", "Diabetic"])
healthy, diabetics = df['Outcome'].value_counts().values
print("당뇨병 환자의 수: ", diabetics)
print("건강한 사람의 수: ", healthy)
# 약 13대 7 비율로 정상인과 당뇨병으로 구성 → Class imbalance 한 것을 확인 가능함!
#### 각 변수에 따라서 당뇨병 결과 확인
plt.figure(figsize = (12,12))
for i,col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(3,3, i+1)
    sns.histplot(x = col, data = df, kde = True)
plt.show()
sns.pairplot(df, hue="Outcome", palette="husl")
#### Correlation Matrix using Heat map

matrix = df.corr()
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(matrix, vmax=.7, square=True, cmap="YlGnBu",annot=True,linewidths=.5); #.set_title('Heat Map',fontsize=20);
plt.title('Heat Map', fontsize=30)
# Outcome 행의 상관계수를 살펴보면, `Glucose`, `BMI`, `Age`가 가장 높은 상관계수를 가짐

# `BloodPressure`, `SkinThickness는` 상관관계가 낮아 예측에 큰 기여를 하지 않으므로 제외 가능함!

# → `Glucose`는 당뇨병 예측에 있어 가장 중요한 지표로 보임
#### Handling Outliers 및 Feature Scaling
# Feature Scaling
from sklearn.preprocessing import QuantileTransformer
quantile  = QuantileTransformer()
dfpima=df.drop('Outcome',axis='columns')
# median과 IQR 값 계산
quantile.fit(dfpima)
dfq = quantile.transform(dfpima)
df_scaled=pd.DataFrame(dfq)
df_scaled.columns =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df_scaled.head()

plt.figure(figsize = (12,12))
for i,col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(3,3, i+1)
    sns.boxplot(x = col, data = df_scaled)
plt.show()

''' x는 feature, y는 label '''
x= df.drop(['Outcome'], axis=1) # 정답은 따로 빼기
y= df['Outcome'] # target variable
from sklearn.impute import SimpleImputer
fill=SimpleImputer(missing_values=0,strategy="mean") # median
x=fill.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0) # random_state=42
### Evaluate Metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
### Model 성능 평가

