

import pandas as pd 
import numpy as np
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
%matplotlib inline
import random as rn
## Data Load
# load the dataset
df = pd.read_csv('data/creditcard.csv')

# manual parameters
RANDOM_SEED = 42
TRAINING_SAMPLE = 200000
VALIDATE_SIZE = 0.2

# setting random seeds for libraries to ensure reproducibility
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)  # PyTorch용 시드 설정

df
## Data Preprocessing
# 모든 열의 이름 소문자 처리 후 class를 lable로 이름 변경
df.columns = map(str.lower, df.columns)
df.rename(columns={'class': 'label'}, inplace=True)
df.head()
# amount 특성을 정규 분포처럼 만들기 위해 로그 변환을 적용
df['log10_amount'] = np.log10(df.amount + 0.00001) # 0 방지를 위해 더해 줌
df = df[
    [col for col in df if col not in ['label', 'log10_amount']] + 
    ['log10_amount', 'label']
]
# 불필요한 열 제거 및 label을 정상거래와 비정상 거래로 분리

# 참고로, AutoEncoder를 이용할 시에는 비정상은 학습에 사용하지 않으므로 필요없음. 다만 ML을 이용할 때 쓰려고 SMOTE or UnderSampling 진행
# 불필요한 열 제거
df = df.drop(['time', 'amount'], axis=1)
# 라벨로 분리
fraud = df[df.label == 1]
clean = df[df.label == 0]
df.describe()
# 클래스 분포 확인
print("정상/사기 거래 수:")
print(df['label'].value_counts())
print("클래스 비율 (%):")
print(df['label'].value_counts(normalize=True) * 100)

# 현재 정상거래의 비율이 약 99%에 해당함 굉장히 불균형한 데이터
df.hist(figsize=(16,12), bins=50)
plt.suptitle("vairable distribution", fontsize=18)
plt.tight_layout()
plt.show()

# 여기서부터 ML기반과 AE 기반으로 나눠서 데이터를 사용 ML은 불균형 데이터 처리를 해줘야하는 과정  

# AE는 오직 normal transactions(정상 거래)을 학습 시켜 model이 정확하게 reconstruction 하도록 함  

# >그 결과, 사기 거래가 정상 거래와 충분히 다르면, AE는 이를 잘 재구성하지 못하고, 재구성 오차(reconstruction loss) 가 커짐  
# >
# >재구성 오차가 특정 임계값(threshold)을 넘는 경우, 이상(anomalous) 으로 간주하고 사기로 분류
### ML 기반 model 방식
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
# 학습하는 과정을 ML은 하나로 통합
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model'):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    print(f"{model_name} - Classification Report")
    print(classification_report(y_test, y_pred, digits=4))

    auc_score = roc_auc_score(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    print(f"AUC: {auc_score:.4f} / Average Precision: {ap_score:.4f}")
    return y_pred, y_proba
# ML은 oversmapling 기법인 SMOTE 적용  
# 다만, test에는 smote를 적용하면 안되므로 미리 분류
# 분류
from sklearn.model_selection import train_test_split

# 공통 데이터 분할
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=RANDOM_SEED)

# 공통 테스트셋
X_test_common = df_test.drop('label', axis=1).values
y_test_common = df_test['label'].values


# 학습 데이터 분리
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values

# Oversampling을 진행한 경우
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=RANDOM_SEED)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("SMOTE 후 학습셋 클래스 분포:")
print(pd.Series(y_train_balanced).value_counts())
# Undersampling을 진행한 경우  
# 현재는 시간이 너무 오래 걸려 under로 진행
# 사기 거래 수만큼 정상 거래 샘플링
fraud_train = df_train[df_train.label == 1]
clean_train = df_train[df_train.label == 0]

# undersample 정상 거래
clean_sampled = clean_train.sample(n=len(fraud_train), random_state=RANDOM_SEED)
df_ml_train = pd.concat([clean_sampled, fraud_train]).sample(frac=1, random_state=RANDOM_SEED)

# 학습용 feature, label 분리
X_train_ml = df_ml_train.drop('label', axis=1).values
y_train_ml = df_ml_train['label'].values

# Data scaling 적용
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# 전체 feature에 대해 스케일링 적용
X_train_ml = scaler.fit_transform(X_train_ml)  # ML용 학습 데이터
X_test_common = scaler.transform(X_test_common)  # 공통 테스트셋

# 머신러닝 모델 정의 (undersampling 이용)

# 모델 정의
logistic = LogisticRegression(max_iter=1000)
dtree = DecisionTreeClassifier(random_state=RANDOM_SEED)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)

results = {}

# 전체 모델 학습 및 평가
for model, name in zip(
    [logistic, dtree, rf, xgb],
    ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
):
    print("="*70)
    y_pred, y_proba = train_and_evaluate_model(model, X_train_ml, y_train_ml, X_test_common, y_test_common, model_name=name)
    results[name] = {
        'y_pred': y_pred,
        'y_proba': y_proba
    }

### ML 모델 PR Curve / ROC Curve 시각화
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# PR Curve 시각화 함수
def plot_pr_curves(results, y_test):
    plt.figure(figsize=(8,6))
    
    for model_name, output in results.items():
        y_score = output['y_proba']
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

# ROC Curve 시각화 함수
def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8,6))
    
    for model_name, output in results.items():
        y_score = output['y_proba']
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})")
    
    plt.plot([0,1], [0,1], 'k--', lw=2)  # 대각선 기준선
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

# PR Curve
plot_pr_curves(results, y_test_common)

# ROC Curve
plot_roc_curves(results, y_test_common)

# Undersampling을 사용하다보니 성능이 좋진 않음
### Autoencoder (AE) 모델 설계 및 학습

# Train / Test data 구성
# 검증용: 학습용에서 다시 나누기
from sklearn.model_selection import train_test_split
X_train_ae = clean_train.drop('label', axis=1)
X_train_ae, X_val_ae = train_test_split(X_train_ae, test_size=VALIDATE_SIZE, random_state=RANDOM_SEED)

# 동일한 공통 테스트셋 사용
X_test_ae = X_test_common  # AE도 동일하게 사용
y_test_ae = y_test_common

#### AE model 정의
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 16),
        #     nn.ELU(),
        #     nn.Linear(16, 8),
        #     nn.ELU(),
        #     nn.Linear(8, 4),
        #     nn.ELU(),
        #     nn.Linear(4, 2),
        #     nn.ELU()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(2, 4),
        #     nn.ELU(),
        #     nn.Linear(4, 8),
        #     nn.ELU(),
        #     nn.Linear(8, 16),
        #     nn.ELU(),
        #     nn.Linear(16, input_dim),
        #     nn.ELU()
        # )

        # 좀 더 무거운 버전
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

#### Data scaling 및 tensor로 변환
import torch
from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data scaling 적용
X_train_ae = scaler.fit_transform(X_train_ae)  # AE용 학습 데이터
X_val_ae = scaler.transform(X_val_ae)          # AE용 validation 데이터

# numpy → tensor
X_train_tensor = torch.tensor(X_train_ae, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_ae, dtype=torch.float32)

# Dataloader
train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, X_val_tensor), batch_size=256, shuffle=False)


input_dim = X_train_ae.shape[1]
model_ae = AutoEncoder(input_dim).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_ae.parameters(), lr=1e-4)

# AE training
num_epochs = 30

for epoch in range(num_epochs):
    model_ae.train()
    train_loss = 0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        x_hat = model_ae(x_batch)
        loss = criterion(x_hat, x_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # 검증
    model_ae.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            x_hat = model_ae(x_batch)
            loss = criterion(x_hat, x_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1:02}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# 테스트셋 → Tensor 변환
X_test_tensor = torch.tensor(X_test_ae, dtype=torch.float32).to(device)

# 재구성 실행
model_ae.eval()
with torch.no_grad():
    X_recon = model_ae(X_test_tensor).cpu().numpy()

# 재구성 오차 계산 (MSE per row)
import numpy as np
reconstruction_errors = np.mean((X_test_ae - X_recon)**2, axis=1)

results['Autoencoder'] = {
    'y_proba': reconstruction_errors,
    'y_pred': (reconstruction_errors > np.percentile(reconstruction_errors, 95)).astype(int),  # 예: 상위 5% 이상을 이상치로 간주
    'y_true': y_test_common
}

# 상위 5% 이상을 이상치로 간주해서 detection 함
plot_pr_curves(results, y_test_common)
plot_roc_curves(results, y_test_common)
