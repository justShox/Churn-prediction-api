"""
train.py — standalone скрипт для обучения модели.

Представляет собой Production-ready версию пайплайна из ноутбуков:
    - 01_EDA.ipynb → предобработка и очистка данных
    - 02_preprocessing.ipynb → кодирование, FE, масштабирование
    - 03_modeling.ipynb → подбор гиперпараметров и обучение

Использование:
    python src/train.py

Результат:
    models/catboost_model.pkl     — обученная модель
    models/preprocessing.pkl      — артефакты препроцессинга
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier
import joblib
import optuna
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Пути ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / 'data' / 'processed' / 'churn_cleaned.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Загрузка данных ───────────────────────────────────────────────────────
print("📂 Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f"   Размер датасета: {df.shape}")

# ── 2. Предобработка ─────────────────────────────────────────────────────────
print("\n⚙️  Предобработка...")

# Переименование Rus → Eng
COLUMN_MAPPING = {
    'кредитный_рейтинг': 'credit_score',
    'город': 'city',
    'пол': 'gender',
    'возраст': 'age',
    'стаж_в_банке': 'years_with_bank',
    'баланс_депозита': 'deposit_balance',
    'число_продуктов': 'num_products',
    'есть_кредитка': 'has_credit_card',
    'активный_клиент': 'is_active_member',
    'оценочная_зарплата': 'estimated_salary',
    'ушел_из_банка': 'target',
    'есть_депозит': 'has_deposit'
}
df = df.rename(columns=COLUMN_MAPPING)

# Флаг наличия депозита + заполнение пропусков
df['has_deposit'] = df['deposit_balance'].notna().astype(int)
df['deposit_balance'] = df['deposit_balance'].fillna(0)

# OHE
df = pd.get_dummies(df, columns=['city', 'gender'], drop_first=True)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Feature Engineering
risk_mapping = {1.0: 2, 2.0: 0, 3.0: 3, 4.0: 3}
df['product_risk_level'] = df['num_products'].map(risk_mapping)
df['balance_to_salary_ratio'] = df['deposit_balance'] / (df['estimated_salary'] + 1)
df['active_single_product'] = ((df['is_active_member'] == 1) & (df['num_products'] == 1)).astype(int)
df['loyal_high_balance'] = ((df['years_with_bank'] >= 5) & (df['deposit_balance'] > 50000)).astype(int)
df['young_no_card'] = ((df['age'] < 30) & (df['has_credit_card'] == 0)).astype(int)
df['age_tenure_interaction'] = df['age'] * df['years_with_bank']
df['salary_products_interaction'] = df['estimated_salary'] * df['num_products']
df['age_squared'] = df['age'] ** 2

print(f"   Признаков после FE: {df.shape[1] - 1}")

# ── 3. Разделение на train/test ──────────────────────────────────────────────
print("\n✂️  Разделение на train/test (80/20)...")
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"   Доля оттока — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

# ── 4. Масштабирование ───────────────────────────────────────────────────────
print("\n📏 Масштабирование числовых признаков...")
NUM_COLS = ['credit_score', 'age', 'years_with_bank', 'deposit_balance', 'estimated_salary',
            'balance_to_salary_ratio', 'age_tenure_interaction', 'salary_products_interaction',
            'age_squared', 'product_risk_level', 'age_group']

# Удаляем age_group если его нет (он может быть добавлен через pd.cut)
NUM_COLS = [c for c in NUM_COLS if c in X_train.columns]

scaler = StandardScaler()
X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_test[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

# Удаляем мультиколлинеарные признаки
FEATURES_TO_DROP = ['age_squared', 'age_group']
X_train = X_train.drop(columns=FEATURES_TO_DROP, errors='ignore')
X_test  = X_test.drop(columns=FEATURES_TO_DROP, errors='ignore')

print(f"   Финальных признаков: {X_train.shape[1]}")

# ── 5. Подбор гиперпараметров (Optuna) ──────────────────────────────────────
print("\n🔍 Подбор гиперпараметров (Optuna, 50 trials)...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'iterations':          trial.suggest_int('iterations', 200, 1000),
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':               trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength':     trial.suggest_float('random_strength', 0, 1),
        'auto_class_weights':  'Balanced',
        'random_seed':         42,
        'verbose':             False
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"   Лучший CV ROC-AUC: {study.best_value:.4f}")

# ── 6. Обучение финальной модели ─────────────────────────────────────────────
print("\n🚀 Обучение финальной модели...")
best_params = study.best_params
best_params.update({'auto_class_weights': 'Balanced', 'random_seed': 42, 'verbose': 100})

model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# ── 7. Оценка модели ─────────────────────────────────────────────────────────
print("\n📊 Оценка на тестовой выборке:")
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
    'F1':        round(f1_score(y_test, y_pred), 4),
    'Precision': round(precision_score(y_test, y_pred), 4),
    'Recall':    round(recall_score(y_test, y_pred), 4),
}
for k, v in metrics.items():
    print(f"   {k:<12}: {v}")

# ── 8. Сохранение ────────────────────────────────────────────────────────────
joblib.dump(model, MODELS_DIR / 'catboost_model.pkl')

OHE_COLUMNS = ['city', 'gender']
preprocessing_artifacts = {
    'column_mapping': COLUMN_MAPPING,
    'ohe_columns': OHE_COLUMNS,
    'final_features': list(X_train.columns),
    'scaler': scaler,
}
joblib.dump(preprocessing_artifacts, MODELS_DIR / 'preprocessing.pkl')

print(f"\n✅ Модель сохранена → models/catboost_model.pkl")
print(f"✅ Препроцессинг сохранён → models/preprocessing.pkl")
print(f"   Фичи: {list(X_train.columns)}")
print("Обучение завершено!")