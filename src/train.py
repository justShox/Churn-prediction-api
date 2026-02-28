"""
train.py — standalone скрипт для обучения модели.

Представляет собой Production-ready версию пайплайна из ноутбуков:
    - 01_EDA.ipynb → предобработка и очистка данных
    - 02_preprocessing.ipynb → кодирование и масштабирование
    - 03_modeling.ipynb → подбор гиперпараметров и обучение

Использование:
    python src/train.py

Результат:
    models/catboost_model.pkl — обученная модель
    models/scaler.pkl — fitted StandardScaler
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
DATA_PATH  = BASE_DIR / 'data' / 'raw' / 'TZ.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Загрузка данных ───────────────────────────────────────────────────────
print("📂 Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f"   Размер датасета: {df.shape}")

# ── 2. Предобработка ─────────────────────────────────────────────────────────
print("\n⚙️  Предобработка...")

# Удаляем нерелевантные признаки
df = df.drop(columns=['ID', 'ID_клиента', 'фамилия'], errors='ignore')

# Флаг наличия депозита + заполнение пропусков
df['есть_депозит']  = df['баланс_депозита'].notna().astype(int)
df['баланс_депозита'] = df['баланс_депозита'].fillna(0)

# OHE для категориальных признаков
df = pd.get_dummies(df, columns=['город', 'пол'], drop_first=True)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"   Признаков после предобработки: {df.shape[1] - 1}")

# ── 3. Разделение на train/test ──────────────────────────────────────────────
print("\n✂️  Разделение на train/test (80/20)...")
X = df.drop(columns=['ушел_из_банка'])
y = df['ушел_из_банка']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"   Доля оттока — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

# ── 4. Масштабирование ───────────────────────────────────────────────────────
print("\n📏 Масштабирование числовых признаков...")
num_cols = ['кредитный_рейтинг', 'возраст', 'стаж_в_банке',
            'баланс_депозита', 'оценочная_зарплата']

scaler = StandardScaler()
scaler.fit(X_train[num_cols])
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
print("   Скейлер сохранён → models/scaler.pkl")

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

# ── 8. Сохранение модели ─────────────────────────────────────────────────────
joblib.dump(model, MODELS_DIR / 'catboost_model.pkl')
print("\n Модель сохранена → models/catboost_model.pkl")
print("Обучение завершено!")