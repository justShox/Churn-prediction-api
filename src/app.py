from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

# ── Загрузка артефактов ─────────────────────────────────────────────────────
preprocessing    = joblib.load(BASE_DIR / 'models' / 'preprocessing.pkl')

column_mapping   = preprocessing['column_mapping']   # Rus → Eng
ohe_columns      = preprocessing['ohe_columns']      # ['city', 'gender']
final_features   = preprocessing['final_features']   # порядок фичей
scaler           = preprocessing['scaler']            # fitted StandardScaler

# Модель: калиброванная или обычная
# Калиброванная даёт более точные вероятности, обычная — быстрее
USE_CALIBRATED = True
if USE_CALIBRATED:
    model = joblib.load(BASE_DIR / 'models' / 'catboost_model_calibrated.pkl')
else:
    model = joblib.load(BASE_DIR / 'models' / 'catboost_model.pkl')

# Числовые колонки для скейлера (из preprocessing notebook — scale ПЕРЕД удалением age_squared/age_group)
NUM_COLS = ['credit_score', 'age', 'years_with_bank', 'deposit_balance', 'estimated_salary',
            'balance_to_salary_ratio', 'age_tenure_interaction', 'salary_products_interaction',
            'age_squared', 'product_risk_level', 'age_group']

app = FastAPI(
    title='Churn Prediction API',
    description='Предсказание вероятности оттока клиента банка',
    version='1.0.0'
)


class CustomerData(BaseModel):
    кредитный_рейтинг:  float = Field(..., ge=300, le=850,  json_schema_extra={'example': 650.0})
    город:              str   = Field(...,                   json_schema_extra={'example': 'Алматы'})
    пол:                str   = Field(...,                   json_schema_extra={'example': 'Male'})
    возраст:            float = Field(..., ge=18,  le=100,   json_schema_extra={'example': 35.0})
    стаж_в_банке:       float = Field(..., ge=0,   le=50,    json_schema_extra={'example': 5.0})
    баланс_депозита:    float = Field(..., ge=0,             json_schema_extra={'example': 0.0})
    число_продуктов:    float = Field(..., ge=1,   le=4,     json_schema_extra={'example': 1.0})
    есть_кредитка:      float = Field(..., ge=0,   le=1,     json_schema_extra={'example': 1.0})
    активный_клиент:    float = Field(..., ge=0,   le=1,     json_schema_extra={'example': 0.0})
    оценочная_зарплата: float = Field(..., ge=0,             json_schema_extra={'example': 50000.0})
    есть_депозит:       float = Field(..., ge=0,   le=1,     json_schema_extra={'example': 0.0})

    @field_validator('город')
    @classmethod
    def validate_city(cls, v):
        allowed = ['Алматы', 'Астана', 'Атырау']
        if v not in allowed:
            raise ValueError(f'город должен быть одним из: {allowed}')
        return v

    @field_validator('пол')
    @classmethod
    def validate_gender(cls, v):
        allowed = ['Male', 'Female']
        if v not in allowed:
            raise ValueError(f'пол должен быть одним из: {allowed}')
        return v


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction:        int
    risk_level:        str


def preprocess(data: CustomerData) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])

    # ── OHE (ДО rename, работаем с оригинальными Rus колонками) ──────────
    df['city_Астана'] = int(data.город == 'Астана')
    df['city_Атырау'] = int(data.город == 'Атырау')
    df['gender_Male'] = int(data.пол == 'Male')
    df = df.drop(columns=['город', 'пол'])

    # ── Feature Engineering (зеркало из 02_preprocessing.ipynb) ──────────
    risk_mapping = {1.0: 2, 2.0: 0, 3.0: 3, 4.0: 3}
    df['product_risk_level'] = df['число_продуктов'].map(risk_mapping)

    df['balance_to_salary_ratio'] = df['баланс_депозита'] / (df['оценочная_зарплата'] + 1)
    df['active_single_product'] = ((df['активный_клиент'] == 1) & (df['число_продуктов'] == 1)).astype(int)
    df['loyal_high_balance'] = ((df['стаж_в_банке'] >= 5) & (df['баланс_депозита'] > 50000)).astype(int)
    df['young_no_card'] = ((df['возраст'] < 30) & (df['есть_кредитка'] == 0)).astype(int)
    age_group_mapping = {'young': 0, 'adult': 1, 'middle_aged': 2, 'senior': 3}
    df['age_group'] = pd.cut(
        df['возраст'],
        bins=[0, 25, 35, 50, 100],
        labels=['young', 'adult', 'middle_aged', 'senior']
    ).map(age_group_mapping)

    df['age_tenure_interaction'] = df['возраст'] * df['стаж_в_банке']
    df['salary_products_interaction'] = df['оценочная_зарплата'] * df['число_продуктов']
    df['age_squared'] = df['возраст'] ** 2

    # ── Rename Rus → Eng (после OHE и FE) ──────────────────────────────
    df = df.rename(columns=column_mapping)

    # ── Масштабирование числовых ────────────────────────────────────────
    cols_to_scale = [c for c in NUM_COLS if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # ── Порядок фичей как при обучении ──────────────────────────────────
    df = df[final_features]

    return df


@app.get('/')
def root():
    return {'message': 'Churn Prediction API is running'}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        df = preprocess(customer)
        prob  = float(model.predict_proba(df)[0][1])
        pred  = int(prob >= 0.5)
        risk  = 'high' if prob >= 0.7 else 'medium' if prob >= 0.4 else 'low'

        return PredictionResponse(
            churn_probability=round(prob, 4),
            prediction=pred,
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict_batch')
def predict_batch(customers: list[CustomerData]):
    try:
        results = []
        for customer in customers:
            df   = preprocess(customer)
            prob = float(model.predict_proba(df)[0][1])
            pred = int(prob >= 0.5)
            risk = 'high' if prob >= 0.7 else 'medium' if prob >= 0.4 else 'low'
            results.append({
                'churn_probability': round(prob, 4),
                'prediction': pred,
                'risk_level': risk
            })
        return {'predictions': results, 'count': len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
