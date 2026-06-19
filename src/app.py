from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
model  = joblib.load(BASE_DIR / 'models' / 'catboost_model.pkl')
scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')

NUM_COLS = ['кредитный_рейтинг', 'возраст', 'стаж_в_банке',
            'баланс_депозита', 'оценочная_зарплата']

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

    df['город_Астана'] = int(df['город'].iloc[0] == 'Астана')
    df['город_Атырау'] = int(df['город'].iloc[0] == 'Атырау')
    df['пол_Male']     = int(df['пол'].iloc[0] == 'Male')
    df = df.drop(columns=['город', 'пол'])

    feature_order = [
        'кредитный_рейтинг', 'возраст', 'стаж_в_банке', 'баланс_депозита',
        'число_продуктов', 'есть_кредитка', 'активный_клиент',
        'оценочная_зарплата', 'есть_депозит', 'город_Астана',
        'город_Атырау', 'пол_Male'
    ]
    df = df[feature_order]

    return df

# Эндпоинты ────────────────────────────────────────────────────────────────
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