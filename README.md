# 🏦 Bank Customer Churn Prediction

Сервис предсказания вероятности оттока клиентов банка на основе ML-модели CatBoost.  
Реализован в виде REST API с документацией Swagger и упакован в Docker-контейнер.

---

## 📊 Результаты модели

| Метрика    | Logistic Regression | CatBoost (tuned) |
|------------|:-------------------:|:----------------:|
| ROC-AUC    | 0.8823              | 0.9347           |
| F1         | 0.6410              | 0.7169           |
| Precision  | 0.5330              | 0.6175           |
| Recall     | 0.8039              | 0.8546           |
| CV ROC-AUC | 0.8884 ± 0.0083     | 0.9368 ± 0.0043  |

---

## 🗂️ Структура проекта
```
churn-prediction/
├── data/
│   ├── raw/                  # Исходные данные
│   ├── processed/            # Очищенный датасет после EDA
│   └── train_test/           # Train/test сплиты
├── models/
│   ├── catboost_model.pkl    # Обученная модель
│   └── scaler.pkl            # StandardScaler
├── notebooks/
│   ├── 01_EDA.ipynb          # Разведочный анализ данных
│   ├── 02_preprocessing.ipynb # Предобработка
│   ├── 03_modeling.ipynb     # Обучение и сравнение моделей
│   └── 04_validation.ipynb   # Валидация и SHAP
├── src/
│   └── app.py                # FastAPI сервис
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Запуск проекта

### Вариант 1 — Docker (рекомендуется)
```bash
# Клонировать репозиторий
git clone <your-repo-url>
cd churn-prediction

# Собрать и запустить контейнер
docker-compose up
```

### Вариант 2 — Локально
```bash
# Клонировать репозиторий
git clone <your-repo-url>
cd churn-prediction

# Установить зависимости
pip install -r requirements.txt

# Запустить сервис
python -m uvicorn src.app:app --reload
```

После запуска сервис доступен по адресу: `http://localhost:8000`  
Swagger документация: `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET   | `/`      | Проверка работы сервиса |
| GET   | `/health` | Health check |
| POST  | `/predict` | Предсказание для одного клиента |
| POST  | `/predict_batch` | Предсказание для нескольких клиентов |

### Пример запроса `/predict`
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "кредитный_рейтинг": 620,
       "город": "Атырау",
       "пол": "Female",
       "возраст": 48,
       "стаж_в_банке": 3,
       "баланс_депозита": 95000,
       "число_продуктов": 1,
       "есть_кредитка": 1,
       "активный_клиент": 0,
       "оценочная_зарплата": 75000,
       "есть_депозит": 1
     }'
```

### Пример ответа
```json
{
  "churn_probability": 0.87,
  "prediction": 1,
  "risk_level": "high"
}
```

### Описание полей ответа

| Поле | Тип | Описание |
|------|-----|----------|
| `churn_probability` | float | Вероятность оттока от 0 до 1 |
| `prediction` | int | Бинарный прогноз (1 — уйдёт, 0 — останется) |
| `risk_level` | string | Уровень риска: `low` (<0.4), `medium` (0.4–0.7), `high` (>0.7) |

---

## 🔍 Ключевые выводы

- **Главные факторы оттока:** возраст, число продуктов, активность клиента, город
- **Портрет клиента группы риска:** женщина 45+, из Атырау, неактивный клиент с 1 продуктом
- **Атырау** — аномально высокий отток (42% vs 15-16% в других городах)
- Подключение второго продукта значительно снижает риск оттока

---

## 🛠️ Стек технологий

- **ML:** CatBoost, Scikit-learn, SHAP, Optuna
- **API:** FastAPI, Uvicorn, Pydantic
- **Инфраструктура:** Docker, Docker Compose
- **Анализ данных:** Pandas, NumPy, Matplotlib, Seaborn