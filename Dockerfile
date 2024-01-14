FROM python:3.11.1-slim

WORKDIR /models

COPY "models/grade_model_V1.joblib" .
COPY "models/subgrade_model_V1.joblib" .
COPY "models/int_rate_model_V1.joblib" .
COPY "models/grade_preprocessor_V1.joblib" .

WORKDIR /api

COPY app/requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ["app/app.py", "app/functions.py", "./"] .

EXPOSE 80

CMD ["uvicorn", "app:app","--host", "0.0.0.0","--port", "80"]