FROM agrigorev/zoomcamp-model:3.8.12-slim
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "churn_docker_api.py", "./"]
RUN pip install pipenv && pipenv install --system --deploy
EXPOSE 7000
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:7000", "churn_docker_api:app"]