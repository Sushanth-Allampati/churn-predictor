@echo off
echo Starting MLflow UI...
echo Open http://localhost:5000 in your browser
echo Press Ctrl+C to stop
mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
pause