# 1 - Image de base
FROM python:3.11-slim

# 2 - Définir le répertoire de travail
WORKDIR /app

# 3 - Copier les fichiers nécessaires
COPY api/app.py /app/app.py
COPY model/sentiment_model.pt /app/model/sentiment_model.pt
COPY requirements.txt /app/requirements.txt

# 4 - Installer les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5 - Exposer le port 8000
EXPOSE 8000

# 6 - Lancer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
