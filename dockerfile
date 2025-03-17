# Étape 1 : Utiliser une image de base Python 3.12
FROM python:3.12-slim

# Étape 2 : Installer Java pour PySpark
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk-headless procps && \
    rm -rf /var/lib/apt/lists/*

# Étape 3 : Définir JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Étape 4 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 5 : Copier les fichiers du projet dans le conteneur
COPY . .

# Étape 6 : Mettre à jour pip et installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Étape 7 : Exposer le port utilisé par l'application Flask
EXPOSE 5002

# Étape 8 : Démarrer l'application Flask
CMD ["python", "api/app.py"]