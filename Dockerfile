FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование ВСЕХ файлов проекта (включая .whl файлы)
COPY . .

# Создание папки для загрузок
RUN mkdir -p /app/app/static/uploads && \
    chmod -R 755 /app/app/static/uploads

# Установка PyTorch из ЛОКАЛЬНЫХ .whl файлов
# Важно: устанавливаем в правильном порядке
RUN pip install --upgrade pip && \
    pip install torchaudio-*.whl && \
    pip install torchvision-*.whl && \
    pip install torch-*.whl

# Установка остальных зависимостей
RUN pip install -r requirements.txt

# Настройка окружения
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "run.py"]
