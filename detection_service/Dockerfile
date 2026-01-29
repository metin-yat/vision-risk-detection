FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Sistem kütüphanelerini kur ve Python 3.11'i yapılandır
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pip'i güncelle (AssertionError hatasını genellikle bu çözer)
RUN python3.11 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Kurulumu doğrudan python3.11 üzerinden yap
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/scenes /app/source

# Çalıştırma komutunu python3.11 olarak netleştir
CMD ["python3.11", "main.py"]