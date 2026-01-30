FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/vlm_env
ENV PATH="/opt/vlm_env/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir transformers==4.48.0 accelerate==0.34.0
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]