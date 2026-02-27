FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv in a robust way
RUN pip install --no-cache-dir uv

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/generated_scripts

CMD ["python", "main.py"]
