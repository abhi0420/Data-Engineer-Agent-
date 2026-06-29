FROM python:3.13-slim

WORKDIR /de_Agent

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY . .

RUN mkdir -p /de_Agent/data

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]