FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system wm811k \
    && useradd --system --gid wm811k --create-home wm811k

RUN python -m pip install --no-cache-dir \
    "psycopg[binary]>=3.1,<4"

COPY main.py ./main.py
COPY src ./src

USER wm811k

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]