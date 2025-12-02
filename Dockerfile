FROM python:3.10

WORKDIR /app

RUN pip install poetry

# Copia primeiro o pyproject e o lock
COPY pyproject.toml poetry.lock ./

# Copia o código antes do poetry install
COPY src ./src

# Agora sim instala as dependências
RUN poetry install --no-interaction --no-ansi

CMD ["poetry", "run", "api"]
