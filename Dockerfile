FROM python:3.10-slim

WORKDIR /app

# Instala dependências básicas
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala Poetry (versão 2.x)
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

# Copia pyproject e lock
COPY pyproject.toml poetry.lock ./

# Copia a aplicação
COPY . .

# Instala dependências
RUN poetry install --no-interaction --no-ansi

# Expor a porta
EXPOSE 8000

# Comando padrão
CMD ["poetry", "run", "api"]
