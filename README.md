
# 🛡️ Credit Card Fraud Detection — Full MLOps Pipeline

Este projeto implementa um pipeline **completo de Machine Learning + MLOps**, incluindo:

✔ DVC Pipeline (download, preprocess, train, eval)
✔ FastAPI com endpoints de predição (single e batch)
✔ Re-treino via API
✔ Versionamento de modelos
✔ Orquestração com Poetry
✔ Notebook de análise incluído

---

## 🚀 Como rodar o projeto

### 1️⃣ Instalar dependências
```sh
poetry install
```

### 2️⃣ Executar o pipeline completo
```sh
poetry run dvc repro
```

### 3️⃣ Subir API
```sh
poetry run api
```

---

## 📁 Estrutura do Projeto

```
src/
 └── creditcard_ml/
      ├── api/                  # FastAPI
      ├── core/                 # Lógica de ML
      ├── model/                # Modelos + scalers
      ├── training/             # Script de treino
      ├── scripts/              # Pipeline DVC
```

---

## 📘 Notebook

O notebook final está disponível em:

📄 **final_notebook.ipynb**

Inclui:

- Estatísticas do dataset
- Distribuição das classes
- Feature engineering
- Métricas do modelo
- Comparação entre modelos
- Curva ROC + Matriz de Confusão

---

## ⚙️ Endpoints Principais

### 🔹 Predição

`POST /predict`

### 🔹 Predição em Lote

`POST /batch`

### 🔹 Re-treinar modelo

`POST /mlops/retrain`

---

## 📦 Versionamento de Modelos

Usa:

- `DVC`
- Remote storage local configurado
- Cada treino gera um modelo novo versionado

---

## ✨ Autor

Criado por Rodrigo — Projeto completo com ML + API + MLOps.
