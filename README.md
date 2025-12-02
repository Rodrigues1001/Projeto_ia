# Credit Card Fraud Detection – Machine Learning Project

Este projeto implementa uma solução completa de **detecção de fraude em cartões de crédito**, seguindo boas práticas de **Machine Learning Engineering** e **MLOps**, contemplando:

- ✔️ Predições instantâneas via API (FastAPI)
- ✔️ Predições em lote (batch) com suporte a grandes volumes (GB–TB)
- ✔️ Pipeline de features reproduzível
- ✔️ Treinamento automatizado
- ✔️ Versionamento e empacotamento com Poetry
- ✔️ Ambiente isolado e reprodutível com Docker
- ✔️ Testes automatizados (pytest)
- ✔️ Arquitetura profissional com separação clara de camadas

O objetivo é demonstrar a capacidade de estruturar um sistema de Machine Learning robusto, escalável e pronto para produção.

---

# 🧱 Arquitetura

```text
src/
 └── creditcard_ml/
      ├── api/                  # Camada de interface (FastAPI)
      │     ├── app.py
      │     ├── routes/
      │     │     ├── predict_route.py
      │     │     └── batch_route.py
      │     └── schemas/
      │           └── predict_schema.py
      │
      ├── core/                 # Camada de lógica do modelo
      │     ├── model_loader.py
      │     ├── feature_engineering.py
      │     ├── predict_single.py
      │     └── predict_batch.py
      │
      ├── data/
      │     └── loader.py       # Download/ingestão do dataset (KaggleHub)
      │
      ├── model/                # Artefatos treinados
      │     ├── model.pkl
      │     └── scaler.pkl
      │
      ├── training/             # Treinamento
      │     └── train.py
      │
      └── test/                 # Testes automatizados
            ├── test_features.py
            └── test_predict.py

# 🚀 Rodando o projeto com Docker

docker compose up --build

Treinar modelo:
  docker compose run api poetry run train

Testar end-point:
  link-ambiente/docs


