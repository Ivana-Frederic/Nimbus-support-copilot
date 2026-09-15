# Nimbus Support Copilot

Support chatbot for a made up SaaS company, Nimbus Cloud. Answers questions using RAG over a small knowledge base, generates the reply with a local LLM through Ollama, and uses a contextual bandit to pick the retrieval strategy, learning from 👍/👎 feedback as people use it.

Built this to practice the stack that keeps showing up in ML engineer and applied AI postings: RAG, local LLM serving, online RL, fine tuning a transformer, and shipping it in Docker and Kubernetes instead of a notebook.

## What's in it

- RAG: markdown docs chunked, embedded with all-MiniLM-L6-v2, stored in Chroma
- Intent classification: fine tuned DistilBERT, falls back to keyword matching if not trained yet
- LLM answers from Ollama, streamed back as they generate
- LinUCB contextual bandit (written from scratch) picking the retrieval weighting per question, updated from feedback
- Docker Compose for running locally, Kubernetes manifests for deploying it

Arms/weights for the bandit are in `app/rag/retriever.py`, the bandit itself is in `app/rl/bandit.py`. Bandit stats are at `/api/admin/bandit-stats`.

## Folder structure

```
backend/
  app/
    rag/          chunking, embeddings, vector store, retriever, ingestion
    llm/          Ollama client + prompt building
    rl/           the bandit
    classifier/   DistilBERT classifier + keyword fallback
    routers/      /api/chat, /api/chat/stream, /api/feedback, /api/admin, /health
    main.py       wires everything together, auto ingests KB on first run
  scripts/
    ingest_kb.py               re-index the knowledge base
    train_intent_classifier.py fine tune the classifier
  data/
    kb/           Nimbus Cloud docs (markdown)
    intents/      training data for the classifier
  tests/          pytest, no GPU or model downloads needed
frontend/         plain JS chat UI
k8s/              Kubernetes manifests
docker-compose.yml
```

## Running it

```bash
docker compose up --build
```

Needs Docker Desktop and about 4GB free for models. Pulls `qwen2.5:0.5b` by default, which is honestly too small and sometimes ignores the context it's given. Bumping it up helps:

```
# .env file next to docker-compose.yml
OLLAMA_MODEL=qwen2.5:1.5b
```

To train the real intent classifier:

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt
python -m scripts.train_intent_classifier
```

Weights land in `backend/app/classifier/model/`, mounted into the backend container already, so restart it after training.

## Kubernetes

```bash
docker build -t nimbus-copilot-backend:local ./backend
docker build -t nimbus-copilot-frontend:local ./frontend
kind load docker-image nimbus-copilot-backend:local nimbus-copilot-frontend:local
kubectl apply -k k8s/
kubectl -n nimbus-copilot wait --for=condition=complete job/ollama-pull-model --timeout=600s
kubectl -n nimbus-copilot port-forward svc/frontend 8080:8080
```

## Without Docker

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt
cp ../.env.example .env
uvicorn app.main:app --reload
```

## Tests

```bash
cd backend
pip install -r requirements-test.txt
pytest -v
```

28 tests. Vector store and LLM are faked, bandit/database/keyword classifier are real, so the wiring between them is actually tested.

## Known limitations

- SQLite and embedded Chroma only work with one writer, fine for this but not for real horizontal scaling
- Intent classifier training data is made up, not real tickets
- No auth or rate limiting on the API
