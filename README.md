# 📚 RAG Knowledge Assistant

A **Retrieval-Augmented Generation (RAG)** software for companies to centralize their internal knowledge (procedures, policies, technical documents, etc.) and make it accessible through an intuitive web interface.  

Employees can ask questions in natural language, and the system retrieves answers grounded in the company’s documents — ensuring **reliable, deterministic, and auditable responses**.  

---

## ✨ Core Features

- 📂 **Automatic ingestion of documents**  
  - On startup, RAG scans all files in `./data` (`.csv`, `.pdf`, `.txt`).  
- ❓ **Ask questions about company knowledge**  
  - Employees query policies, procedures, or technical docs.  
- 🔄 **Automatic tracking of file changes**  
  - New, updated, or deleted files are detected and synced with the vector database.  
  - Uses **timestamps + file hashes** for efficiency.  
  - **No re-scan of unchanged files**.  
- 🗄 **Persistent vector database**  
  - Stored between restarts.  
  - Manual update available via **"Update Vector Database"** button.  
- 📑 **Cited answers**  
  - Responses include **document references**.  
- 💬 **Conversation memory**  
  - Maintains context for follow-up questions.  
  - Full chat history is visible to the user.  
- ⚠️ **Safe responses**  
  - If data is insufficient: responds with  
    > `"I don't have answer for this."`  
  - No hallucinations or random web answers.  
- 🔍 **Database insights**  
  - **"Vector Database Information"** → shows scanned files.  
  - **"Show Database Stats"** → displays statistics.  

---

## 🛠️ Tech Stack

| Component         | Technology                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Backend**       | ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)        |
| **Frontend**      | ![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit)     |
| **Container**     | ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) |
| **Vector DB**     | ![Chroma](https://img.shields.io/badge/Chroma-VectorDB-green)               |
| **LLM/Embeddings**| ![OpenAI](https://img.shields.io/badge/OpenAI-black?logo=openai)            |
| **CI/CD**         | ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white) |
| **Testing**       | ![pytest](https://img.shields.io/badge/pytest-yellow)                       |
| **Code Quality**  | ![Black](https://img.shields.io/badge/black-000000?logo=python) ![isort](https://img.shields.io/badge/isort-lightgrey) ![Ruff](https://img.shields.io/badge/ruff-orange) |

---

## 🔐 Security

- 🔒 Secrets are **never hardcoded** — environment variables (`.env`) are passed securely via **docker-compose**.  
- 🔒 `.env` file is excluded from the repo.  
- 🔒 Docker containerization ensures environment isolation.  

---

## ⚡ Scalability & Maintainability

- 🐳 **Docker Compose** → easy to scale into microservices (separate DB, frontend, etc.).  
- ♻️ **Healthchecks** → automatic restart on failure.  
- 💻 **Environment agnostic** → runs on any machine with Docker.  
- 🚀 **Optimized Docker layers** → requirements installed first, then code copied → faster builds & caching.  
- 📂 `./data` is company-specific (excluded from repo for real deployments, included only for testing).  

---

## 📦 Useful Commands

```bash
# Build and run the application
docker-compose up --build

# Run the linter (check only)
make lint

# Run the tests
make test

# Auto-fix lint errors
make lint-fix

# Delete unused Docker images, containers, networks
make docker-prune

# Delete the vector database
make docker-delete
```

## 🚀 Future Improvements

- 🔗 Use **LangChain** instead of custom code for more advanced RAG orchestration.  
- 🎨 Switch from **Streamlit** to **Vue.js** frontend.  
- 🧪 Implement **end-to-end system tests** without mocks, integrated into the pipeline.  
- 📊 Add **logging and monitoring** for better observability and debugging.  

---

## 📈 Methodology

- Followed **Agile autoincremental methodology**.  
- Multiple iterations with planned **backlog and issues** per iteration.  
- Well-defined **milestones** with deadlines.  
- Each issue assigned to an engineer with:  
  - **Priority:** MUST, SHOULD, COULD  
  - **Estimation:** days, hours, minutes  
  - **Acceptance criteria**, instructions, dev notes, etc.  
- Issues progressed through workflow: **Backlog → Ready → In Progress → Under Review → Done**

## 💻 Git workflow:
- Clean, consistent commit messages  
- Rebasing to maintain linear history  
- Easy to revert changes if needed  
- **Branching & PRs:**  
  - Separate branch per issue  
  - Pull Requests to merge into `main`  
  - Branch auto-deletion after merge  
  - PRs require passing **linter + tests**  
- All enforcement automated via **GitHub Actions pipelines and rules**
