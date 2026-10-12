# RxReveal: Medical Document Assistant

RxReveal answers questions about your medical PDF documents. It finds the most relevant passages with semantic search (Gemini embeddings), then asks Google Gemini to write a short answer that cites those passages with page numbers. One free Gemini API key covers both.

It started as a Kaggle notebook for the Google x Kaggle Gen AI Intensive Course Capstone 2025Q1 (team: Md Haaris Hussain, Venkata Satyanarayana Gudala). This version is a full app you can run in VS Code, keep on GitHub and deploy with Docker. The original notebook is in `notebooks/`.

> **Important: patient data and the free Gemini tier.** The text of your documents is sent to Google twice: once to build the search index, and again as context for each answer. On the free (unpaid) tier, Google may use this data to improve its products, and human reviewers may read it. Google's terms say not to send sensitive, confidential or personal information to the free tier ([Gemini API terms](https://ai.google.dev/gemini-api/terms)). Use the free tier only with synthetic or public documents, like the sample file in `data/`. For real patient records, use a paid Gemini plan or Vertex AI with the right data agreements, and follow the laws that apply to you (for example HIPAA in the US or the DPDP Act in India).

## How it works

```
PDF files -> text by page -> one chunk per patient record -> embeddings (gemini-embedding-2)
          -> FAISS index -> top passages for a question -> Gemini answer with [1][2] citations
```

- **Chunking:** records are split on `Patient <n>:` headings, so each patient stays in one chunk, even across a page break. Other documents are split into paragraphs of about 1,200 characters.
- **Search:** Gemini embeddings and cosine similarity in FAISS. No PyTorch needed, so it installs on any laptop and the Docker image stays small. Passages below a minimum match score are dropped.
- **Answers:** Gemini answers only from the passages. If nothing relevant is found, it says so.
- **Offline search (optional):** set `RX_EMBED_BACKEND=local` and install `requirements-local.txt` to embed with a local MiniLM model instead. This needs PyTorch, and on Windows the Microsoft Visual C++ Redistributable. The app re-indexes automatically when you switch.

## Project structure

```
rxreveal/
  app.py                  Streamlit app (UI and theme)
  rxreveal/
    config.py             Settings from environment variables
    ingest.py             PDF reading and chunking
    index.py              Embeddings and FAISS index
    llm.py                Gemini client
    rag.py                Retrieve, then generate
    build_index.py        Command to build the index
  data/                   Put your PDFs here (sample file included, fake data)
  scripts/
    make_sample_pdf.py    Creates a PDF of fake patient records
    start.sh              Container start script
  tests/                  Unit tests (pytest)
  notebooks/              Original Kaggle notebook
  .streamlit/config.toml  Clean medical theme
  Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## 1. Run it in VS Code

You need Python 3.11 or 3.12 and VS Code with the Python extension.

1. Open the `rxreveal` folder in VS Code (**File > Open Folder**).
2. Open a terminal (**Terminal > New Terminal**) and create a virtual environment:

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   macOS or Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the packages:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey). Copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=`. The key is required, because search uses Gemini embeddings.

5. Add documents. A sample file of fake records is already in `data/`. To use the Kaggle dataset, download `synthetic_medical_records.pdf` from Kaggle and put it in `data/`.

6. Build the search index:
   ```bash
   python -m rxreveal.build_index
   ```
   It should print `Indexed 31 chunks from 1 file(s) ... using gemini:gemini-embedding-2:768`. The app also builds the index on first start if it is missing.

7. Start the app:
   ```bash
   streamlit run app.py
   ```
   It opens at http://localhost:8501. You can also press **F5** in VS Code and pick **Run RxReveal (Streamlit)**.

Run the tests with `pytest`, or from the Testing panel in VS Code.

## 2. Put it on GitHub

1. Create an empty repository on GitHub (no README, no .gitignore), for example `rxreveal`.
2. In the VS Code terminal:
   ```bash
   git init
   git add .
   git commit -m "RxReveal: Streamlit app with Gemini RAG"
   git branch -M main
   git remote add origin https://github.com/<your-username>/rxreveal.git
   git push -u origin main
   ```

`.gitignore` keeps your `.env` file, the built index and any uploaded PDFs out of GitHub. Only the fake sample PDF is committed. Every push runs the tests and a Docker build on GitHub Actions (see the **Actions** tab).

## 3. Run it with Docker

```bash
docker compose up --build
```

Open http://localhost:8501. The container reads your `.env` file, keeps `data/` on your machine, and builds the index on first start.

Without Compose:
```bash
docker build -t rxreveal .
docker run -p 8501:8501 --env-file .env rxreveal
```

## 4. Deploy to production

The image runs on any service that runs containers. It listens on the `PORT` environment variable (default 8501) and has a health check at `/_stcore/health`.

**Google Cloud Run** (good fit, since you already use Gemini):
```bash
gcloud run deploy rxreveal --source . --region asia-south1 \
  --memory 1Gi --allow-unauthenticated \
  --set-env-vars GEMINI_MODEL=gemini-3.8-flash,RX_ALLOW_UPLOAD=false \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,APP_PASSWORD=rxreveal-password:latest
```
Create the two secrets in Secret Manager first.

**Render or Railway:** create a new web service from your GitHub repository, choose Docker, and add `GEMINI_API_KEY` and `APP_PASSWORD` as environment variables. 1 GB of memory is enough.

### Production checklist

- Set `APP_PASSWORD` so only your team can use the app.
- Keep the API key in the host's secret settings, never in the code or the image.
- Decide how documents get in. On Cloud Run and similar services the disk is wiped on restart, so either bake approved PDFs into the image (put them in `data/` before building) or set `RX_ALLOW_UPLOAD=false` and attach persistent storage.
- Read the patient data note at the top before using real records.
- If Google renames the model, update `GEMINI_MODEL`. The app shows a clear message when a model is not found.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | none | Gemini key, used for search and answers |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Model that writes the answers |
| `RX_EMBED_BACKEND` | `gemini` | Search backend: `gemini` or `local` (offline, needs PyTorch) |
| `RX_GEMINI_EMBED_MODEL` | `gemini-embedding-2` | Gemini embedding model |
| `RX_EMBED_DIMENSION` | `768` | Size of the Gemini embeddings |
| `APP_PASSWORD` | none | Turns on a password screen |
| `RX_ALLOW_UPLOAD` | `true` | Allow PDF uploads in the browser |
| `RX_TOP_K` | `5` | Passages given to the model per question |
| `RX_MIN_SCORE` | `0.20` | Minimum match score for a passage |
| `RX_CHUNK_SIZE` | `1200` | Maximum characters per chunk |
| `RX_MAX_UPLOAD_MB` | `25` | Maximum upload size per file |
| `RX_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model (only with `RX_EMBED_BACKEND=local`) |

The app re-indexes automatically when the embedding settings change. If you change the chunk settings, click **Rebuild index** in the sidebar.

## Credits

Built on the Gen AI Intensive Course Capstone 2025Q1 by Google and Kaggle. Blog post: [Building a Medical Document Q&A Assistant with Gen AI and RAG](https://medium.com/@mdhaarishussain/building-a-medical-document-q-a-assistant-with-gen-ai-and-rag-77544b66aaa7).

RxReveal is for information only and does not give medical advice.
