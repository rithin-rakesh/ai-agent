# AI Job Application Agent

Autonomous, intelligent job discovery, matching, autofill, and application submission platform.

---

## 1. Project Purpose

The **AI Job Application Agent** is designed to streamline and automate the end-to-end job search and application lifecycle. It empowers job seekers by discovering relevant opportunities across multiple job boards, performing LLM-assisted compatibility scoring against a comprehensive user profile, auto-filling complex application forms, and submitting applications while recording structured audit logs and telemetry.

---

## 2. Phase 1 Architecture

Phase 1 focuses entirely on building the foundational core, database schema, data models, configuration system, API health check, and containerization scaffolding.

```
AI-job-application-agent/
│
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Pydantic Settings with SecretStr protection
│   ├── database/
│   │   ├── __init__.py
│   │   └── supabase.py          # Supabase client factory (zero business logic)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── profile.py           # Profile & Skill domain schemas
│   │   ├── job.py               # Job domain schema with unique constraint
│   │   ├── match.py             # Match scoring & decision schemas
│   │   └── application.py       # Application, Q&A, and Automation log schemas
│   └── api/
│       ├── __init__.py
│       └── main.py              # FastAPI application & health endpoints
│
├── sql/
│   └── schema.sql               # PostgreSQL DDL for Supabase
│
├── tests/
│   ├── __init__.py
│   └── test_health.py           # pytest test suite
│
├── data/
│   ├── profile/                 # Profile JSON configurations
│   └── resumes/                 # Resume files (PDF/DOCX)
│
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Core Architecture Components:
- **FastAPI Layer**: Exposes root `/` and `/health` endpoints with structured JSON responses.
- **Pydantic Settings**: Strongly typed configuration loaded from `.env` files with secret masking.
- **Supabase Integration**: Isolated database client manager and PostgreSQL schema with UUIDs, foreign keys, cascade rules, indexing, and uniqueness constraints.
- **Domain Models**: Pydantic v2 schemas providing data validation and serialization.

---

## 3. Local Python Setup

### Prerequisites
- Python 3.11+ (Python 3.11, 3.12, 3.13, 3.14)
- Git

### Installation Steps

1. **Clone or Navigate to the repository**:
   ```bash
   cd AI-job-application-agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install project dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Environment Variable Setup

Copy `.env.example` to create your local `.env` file:

```bash
cp .env.example .env
```

### Supported Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `SUPABASE_URL` | `string` | `""` | Supabase Project URL (`https://xyz.supabase.co`) |
| `SUPABASE_ANON_KEY` | `string` | `""` | Supabase Public / Anon API Key |
| `SUPABASE_SERVICE_ROLE_KEY`| `secret` | `""` | Supabase Service Role Secret Key (never logged) |
| `APP_ENV` | `string` | `development` | Application environment (`development`, `staging`, `production`, `test`) |
| `APP_HOST` | `string` | `0.0.0.0` | Server host binding |
| `APP_PORT` | `integer`| `8000` | Server port |
| `LOG_LEVEL` | `string` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 5. How to Start FastAPI

### Running with Uvicorn CLI
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running directly via Python
```bash
python -m app.api.main
```

### Verifying Endpoints
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
  ```json
  {
      "status": "ok",
      "environment": "development"
  }
  ```
- **Root Status**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 6. How to Run Tests

The test suite validates endpoints, settings, and database client isolation without requiring active Supabase credentials.

Run tests using `pytest`:
```bash
pytest tests/ -v
```

Run tests with test coverage:
```bash
pytest tests/ -v --tb=short
```

---

## 7. How to Run Docker

### Using Docker Compose (Recommended)
```bash
# Build and start container in the background
docker compose up -d --build

# View container logs
docker compose logs -f

# Stop container
docker compose down
```

### Using Plain Docker
```bash
# Build image
docker build -t ai-job-agent .

# Run container
docker run -d -p 8000:8000 --env-file .env --name ai-job-agent-app ai-job-agent
```

---

## 8. How to Connect the Project to Supabase

1. **Create a Supabase Project**:
   - Log in to [Supabase](https://supabase.com/).
   - Click **New Project** and configure your organization and project name.

2. **Execute Database Schema**:
   - Open your project dashboard and navigate to the **SQL Editor**.
   - Open [`sql/schema.sql`](sql/schema.sql).
   - Copy and paste the contents into the SQL Editor and click **Run**.
   - This creates 7 tables (`profiles`, `skills`, `jobs`, `job_matches`, `applications`, `application_answers`, `automation_logs`), indexes, and triggers.

3. **Configure API Keys**:
   - Navigate to **Project Settings > API**.
   - Copy **Project URL**, **anon public key**, and **service_role secret key**.
   - Add these values to your `.env` file:
     ```env
     SUPABASE_URL=https://<your-project-ref>.supabase.co
     SUPABASE_ANON_KEY=<your-anon-key>
     SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
     ```

---

## 9. What Phase 1 Does NOT Contain Yet

Per design and scope boundaries, Phase 1 strictly excludes:
- ❌ JobSpy integration & external job scrapers
- ❌ Job discovery workflows & automated crawlers
- ❌ NVIDIA LLM / OpenAI / Anthropic integrations
- ❌ Algorithmic & LLM job matching pipelines
- ❌ Playwright / Selenium browser automation
- ❌ `pywinauto` desktop automation
- ❌ Browser session / cookie management & login workflows
- ❌ Automatic application submission agents

These capabilities will be introduced in subsequent modular phases.
