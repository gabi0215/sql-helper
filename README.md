# 📝 SQL Helper: Natural Language ↔ SQL Interactive System

An AI system that provides:
- Natural Language → SQL: Convert user questions to SQL queries
- SQL → Natural Language: Explain query results in plain language
- Interactive Refinement: Multi-turn conversation to improve query accuracy

## Core Components
- Frontend: Streamlit web interface
- Backend: LangGraph query generation
- RAG: Database schema-aware system using FAISS vector store
  - Indexes table structures and relationships
  - Uses metadata for precise SQL generation
  - Enables context-aware query refinement
- Database: MySQL integration

## System Architecture
- Backend Server (GPU)
  - NVIDIA L4 GPU minimum
  - VRAM: 23GB+ (Tested: 23034MiB)
  - CUDA Version: 12.2
  - Driver Version: 535.183.01+
  - Purpose: LLM processing & SQL generation

- Frontend Server (CPU)
  - Standard CPU instance
  - Memory: 8GB+ recommended
  - Purpose: Web interface & user interactions

## Required Open Ports
- Port 8501: Streamlit web interface access
- Port 8000: Backend FastAPI server access 
- Port 3306: MySQL database connection


# Installation and Setup
## Frontend Setup Guide

0. Clone and start application:
```bash
git clone https://github.com/100suping/sql-helper.git
```

1. Run environment setup script:
```bash
cd sql-helper/frontend
chmod +x frontend_env_setup.sh
./frontend_env_setup.sh
```

2. Open new terminal and activate environment:
```bash
pyenv activate frontend
```

3. Start application:
```   
cd sql-helper/frontend
pip install -r requirements.txt
streamlit run app.py
```

## Backend Setup Guide
## GPU Server Initial Setup

1. Install CUDA and NVIDIA drivers:
```bash
cd sql-helper/backend/GPUsetting
chmod +x cuda_install.sh
./cuda_install.sh
# System will reboot
```
Note: Server requires reboot after CUDA installation. Ensure all commands are executed in order.

2. After reboot, install PyEnv dependencies:
```bash
cd sql-helper/backend/GPUsetting
chmod +x pyenv_dependencies.sh
./pyenv_dependencies.sh
```
3. Setup PyEnv:
```bash
chmod +x pyenv_setup.sh
./pyenv_setup.sh
```

4.Create Python environment:
```bash
chmod +x pyenv_virtualenv.sh
./pyenv_virtualenv.sh
# Enter Python version: 3.11.8
# Enter environment name: backend
```
5. Verify GPU setup:
```bash
nvidia-smi
# Should show NVIDIA L4 GPU info
```

## Backend Application Setup
1. Setup backend application:
```bash
cd sql-helper/backend
chmod +x backend_env_setup.sh
./backend_env_setup.sh
```

2. Configure environment variables:
Create .env file in project root:
```
OPENAI_API_KEY="your-api-key"
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT="your-project-name"
URL="your-mysql-database-url"
HUGGINGFACE_TOKEN='your-huggingface-token"
```

3. Start backend:
```
python main.py
```

## Project Structure
```
sql-helper/
├── frontend/
│   ├── app.py                # Streamlit interface
│   ├── requirements.txt      # Frontend dependencies
│   ├── README.md            # Frontend docs
│   └── frontend_env_setup.sh # Frontend setup script
├── backend/
│   ├── GPUsetting/          # GPU/CUDA setup scripts
│   │   ├── cuda_install.sh
│   │   ├── pyenv_dependencies.sh
│   │   ├── pyenv_setup.sh
│   │   └── pyenv_virtualenv.sh
│   ├── langgraph_/          # Core backend logic
│   │   ├── init.py
│   │   ├── faiss_init.py
│   │   ├── graph.py
│   │   ├── node.py
│   │   ├── task.py
│   │   └── utils.py
│   ├── prompts/             # LLM prompts
│   │   ├── additional_question/
│   │   ├── general_conversation/
│   │   ├── query_creation/
│   │   ├── question_analysis/
│   │   ├── question_evaluation/
│   │   ├── question_refinement/
│   │   ├── sql_conversation/
│   │   └── table_selection/
│   ├── backend_env_setup.sh # Backend setup script
│   ├── main.py             # Backend entry point
│   ├── README.md           # Backend docs
│   └── requirements.txt    # Backend dependencies
├── .env                    # Environment variables for backend
├── README.md              # Project documentation
└── .gitignore

Note: `.env` file should be placed in project root and backend directory needs access to it for database and API configurations.

```


