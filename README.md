# 📝 SQL Helper: Natural Language to SQL Conversion

An AI system that converts natural language to SQL queries using LangGraph and Streamlit. Built with Qwen model for intuitive database interactions.

## Core Components
- Frontend: Streamlit web interface
- Backend: LangGraph query generation
- RAG: Context-aware response system
- Database: MySQL integration

### System Architecture
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

### Required Open Ports
- Port 8501: Streamlit web interface access
- Port 8000: Backend FastAPI server access 
- Port 3306: MySQL database connection


## Installation and Setup
## Frontend Setup Guide

1. Run environment setup script:
```bash
chmod +x frontend_env_setup.sh
./frontend_env_setup.sh
```

2. Open new terminal and activate environment:
```bash
pyenv activate frontend
```
3. Clone and start application:
```bash
git clone -b refactor https://github.com/100suping/sql-helper.git
cd sql-helper/frontend
pip install -r requirements.txt
streamlit run app.py
```

## Backend Setup Guide
## GPU Server Initial Setup

1. Install CUDA and NVIDIA drivers:
```bash
chmod +x cuda_install.sh
./cuda_install.sh
# System will reboot
```
2. After reboot, install PyEnv dependencies:
```bash
chmod +x pyenv_dependencies.sh
./pyenv_dependencies.sh
```
3. Setup PyEnv:
```bash
Copychmod +x pyenv_setup.sh
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

6. Setup backend application:
```bash
pyenv activate backend
git clone -b refactor https://github.com/100suping/sql-helper.git
cd sql-helper/backend
chmod +x backend_env_setup.sh
./backend_env_setup.sh
python main.py

```
Note: Server requires reboot after CUDA installation. Ensure all commands are executed in order.



## Configure Environment Variables:
Create .env file:

```
OPENAI_API_KEY="your-api-key"
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT="your-project-name"
URL="your-mysql-database-url"
```

## Project Structure
```
sql-helper/
├── frontend/                  # Frontend application
│   ├── app.py                # Streamlit web interface
│   ├── requirements.txt      # Frontend dependencies
│   └── README.md            # Frontend documentation
│
├── backend/                  # Backend services
│   └── langgraph_           # LangGraph implementation
│       ├── init.py        # Package initialization
│       ├── faiss_init.py    # Vector database initialization
│       ├── graph.py         # Conversation flow control
│       ├── node.py          # Graph node definitions
│       ├── task.py          # Task implementations
│       └── utils.py         # Utility functions
│
├── prompts/                  # LLM prompt templates
│   ├── additional_question/  # Follow-up question prompts
│   ├── general_conversation/ # Basic conversation handling
│   ├── query_creation/      # SQL generation prompts
│   ├── question_analysis/   # Query intent analysis
│   ├── question_evaluation/ # Input quality check
│   ├── question_refinement/ # Query improvement
│   ├── sql_conversation/    # SQL results discussion
│   └── table_selection/     # DB table selection logic
│
├── frontend_env_setup.sh     # Environment setup script
├── main.py                  # Application entry point
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
└── .gitignore              # Git ignore rules
```
