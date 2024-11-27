# 📝 SQL Helper: Natural Language to SQL Conversion

An AI system that converts natural language to SQL queries using LangGraph and Streamlit. Built with Qwen model for intuitive database interactions.

## Core Components
- Frontend: Streamlit web interface
- Backend: LangGraph query generation
- RAG: Context-aware response system
- Database: MySQL integration

## System Requirements

### Hardware (Mandatory)
- NVIDIA GPU
  - Minimum: NVIDIA L4
  - VRAM: 23GB+ (Tested: 23034MiB)
  - CUDA Version: 12.2
  - Driver Version: 535.183.01+

### Software
- Python 3.8+
- MySQL Database
- CUDA Toolkit 12.2

## Key Dependencies
```
faiss-cpu         1.9.0
fastapi           0.115.5
langchain-community 0.3.2
langgraph         0.2.37
PyMySQL           1.1.1
torchaudio        2.5.1+cu121
torchvision       0.20.1+cu121
unsloth           2024.11.11
uvicorn           0.32.1
```

## Installation and Setup
1. Clone and Setup:
```
git clone <repository-url>
cd sql-helper
chmod +x frontend_env_setup.sh
./frontend_env_setup.sh
```
2. Environment Setup Script (frontend_env_setup.sh):
```
#!/bin/bash

pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip && pip install unsloth && echo "Backend setup complete!"
```

3. Configure Environment Variables:
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
