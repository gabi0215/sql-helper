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
