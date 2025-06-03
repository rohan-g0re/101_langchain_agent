# Project Title

Simple agent with tools for responding to simnple queries.

## Setup Instructions

### 1. Create a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**For Windows:**

```bash
.\venv\Scripts\activate
```

**For macOS and Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Agent

python simple_agent.py

## Project Structure

```
.
├── .git/
├── .gitignore
├── simple_agent.py
├── knowledge_base/
├── venv/
├── requirements.txt
└── README.md
``` 

Add text files to the knowledge_base folder --> to use it as an additional tool.