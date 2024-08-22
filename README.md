# AutoClassed: Automated Supplier and Item Classification System

## Introduction

AutoClassed is an intelligent system designed to automatically classify suppliers and items in a procurement database. It uses advanced AI and web search capabilities to gather information, assign classification codes, and update your database with minimal human intervention.

## Features

- Automated classification of suppliers and items
- Integration with OpenAI's GPT model for intelligent processing
- Web search functionality using Google Serper API
- Concurrent processing for improved efficiency
- Compatible with SQLite and PostgreSQL databases

## Prerequisites

Before you begin, ensure you have the following:

1. A computer with internet access
2. Python 3.11 or higher installed (if you don't have it, download it from [python.org](https://www.python.org/downloads/))
3. Access to your procurement database (SQLite or PostgreSQL)
4. API keys for OpenAI and Google Serper (instructions for obtaining these are provided below)

## Installation

1. Open your computer's terminal or command prompt.

2. Clone the repository by typing:
   ```
   git clone https://github.com/your-username/autoclassed.git
   ```

3. Navigate to the project directory:
   ```
   cd autoclassed
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a file named `.env` in the project directory.

2. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```

3. If you're using a PostgreSQL database, add your database credentials to the `.env` file:
   ```
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   DB_NAME=your_database_name
   DB_USER=your_database_username
   DB_PASSWORD=your_database_password
   ```

## Usage

1. To process suppliers:
   ```
   python agent_company.py
   ```

2. To process items:
   ```
   python agent_item.py
   ```

3. To process both suppliers and items:
   ```
   python agent_combo.py
   ```

4. To view the results in a web interface:
   ```
   streamlit run main.py
   ```
   Then open your web browser and go to the URL displayed in the terminal (usually http://localhost:8501).

## Obtaining API Keys

### OpenAI API Key
1. Go to [OpenAI's website](https://openai.com/)
2. Sign up or log in
3. Navigate to the API section
4. Generate a new API key

### Google Serper API Key
1. Visit [Serper.dev](https://serper.dev/)
2. Sign up for an account
3. Navigate to the API Keys section
4. Generate a new API key

## Troubleshooting

If you encounter any issues:
1. Ensure all prerequisites are met
2. Check that your API keys are correctly entered in the `.env` file
3. Verify your database connection details
4. If problems persist, please open an issue on our GitHub repository

## Support

For additional help or questions, please open an issue on the GitHub repository.

Thank you for using AutoClassed! We hope this tool significantly improves your procurement data management process.
