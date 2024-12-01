# GPT-4 Evaluation Tool

This tool evaluates data using the GPT-4 model. It supports two types of evaluations: "ethics" and "rag".

## Setup

1. **Set Up Environment Variables**

   Before running the tool, you need to set the following environment variables:

   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `OPENAI_API_BASE`: The base URL for the OpenAI API.

   You can set these variables in your terminal or command prompt:

   ```bash
   export OPENAI_API_KEY='your-api-key'
   export OPENAI_API_BASE='https://api.openai.com'


## Usage

**Run the script with the required arguments:**
```bash
python evaluation_tool.py -e /path/to/evaluation_data.json -t [ethics|rag] -r /path/to/results.json
