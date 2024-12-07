# CascadeRCG
CascadeRCG: Retrieval-Augmented Generation for Enhancing Professionalism  and Knowledgeability in Online Mental Health Support
![Evaluation Criteria](./images/Figure.png)
## :relaxed: Generation
```bash
pip install -r requirements.txt
python main.py -e <embedding_model_path> -k <know_db_path> -p <pro_db_path> -a <all_db_path> -r <reranker_model_path> -m <inference_model_type> -d <data_path> -s <save_path> --K_1 <value> --K_2 <value> --J <value> --single_turn
```
**Notice**: 
Please be aware that due to copyright restrictions, the actual content of the database is
not publicly available.

## :mag: Evaluation 
### Criteria:
![Evaluation Criteria](./images/evaluation.png)

### Evaluation Steps:
This tool evaluates data using the GPT-4 model. It supports two types of evaluations: "ethics" and "rag".

#### Setup

1. **Set Up Environment Variables**

   Before running the tool, you need to set the following environment variables:

   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `OPENAI_API_BASE`: The base URL for the OpenAI API.

   You can set these variables in your terminal or command prompt:

   ```bash
   export OPENAI_API_KEY='your-api-key'
   export OPENAI_API_BASE='https://api.openai.com'


#### Usage

**Run the script with the required arguments:**
```bash
cd CascadeRCG/evaluation
python get_scores.py -e /path/to/evaluation_data.json -t [ethics|rag] -r /path/to/results.json
