# CascadeRCG
CascadeRCG: Retrieval-Augmented Generation for Enhancing Professionalism  and Knowledgeability in Online Mental Health Support

[Paper Link](https://dl.acm.org/doi/10.1145/3701716.3715466)

![Evaluation Criteria](./images/Figure.png)
## :cherry_blossom: Vector Database
### Features

1. **Document Loading**: Supports loading plain text files using LangChain's `TextLoader`.
2. **Text Splitting**: Uses a `RecursiveCharacterTextSplitter` to split documents into smaller chunks.
3. **Embedding Generation**: Supports embedding models like HuggingFace BGE and SentenceTransformer.
4. **Storage**: Stores embeddings in a Chroma vector database.
### Example
```bash
python construct_vector_db.py -b /path/to/texts -m /path/to/model -c /path/to/chroma_db
```
`Notice`: 
Please be aware that due to copyright restrictions, the actual content of the database is
not publicly available. However, our list of book names can refer to the `books_list/list.json` file.

## :relaxed: Generation
```bash
pip install -r requirements.txt
python main.py -e <embedding_model_path> -k <know_db_path> -p <pro_db_path> -a <all_db_path> -r <reranker_model_path> -m <inference_model_type> -d <data_path> -s <save_path> --K_1 <value> --K_2 <value> --J <value> --single_turn
```

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
```

```bibtex
@inproceedings{10.1145/3701716.3715466,
author = {Yang, Di and Zhu, Jingwei and Wu, Haihong and Tan, Minghuan and Li, Chengming and Yang, Min},
title = {CascadeRCG: Retrieval-Augmented Generation for Enhancing Professionalism and Knowledgeability in Online Mental Health Support},
year = {2025},
isbn = {9798400713316},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3701716.3715466},
doi = {10.1145/3701716.3715466},
abstract = {Online mental health support(OMHS) plays a crucial role in promoting well-being, but the shortage of mental health professionals necessitates automated systems to address complex care needs. While large language models (LLMs) are widely adopted, they often fall short in OMHS settings due to the complexity and ambiguity of the questions posed. Additionally, providing accurate answers requires extensive knowledge, which LLMs may lack, leading to responses that often lack depth, professionalism, and critical detail. To address these limitations, we introduce a new task tailored to OMHS scenarios, focusing on enhancing the professionalism and knowledgeability of generated responses. Furthermore, we propose a comprehensive benchmark designed to systematically evaluate the quality of responses. Building on these foundations, we propose the CascadeRCG framework, an optimized approach based on Retrieval-Augmented Generation (RAG). This framework first employs a knowledge management strategy, then introduces a two-stage cross-iterative Retrieval mechanism and a Clustering-then-summarizing module, followed by the final Generation stage. Experimental results on both single-turn and multi-turn psychological dialogue datasets, compared to other RAG-based baselines across different LLMs, show significant improvements in response professionalism and knowledge depth. This enhancement in response quality provides an effective methodology and strategy for further improving OMHS systems. Our code is available at https://github.com/CAS-SIAT-XinHai/CascadeRCG.},
booktitle = {Companion Proceedings of the ACM on Web Conference 2025},
pages = {1465–1469},
numpages = {5},
keywords = {LLM, NLP, RAG, online mental health support},
location = {Sydney NSW, Australia},
series = {WWW '25}
}
```
