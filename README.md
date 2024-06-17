# Finchat

Finchat is an innovative financial chatbot application designed to provide users with accurate and insightful financial information and advice. This project leverages advanced technologies, including Retrieval-Augmented Generation (RAG) using Ollama, an intuitive user interface built with Streamlit, and efficient vector storage and search powered by Milvus.

### Key Features

- **Advanced Financial Insights**: Finchat uses state-of-the-art RAG techniques to deliver precise and contextually relevant financial information.
- **User-Friendly Interface**: Developed with Streamlit, Finchat offers a seamless and interactive user experience, making complex financial data accessible to all.
- **Efficient Data Handling**: The Milvus vector database ensures fast and accurate retrieval of financial documents and data, enhancing the chatbot's performance and reliability.

### Technology Stack

- **Ollama**: For Managing the LLM models and provides Python client for interface with LLm models.
- **Streamlit**: To create an engaging and interactive web application interface.
- **Milvus**: A high-performance vector database for managing and querying large-scale vector data efficiently.

---

### Data Flow

1. **User Query**: The user inputs a financial question or query through the Streamlit interface.
2. **Query Processing**: The query is sent to the Ollama component, where it is analyzed and broken down.
3. **Vector Search**: Ollama uses the processed query to perform a vector search in the Milvus database, retrieving relevant financial documents or data.
4. **Response Generation**: The retrieved information is used by Ollama to generate a precise and contextually relevant response.
5. **Display Response**: The generated response is displayed to the user through the Streamlit interface.

---

## Installation and Replication

- PIP dependencies

```shell
pip3 install -r requirements.txt
```
- Milvus Vector database
  - You can also follow this from their website: https://milvus.io/docs/install_standalone-docker-compose.md
```shell
docker-compose up -d
(or)
docker compose up -d   
```

- Data Downloading

```python
from collect_ay_pdf import FYReport

downloader = FYReport()
downloader.download_fy_report(
    bse_code='500035',
    2014,
    2023,
    'Data/AnnualReports'
)
```
The above code has dependency on `Bhart-sm-data` pip package (It's als0 created by me; checkout git/pypi for more info)
install using below command (ignore if all packages are installed using `requirements.txt`)
```shell
pip3 install Bharat-sm-data
```

The above code creates PDF files in this folder `./Data/AnnualReports/500035/` and PDF files are named with {year}.pdf in that folder.

- You can construct the metadata for just ease of tracking and using pandas DFs (Completely Optional; make adjustments accordingly in `ingest_doc.py` in App folder in that case)

    Example for above DF (or) CSV can be found in Data/bse_metadata.csv

- Run the `ingest.py` for creating the Collection in milvus and chunking of the pdf and get the embedding for the text chunk and save it to **Milvus Vector Store**

- Install the Ollama and full the `phi3` model Instructions are [here](https://www.ollama.com/library/phi3)
- start the streamlit server Using
```shell
streamlit run App/app.py
```

---

## Note: This Code is completely free to tune according to our need; Happy Coding :)





