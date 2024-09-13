import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from nifty_50 import nifty_mappings
import torch

milvus_client = MilvusClient(host='localhost', port=19530, timeout=5000)  #
collection_name = 'annual_reports'

print('Milvus Collection Status : ', milvus_client.has_collection(collection_name))

if not milvus_client.has_collection(collection_name):
    print(f'Creating collection {collection_name} ...')
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="symbol", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="year", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields=fields, description="annual_reports", partition_key_field="symbol")
    milvus_client.create_collection(collection_name, schema=schema, dimension=768, metric_type="HNSW")
    milvus_client.load_collection(collection_name=collection_name)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')



def get_embeddings(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=512, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    embeddings = embedding_model.encode([docs[i].page_content for i in range(len(docs))], batch_size=64,
                                        show_progress_bar=True, convert_to_tensor=True)
    return embeddings, docs


def ingest_doc(embeddings, docs, symbol, fy):
    data = [{
        "symbol": symbol,
        "year": fy,
        "text": docs[i].page_content,
        "embedding": embeddings[i]
    } for i in range(len(embeddings))]

    milvus_client.insert(collection_name, data)


def get_nifty_10_yrs_pdfs():
    pdf_df = pd.read_csv('/mnt/ssd/projects/Finchat/Data/bse_metadata.csv')
    pdf_df = pdf_df[pdf_df['bse_code'].isin(list(nifty_mappings.values()))]
    pdf_df['financial_year'] = pdf_df['financial_year'].astype(int)
    pdf_df = pdf_df[pdf_df['financial_year'] >= 2014]
    return pdf_df


def get_symbol_for_bse_code(bse_code):
    for key, value in nifty_mappings.items():
        if value == bse_code:
            return key


def check_if_already_ingested(symbol, fy):
    query = f'symbol == "{symbol}" and year == {fy}'
    result = milvus_client.query(
        collection_name,
        filter=query,
        data=[],
        output_fields=["id"],
        limit=1
    )
    if len(result) > 0:
        return True
    else:
        return False


df = get_nifty_10_yrs_pdfs()

for index, row in df.iterrows():
    symbol = get_symbol_for_bse_code(row['bse_code'])
    fy = row['financial_year']
    file_path = f'/mnt/ssd/projects/Finchat/Data/AnnualReports/{row['bse_code']}/{fy}.pdf'
    if not check_if_already_ingested(symbol, fy):
        print(f'Digesting {symbol} - {fy}', end='\t')
        try:
            embeddings, docs = get_embeddings(file_path)
        except Exception as e:
            print(f'************>>>>> Exception occurred durin reading {symbol}-{fy} File: {file_path} - {e}')
            continue
        print(f'Ingesting {symbol} - {fy}', end='\t')
        ingest_doc(embeddings, docs, symbol, fy)
        print(f'Completed {symbol} - {fy}', end='\t')
    else:
        print(f'Already ingested {symbol} - {fy}')
