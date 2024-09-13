from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
from torch.cuda import is_available


class Search:
    def __init__(self, collection_name: str = 'annual_reports'):
        print("Milvus Host: ", os.getenv('MILVUS_HOST', 'localhost'), "Milvus POrt : ", int(os.getenv('MILVUS_PORT', 19530)))
        if os.getenv('MILVUS_USER', '') == '' or os.getenv('MILVUS_PASSWORD', '') == '':
            print('Milvus User or Password not set; using the default creds')
            self.milvus_client = MilvusClient(uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{int(os.getenv('MILVUS_PORT', '19530'))}", timeout=5)
        else:
            self.milvus_client = MilvusClient(uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{int(os.getenv('MILVUS_PORT', '19530'))}",
                                              user=os.getenv('MILVUS_USER', ''),
                                              password=os.getenv('MILVUS_PASSWORD', ''),
                                              timeout=5)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',
                                                   device='cuda' if is_available() else 'cpu')
        self.collection_name = collection_name

    def embed(self, text: str):
        return self.embedding_model.encode(text).tolist()

    def get_similar_docs(self, query_text: str, symbol: str = None, fy: int = None):
        if fy is None and symbol is None:
            query = ''
        elif fy is None:
            query = f"symbol == '{symbol}'"
        elif symbol is None:
            query = f"year == {fy}"
        else:
            query = f"symbol == '{symbol}' and year == {fy}"

        print('Milvus Query: ', query)

        embeds = self.embed(query_text)

        result = self.milvus_client.search(
            self.collection_name,
            filter=query,
            data=[embeds],
            output_fields=["text"],
            limit=5
        )[0]

        context_text = '\n\n'.join(r.get('entity', {}).get('text', '') for r in result)
        return context_text
