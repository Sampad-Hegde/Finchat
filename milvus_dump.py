import pandas as pd
from pymilvus import MilvusClient
import os

collection_name = 'annual_reports'

client = MilvusClient(host=os.getenv('MILVUS_HOST', 'localhost'),
    port=int(os.getenv('MILVUS_PORT', 19530)),
    timeout=5)

query = 'id > 0'
df = pd.DataFrame()

batch = 1
while True:
    data = client.query(
        collection_name=collection_name,
        filter=query,
        output_fields=['symbol', 'year', 'text', 'embedding'],
        limit=10000
    )
    if len(data) == 0:
        break
    temp_df = pd.DataFrame(data)
    query = f'id > {data[-1]["id"]}'
    df = pd.concat([df, temp_df])
    print(f'Batch {batch} done DF count {len(df)}')
    if batch % 50 == 0:
        df = df[['symbol', 'year', 'text', 'embedding']]
        df.to_csv(f'milvus_dump{batch//50}.csv', index=False)
        df = pd.DataFrame()
    batch += 1

df = df[['symbol', 'year', 'text', 'embedding']]
df.to_csv(f'milvus_dump_final.csv', index=False)