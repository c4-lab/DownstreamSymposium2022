from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import time
from pyarrow import feather

model = SentenceTransformer('paraphrase-mpnet-base-v2')
print('Model loaded.')

# model = hub.load ("https://tfhub.dev/google/universal-sentence-encoder-large/5")
embedding_model = 'bert'
stime = time.time()
df = feather.read_feather('./re_sentences_feather.feather')
ftime = time.time()
print ('time used to load entire feather file: ' + str(ftime - stime) + 's')


# Sentences are encoded by calling model.encode()
stime = time.time()
embeddings = model.encode(df['sentence'][:10000000], batch_size = 100)
ftime = time.time()
print ('time used to encode 100000 sentences: ' + str(ftime - stime) + 's')
#embeddings = model (sentences_text)

embeddings_df = pd.DataFrame(embeddings)
embeddings_df['sentence_id'] = df['sentence_id']

stime = time.time()
with open('./test_embeddings.feather', 'wb') as f:
    feather.write_feather(embeddings_df, f)
ftime = time.time()
print ('time used to save ' + str(len(df)) + ' sentence embeddings: ' + str(ftime - stime) + 's')