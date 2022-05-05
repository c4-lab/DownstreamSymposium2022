from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import time
from pyarrow import feather
import os
from dataloader.load import load_batch

model = SentenceTransformer('paraphrase-mpnet-base-v2')
print('Model loaded.')

testing = False
sentence = False

if testing == True:
    idx, tweets = load_batch(batch_size = 1000)

    # Sentences are encoded by calling model.encode()
    stime = time.time()
    embeddings = model.encode(tweets, batch_size = 200)
    ftime = time.time()
    print ('time used to encode ' + str(len(idx)) + ' sentences: ' + str(ftime - stime) + 's')
    #embeddings = model (sentences_text)

    df = pd.DataFrame(embeddings)
    df['sentence_id'] = idx

    stime = time.time()
    with open('./test_embeddings.feather', 'wb') as f:
        feather.write_feather(df, f)
    ftime = time.time()
    print ('time used to save ' + str(len(idx)) + ' sentence embeddings: ' + str(ftime - stime) + 's')

if sentence == True:
    df = pd.read_feather('/home/lliang06/Documents/belief_landscape_model/src/props.feather')
    batch_size = 1000000
    for i in range(1, 5):
        print('embedding batch ' + str(i) + '...')
        props = df.iloc[i * batch_size:(i + 1) * batch_size, 1]
        ids = df.iloc[i * batch_size:(i + 1) * batch_size, 0].values
        embeddings = model.encode(props.values, batch_size = 200)

        output = pd.DataFrame(embeddings)
        output['sentence_id'] = ids
        with open('./prop_embeddings' + str(i) + '.feather', 'wb') as f:
            feather.write_feather(output, f)


# for f in os.listdir('../SubjData'):
#     n = f.split('.')[0]
#     print('embedding ' + n)
#     df = pd.read_feather('../SubjData/' + f)
#     embeddings = model.encode(df.prep.values, batch_size = 200)
#     output = pd.DataFrame(embeddings)
#     output['HashId'] = df.HashId
#     output['obj'] = df.obj
#     with open('./' + n + '_embedding.feather', 'wb') as f:
#         feather.write_feather(output, f)

if __name__ == "__main__":
    sentence = True