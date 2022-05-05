import time
from pyarrow import feather
from embedding.dataloader import aggregate
import os
from pyarrow import feather
import pandas as pd


def load_batch(batch_num = 0, batch_size = 100, fdir = '../data/sentences/re_sentences.txt'):
    idx = []
    sentences = []
    stime = time.time()
    start = batch_num * batch_size

    with open(fdir, 'r') as f:
        output = f.readlines()[start:start + batch_size]

    ftime = time.time()
    print('time used to load ' + str(batch_size) + ' lines of sentence: ' + str(ftime - stime) + 's')
    
    for o in output:
        i, s = o.split('\t')
        idx.append(i)
        sentences.append(s.replace('\n', ''))
    return idx, sentences
        
def time_load_feather_1m():
    stime = time.time()
    feather_df = feather.read_feather('./sample_sentence.feather')
    ftime = time.time()
    print('time used to load 1000000 lines of sentence with feather: ' + str(ftime - stime) + 's')
    return(feather_df)

def load_all(fdir, save_dir):
    ids = []
    props = []
    for fname in os.listdir(fdir):
        with open(fdir + fname, 'r') as f:
            output = f.readlines()
        for o in output:
            id, prop = aggregate.cleanProp(o)
            ids.append(id)
            props.append(prop)
    df = pd.DataFrame({'ids': ids, 'props': props})
    with open(save_dir + 'props.feather', 'wb') as f:
        feather.write_feather(df, f)


          

