from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import time
from pyarrow import feather
from dataloader.load import load_batch
from torch.utils.data import DataLoader, TensorDataset

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
idx, sentences = load_batch(batch_size = 10000)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

model.cuda()
# encoded_input = {
#     'input_ids': encoded_input['input_ids'].cuda(),
#     'attention_mask': encoded_input['attention_mask'].cuda(),
# }

sentence_loader = DataLoader(TensorDataset(encoded_input['input_ids'], 
                                           encoded_input['attention_mask']), 
                                           batch_size = 50)

stime = time.time()
for batch in sentence_loader:
    batch = tuple(t.cuda() for t in batch)
    batch_output = model(batch[0], batch[1])
ftime = time.time()
print ('time used to encode ' + str(len(idx)) + ' sentences: ' + str(ftime - stime) + 's')

# output = []
# batch_size = 100

# if len(idx) % 100 != 0:
#     print('change a batch size!')
# else:
#     # Compute token embeddings
#     stime = time.time()
#     n_iter = int(len(idx) / batch_size)
#     with torch.no_grad():
#         for i in range(n_iter):
#             sstime = time.time()
#             batch = {
#                 'input_ids': encoded_input['input_ids'][i * batch_size:(i + 1) * batch_size].cuda(),
#                 'attention_mask': encoded_input['attention_mask'][i * batch_size:(i + 1) * batch_size].cuda(),
#             }
#             # stime = time.time()
#             model_output = model(**batch)
#             fftime = time.time()
#             print ('The ' + str(i) + ' iteration took: ' + str(fftime - sstime) + 's')
#     ftime = time.time()
#     print ('time used to encode ' + str(len(idx)) + ' sentences: ' + str(ftime - stime) + 's')

# Perform pooling. In this case, max pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# stime = time.time()
# with open('./test_embeddings_huggingface.feather', 'wb') as f:
#     feather.write_feather(pd.DataFrame({'sentence_id': idx, 'sentence': sentence_embeddings,}), f)
# ftime = time.time()
# print ('time used to save ' + str(len(idx)) + ' sentence embeddings: ' + str(ftime - stime) + 's')