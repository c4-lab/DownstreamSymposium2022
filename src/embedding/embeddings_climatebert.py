import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader

torch.set_grad_enabled(False)

# Store the model we want to use
MODEL_NAME = "climatebert/distilroberta-base-climate-f"

def embed_list(l):
    # input have to be a list of strings
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print('model loaded...')
    
    
    encoded = tokenizer.batch_encode_plus(l,
                                            return_attention_mask=True,
                                            padding=True,
                                            return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    print('text tokenized!')
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=1)
    print('Getting embeddings...')
    outputs = []    
    progress_bar = tqdm(dataloader, leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()
        # outputs.append(model(batch[0], batch[1]).pooler_output.cpu().detach().numpy())
        outputs.append(model(batch[0], batch[1]).last_hidden_state.mean(axis = 1).cpu().detach().numpy())
    return(np.array(outputs).reshape(len(l), 768))
    
# if __name__ == "__main__":
#     # We need to create the model and tokenizer
#     model = AutoModel.from_pretrained(MODEL_NAME)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     print('model loaded...')
#     print('loading dataset...')
#     df = pd.read_feather('../props.feather')
#     print('dataset loaded!')
#     for i in range(5):
#         tmp = df.iloc[1000000*i:1000000*(i + 1), :]
#         print('processing part: ' + str(i + 1))

#         encoded = tokenizer.batch_encode_plus(tmp['props'].values.tolist(),
#                                                     return_attention_mask=True,
#                                                     padding=True,
#                                                     return_tensors='pt')

#         input_ids = encoded['input_ids']
#         attention_masks = encoded['attention_mask']
#         print('text tokenized!')
#         dataset = TensorDataset(input_ids, attention_masks)

#         batch_size = 200
#         dataloader = DataLoader(dataset, batch_size=batch_size)

#         model.cuda()
#         print('Getting embeddings...')
#         outputs = []
#         progress_bar = tqdm(dataloader, leave=False, disable=False)
#         for batch in progress_bar:
#             model.zero_grad()
#             batch = tuple(b.cuda() for b in batch)

#             # outputs.append(model(batch[0], batch[1]).pooler_output.cpu().detach().numpy())
#             outputs.append(model(batch[0], batch[1]).last_hidden_state.mean(axis = 1).cpu().detach().numpy())

#         print('Done with getting embedding! cleaning up...')
#         emb = outputs[0]
#         for j in range(1, len(outputs)):
#             emb = np.vstack([emb, outputs[j]])

#         print('saving to pickle...')
#         # with open('climatebert_embedding_part' + str(i) + '.pickle', 'wb') as f:
#         #     pickle.dump(emb, f)

#         # print('pickle saved to: ' + 'climatebert_embedding_part' + str(i) + '.pickle')
#         with open('climatebert_embedding_lhs_part' + str(i) + '.pickle', 'wb') as f:
#             pickle.dump(emb, f)

#         print('pickle saved to: ' + 'climatebert_embedding_lhs_part' + str(i) + '.pickle')