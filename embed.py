import sgt
sgt.__version__
from sgt import SGT
import pandas as pd
import numpy as np


def split(word):
    return [char for char in word] 

mining_id = 'd9729ac3-e36b-490a-81cc-55bc7d84f0b1'

sequence_data = pd.read_csv(mining_id + '.csv')

X = sequence_data["Sequences"]

sequences = [ [ind,split(x)] for ind,x in enumerate(X)]

corpus = pd.DataFrame(sequences, columns=['id', 'sequence'])

print(sequences[0])

k_values = [1,5,10,20]

for i,k in enumerate(k_values):


    embedder = SGT(
        kappa=k,
        flatten=True,
        lengthsensitive=False,
        mode='default',
        alphabets=['A','D','P','C','N','O','I','T','L','W']
        )
    embedding = embedder.fit_transform(corpus=corpus)

    print(type(embedding))
    print(embedding)

    print(k)

    #embedding.to_csv('sequence_vectors.csv')
    embedding.to_csv(mining_id + '_embeddings_k_'+str(k)+'.csv', index=False)