import os

import numpy as np
from sentence_transformers import SentenceTransformer


def bert_transformer():
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    sentences_path = "/home/newdisk/zjy/datasets/PascalSentenceDataset/merged_sentence/"
    file_list = os.listdir(sentences_path)
    sentences = []
    file_list.sort()
    for f in file_list:
        f = open(sentences_path + f, "r")
        sentences += f.readlines()
        f.close()

    sentence_embeddings = sbert_model.encode(sentences)

    np.save("/home/newdisk/zjy/projects/DSCMR-master/data/pascal_sentence/sentence_features_doc2vec_bert_768",
            sentence_embeddings)


if __name__ == "__main__":
    bert_transformer()
