"""
This is a simple application for sentence embeddings: clustering
Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import process_text, torch
import py_vncorenlp, os, json, pandas as pd
import constants.constants as default
# model_sbert = SentenceTransformer(default.model_sbert_path)
#
#
#
# # Choose device
# try:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.environ["CUDA_VISIBLE_DEVICES"]='2, 3'
#     torch.cuda.empty_cache()
# except:
#     device = 'cpu'
#
#
# print('Your device:', device)
#
# print('Load sbert model successfully')
# model_vncore = py_vncorenlp.VnCoreNLP( save_dir=default.model_vncorenlp_path,
#                                        annotators=["wseg", "pos", "ner", "parse"],
#                                        max_heap_size=default.max_heap_size)
#
# os.chdir(default.root_path)
# print('Load vncorenlp model successfully')
# ### process text support
# #LOAD TEENCODE
# file = open(default.teencodes_path, 'r', encoding="utf8")
# teen_lst = file.read().split('\n')
# teen_dict = {}
# for line in teen_lst:
#     key, value = line.split('\t')
#     teen_dict[key] = str(value)
# file.close()
#
# #LOAD STOPWORDS
# # file = open(default.stopwords_path, 'r', encoding="utf8")
# # stopwords_lst = file.read().split('\n')
# # file.close()
#
# file = open(default.map_languages, "r")
# map_langs = json.loads(file.read())
# file.close()
# df = pd.read_csv('data.csv')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'A man is eating pasta.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.'
          ]
corpus_embeddings = embedder.encode(corpus)

# Perform kmean clustering
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")