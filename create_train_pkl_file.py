import pickle
from collections import defaultdict
col_dic=defaultdict(list)

# replace msmarco collection path
collection_file=open('collection/collection.tsv','r').readlines()

for line in collection_file:
    docid,doctext= line.rstrip().split('\t')
    col_dic[docid]=doctext
q_map_dic_train={}
q_map_dic_test={}

# path to evaluation metric per query
# qid<\t>metric_value
query_map_file=open('train_query_map_20.tsv','r').readlines()
for line in query_map_file:
    qid,qtext,qmap=line.split('\t')
    q_map_dic_train[qid]={}
    q_map_dic_train[qid] ["text"]=qtext
    q_map_dic_train[qid] ["map"]=float(qmap)

# run file including first retrieved documents per query
run_file=open('bm25_first_docs_train.tsv','r').readlines()
for line in run_file:
    qid,docid,rank=line.split('\t')
    if qid in q_map_dic_train.keys():
        q_map_dic_train[qid]["first_doc"]=col_dic[docid]

with open('pklfiles/map_20_train.pkl', 'wb') as f:
    pickle.dump(q_map_dic_train, f, pickle.HIGHEST_PROTOCOL)

