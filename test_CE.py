import pickle 
from scipy.stats import kendalltau,pearsonr

from sentence_transformers.cross_encoder import CrossEncoder
trained_model="tuned_model_bert-base-uncased_e1_b16"

with open('pklfiles/test_dev_map.pkl', 'rb') as f:
    q_map_first_doc_test=pickle.load(f)

sentences = []
map_value_test=[]
queries=[]
for key in q_map_first_doc_test:
    if "first_doc" in q_map_first_doc_test[key].keys():
        sentences.append([q_map_first_doc_test[key]["qtext"],q_map_first_doc_test[key]["doc_text"]])
        queries.append(key)

model = CrossEncoder("models/"+trained_model, num_labels=1)
scores=model.predict(sentences)
actual=[]
predicted=[]
out=open('results/qpp_'+trained_model,'w')
for i in range(len(sentences)):
    predicted.append(float(scores[i]))
    out.write(queries[i]+'\t'+str(predicted[i])+'\n')
out.close()
