from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math,logging


with open('pklfiles/map_20_train.pkl', 'rb') as f:
    q_map_dic_train=pickle.load(f)

mrr={}
mrr_file=open('../mrr_train_set','r').readlines()
for line in mrr_file:
    qid,mrr_=line.rstrip().split('\t')
    mrr[qid]=float(mrr_)


train_set=[]

for key in q_map_dic_train:
    if "first_doc" in q_map_dic_train[key].keys():
        qtext=q_map_dic_train[key]["text"]
        firstdoctext=q_map_dic_train[key]["first_doc"]
        actual_map = q_map_dic_train[key] ["map"]
        train_set.append( InputExample(texts=[qtext,firstdoctext],label=actual_map ))

    if len(train_set)>1000:
        break
print(len(train_set))

batch_size=8
epoch_num=1
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * epoch_num * 0.1) #10% of train data for warm-up
model_name='bert-base-uncased'

model = CrossEncoder(model_name, num_labels=1)
model_name="models/tuned_model_"+model_name+"_e"+str(epoch_num)+'_b'+str(batch_size)
# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=epoch_num,
          warmup_steps=warmup_steps,
          output_path=model_name)
model.save(model_name)
