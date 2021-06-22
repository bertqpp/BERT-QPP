# BERT-QPP

In this paper, we adopt contextual embeddings to perform performance prediction specifically for the task of query performance prediction.The fine-tuned contextual representations can estimate the performance of a query based on the association between the representation of the query and the retrieved documents. We compare the performance of our approach with the state-of-the-art based on the MS MARCO passage retrieval corpus and its three associated query sets: (1) MS MARCO development set, (2) TREC DL 2019, and (3) TREC DL 2020. We show that our approach not only shows significant improved prediction performance compared to all the state-of-the-art methods, but also, unlike past neural predictors, it shows significantly lower latency, making it possible to use in practice.

We adopt two architechtures namely cross-encoder network and bi-encoder network to address QPP task. 

To replicate our results  with BERT-QPP<sub>cross</sub> and BERT-QPP<sub>bi</sub> on MSMARCO passage collection, clone this repository and download MSMARCO collection and put it in ```collection``` repository. In addition, required packages are listed in ```requirement.txt``` on python 3.7+. 

## BERT-QPP<sub>cross</sub>

To train BERT-QPP<sub>cross</sub> we require query, first retrieved document, and the queries' performance. To do so,  in ```create_train_pkl_file.py``` we create a dictionary including the following attributes:
```
    train_dic[qid] ["text"]=query_text
    train_dic[qid] ["map"]=query_performance_value
    train_dic[qid]["first_retrieved_document"]=document_text
 ```
 1. run ```create_train_pkl_file.py``` to save a dictionary including query and document text as well as their associated performance.
 2. run ```train_CE.py```. On a single 24GB RTX3090 GPU, it took less than 2 hours. You may also change the ```epoch_num```,```batch_size```, and initial  pre-trained model in this file. We used ```bert-base-uncased``` in this experiment. The trained model will be saved in ```models``` directory.
 3. You may also download our BERT-QPP<sub>cross</sub> trained model on bert-based-uncased from here.
 
## BERT-QPP<sub>bi</sub>

