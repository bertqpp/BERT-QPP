# BERT-QPP

In this paper, we adopt contextual embeddings to perform performance prediction specifically for the task of query performance prediction.The fine-tuned contextual representations can estimate the performance of a query based on the association between the representation of the query and the retrieved documents. We compare the performance of our approach with the state-of-the-art based on the MS MARCO passage retrieval corpus and its three associated query sets: (1) MS MARCO development set, (2) TREC DL 2019, and (3) TREC DL 2020. We show that our approach not only shows significant improved prediction performance compared to all the state-of-the-art methods, but also, unlike past neural predictors, it shows significantly lower latency, making it possible to use in practice.

We adopt two architechtures namely cross-encoder network and bi-encoder network to address QPP task. 

To replicate our results  with BERT-QPP<sub>cross</sub> and BERT-QPP<sub>bi</sub> on MSMARCO passage collection, clone this repository and download MSMARCO collection and put it in ```collection``` repository. In addition, required packages are listed in ```requirement.txt``` on python 3.7+. 

```bm25_first_docs_train.tsv``` and ```bm25_first_docs_dev.tsv``` includes the run file for first retrieved documents for queries in MSMARCO train and dev set. You can put the runfile of your desired retrieval approach in the folloinwg format for each query per line: 
```
QID\tDOCID\t1
```
## BERT-QPP<sub>cross</sub>

To train BERT-QPP<sub>cross</sub> we require query, first retrieved document, and the queries' performance. To do so,  in ```create_train_pkl_file.py``` we create a dictionary including the following attributes:
```
    train_dic[qid] ["text"]=query_text
    train_dic[qid] ["map"]=query_performance_value
    train_dic[qid]["first_retrieved_document"]=document_text
 ```
 1. run ```create_train_pkl_file.py``` to save a dictionary including query and document text as well as their associated performance. As a result ```train_map.pkl``` will be saved in ```pklfiles``` directory.
 2. run ```create_test_pkl_file.py``` to save a dictionary including query and document text on the MSMARCO developement set. As a result ```test_dev_map.pkl``` will be saved in ```pklfiles``` directory.
 3. run ```train_CE.py``` to learn the map@20 of BM25 retrieval on MSMARCO train set. alternatively, you can train with your desired metric by creating the assosiated train pkl file. me On a single 24GB RTX3090 GPU, it took less than 2 hours. You may also change the ```epoch_num```,```batch_size```, and initial  pre-trained model in this file. We used ```bert-base-uncased``` in this experiment. The trained model will be saved in ```models``` directory.
 4. If you are not willing to train the model, you can download our BERT-QPP<sub>cross</sub> trained model on bert-based-uncased from here.
 5. add the ```trained_model``` you are willing to test in ```test_CE.py``` and  run ```test_CE.py```.
 6. the results will be saved in ```results``` directory in the following format:
    ```QID\tPredicted_QPP_value```
 
## BERT-QPP<sub>bi</sub>

