# BERT-QPP

In this paper, we adopt contextual embeddings to perform performance prediction specifically for the task of query performance prediction.The fine-tuned contextual representations can estimate the performance of a query based on the association between the representation of the query and the retrieved documents. We compare the performance of our approach with the state-of-the-art based on the MS MARCO passage retrieval corpus and its three associated query sets: (1) MS MARCO development set, (2) TREC DL 2019, and (3) TREC DL 2020. We show that our approach not only shows significant improved prediction performance compared to all the state-of-the-art methods, but also, unlike past neural predictors, it shows significantly lower latency, making it possible to use in practice.

We adopt two architechtures namely cross-encoder network and bi-encoder network to address QPP task. 

## BERT-QPP_{Cross}

