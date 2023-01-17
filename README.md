# Textgraph22: Temporal Graph Analysis of Misinformation Spreaders in Social Media

## 1 Overview

Proactively identifying misinformation spreaders is an important step towards mitigating the impact of fake news on our society. Although the news domain is subject to rapid changes over time, the temporal dynamics of the spreaders’ language and network have not been explored yet. In this paper, we analyze the users’ time-evolving semantic similarities and social interactions and show that such patterns can, on their own, indicate misinformation spreading. Building on these observations, we propose a dynamic graph-based framework that leverages the dynamic nature of the users’ network for detecting fake news spreaders. We validate our design choice through qualitative analysis and demonstrate the contributions of our model’s components through a series of exploratory and ablative experiments on two datasets. You can find the dataset [here](https://drive.google.com/drive/folders/1MB6zsrhNerZQlLFBdjJ8sDbvXa2NcELZ).

## 2 Setup

### 2.1 Environment Setup

* With conda
  
    ```conda env export > environment.yml```
* With pip

    ```pip install -r requirements.txt```

## 3 Usage

### 3.1 Reddit Posts Crawling

   Crawl reddit posts using the ids provided in the dataset and fill the empty strings inside the dataframe.  

### 3.2 User Embeddings

  First extract user vocabularies 

  ```
  python create_vocabs_per_month.py --base_dataset=../data/reddit_dataset/factoid_dataset.gzip
  ```

  Then run the codes to generate
          
   * UBERT embeddings

  ```
   python user_embeddings_per_month.py --vocabs_dir='../data/user_vocabs_per_month' --base_dataset='../data/reddit_dataset/factoid_dataset.gzip'
  ```

  * [User2Vec](https://github.com/samiroid/usr2vec)
  * [HGCN codebase](https://github.com/HazyResearch/hgcn)


### 3.3 Generate Graphs and Samples

  To generate graph samples, example script. Change the parameters based on the embeddings you want to use. The argument `embed_type`  takes the following values `['bert', 'usr2vec', 'usr2vec_rand', 'usr2vec_liwc', 'liwc']`. For example:

  ```
  timeframe_scripts/timeframed_graph_loader_bert.sh
  ```

  Then after creating the graph samples, run the following to make the split

  ```
  model_dataloader_scripts/model_dataloader.sh
  ```

### 3.4 Training Model

  After training, validation, test samples are created, run the model using the following

  ```
  graph_scripts/graph_model_bert_1.sh
  ```

