[![Build Badge](https://github.com/moj-analytical-services/airflow-pdf2embeddings/workflows/build/badge.svg)](https://github.com/moj-analytical-services/airflow-pdf2embeddings/actions)
[![PyPI version](https://img.shields.io/pypi/v/pdf2embeddings.svg)](https://pypi.org/project/pdf2embeddings/)
[![Licence](https://img.shields.io/github/license/moj-analytical-services/airflow-pdf2embeddings.svg)](https://github.com/moj-analytical-services/airflow-pdf2embeddings/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/moj-analytical-services/airflow-pdf2embeddings/badge.svg?branch=master)](https://coveralls.io/github/moj-analytical-services/airflow-pdf2embeddings?branch=master)



NLP tool for scraping text from a corpus of PDF files, embedding the sentences in the text and finding semantically similar sentences to a given search query
====================================================================================================================================================================================

The code in this repository performs 3 main tasks.  
- Scraping the text from a corpus of PDF files. The text is then cleaned, split into sentences, and saved into a pd.DataFrame, .csv or .parquet file containing 3 columns. One column contains the text of all the PDFs in the corpus (one sentence per row), the second column contains the title of the PDF where each sentence is taken from, and the third column contains the number of the page where each sentence is located within that PDF. This enables easy lookup.
- Embedding all the scraped sentences in the corpus of PDFs using three different NLP models: Word2Vec (with the option to include Tf-Idf weights), ELMo and BERT. For each model, sentence-level embeddings are generated.
- Corpus querying. This is in the form of a search tool, where the user can input a search query (one to a few words), and the tool will output the most similar sentences in the PDF corpus to the user query. This is done by comparing the embedding of the user query against all the embeddings of each sentence in the scraped corpus of PDFs. This effectively acts as a search engine. It is important that the model used to embed the user's search query matches the NLP model used to embed the PDF corpus. The default similarity metric is cosine similarity, although this can be changed by the user.

Project Organization
------------------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── .envrc             <- The file containing the set up for environment variables (required if using the runner 
    │                         scripts). `$PWD` should correspond to the directory where you clone this repository.
    ├── .gitignore         <- The files (including data) which are not uploaded to GitHub. Edit as required.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling. This is where you cleaned datasets will be saved.
    │   └── raw            <- The original, immutable data dump.
    │       └── pdfs       <- Where your PDF files to be scraped are stored. The other data folders are there if you need
    │                         to do other cleaning tasks, but are not used in this project as provided.
    │
    ├── models             <- Trained and serialized models. This is where your NLP models will be saved. No models have 
    │                         been uploaded to GitHub.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so pdf2embeddings can be imported
    │
    ├── MANIFEST.in        <- Tells setup.py which package data to include and exclude.
    │
    ├── config             <- This folders stores configuration files (for example suggested filenames for saving 
    │   │                     specific objects) that are read in by some of the runner scripts. Edit as required.
    │   ├── filenames.json            
    │   └── words_to_replace.json
    │
    ├── scripts            <- Executable scripts are saved here. They should be run in the order listed below.
    │   ├── data_processing_runner.py   
    │   ├── embeddings_runner.py
    │   ├── user_search_runner.py
    │   └── tmp            <- The folder where the loggers are saved (for example, debug.log, info.log, warning.log)
    │                         Logs have not been uploaded to GitHub.
    │
    ├── pdf2embeddings     <- Source code for use in this project. See description below for how to use the files.
    │   ├── __init__.py    <- Makes pdf2embeddings a Python module
    │   ├── arrange_text.py
    │   ├── embedder.py
    │   ├── json_creator.py
    │   ├── logging.yaml
    │   ├── process_user_queries.py
    │   └── scraper.py
    │
    ├── tests              <- Unit tests for all functions and methods defined in all modules within the pdf2embeddings folder, to                
    │   │                     be run using pytest. It also includes an end-to-end test. These should not be modified by
    │   │                     the user.  
    │   ├── conftest.py
    │   ├── end_to_end_test.py
    │   ├── test_arrange_text.py
    │   ├── test_embedder.py
    │   ├── test_process_user_queries.py
    │   ├── test_scraper.py
    │   └── fixtures       <- This folder contains all the pytest fixtures required to run the tests. These should not
    │       │                 be modified by the user.                
    │       ├── dummy_embeddings.npy
    │       ├── dummy_sentences.csv
    │       ├── dummy_sentences.parquet
    │       ├── dummy_sentences.txt
    │       ├── expected_bert_embeddings.npy
    │       ├── expected_elmo_embeddings.npy
    │       ├──  expected_tfidf_scores.json
    │       ├── expected_w2v_embeddings_tfidf_false.npy
    │       ├── expected_w2v_embeddings_tfidf_true.npy
    │       ├── full_df_with_embeddings.parquet.gzip
    │       ├── test_pdf_1.pdf
    │       ├── test_pdf_2.pdf
    │       ├── tfidf_vectorizer.pickle
    │       ├── word2vec.pickle
    │       ├── word2vec_tfidf.pickle
    │       └── words_to_replace.json
    │
    └── test_environment.py


--------

Clone this repository from GitHub, or install this project by running 
```
$ pip install pdf2embeddings
```
on the terminal command (or `pip3`, as appropriate).  

If you have cloned it from GitHub, you can run a test to ensure your environment is properly set-up. This project has 
not been tested on versions of Python older than 3.6, and some versions of the `numpy` library older than 1.17 are also known 
to cause issues. Please run the following line in your terminal
```
$ PYTHONHASHSEED=123 python3 -m pytest
```
There should be 32 tests. If they all pass, you're good to start using this package. If some of the tests fail, please 
check your environment. This project has only been tested with the environment as described in the `requirements.txt` 
file. Note that the environment variable 
`PYTHONHASHSEED` must be set to `"123"` while running the tests, to ensure deterministic reproducibility of the Word2Vec 
models. Two tests will fail if this is not set up correctly.

Each module has been fully documented.  
Before you start, please configure your environment variables according to your own directory path. Please refer to
the `.envrc` file, where `$PWD` corresponds to the directory path where this repository has been cloned.

In order to scrape the text from a corpus of PDF files, you will need to save your PDFs in the folder (`~/data/raw/pdfs`).  
Alternatively to scraping files from local storage, this package also supports cloud storage on AWS S3 buckets only 
(other cloud storage solutions are not natively supported).  
You can make use of the `data_processing_runner.py` script to scrape the PDFs, clean the text, split all the text into 
sentences,  and save this into a .csv file. The script imports the two modules `scraper` and `arrange_text`.
```python
import os
import yaml
import logging.config
from pdf2embeddings.scraper import DocumentScraper
from pdf2embeddings.arrange_text import CorpusGenerator


if __name__ == "__main__":
    DATA_DIR = os.getenv('DATA_DIR')
    CONFIG_DIR = os.getenv('CONFIG_DIR')
    LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')

    with open(LOGGING_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

    pdfs_folder = os.path.join(DATA_DIR, 'raw', 'pdfs')
    json_path = os.path.join(CONFIG_DIR, 'words_to_replace.json')
    scraper = DocumentScraper(pdfs_folder, json_path)
    df_by_page = scraper.document_corpus_to_pandas_df()
    generator = CorpusGenerator(df_by_page)
    df_by_sentence = generator.df_by_page_to_df_by_sentence()

    df_by_page.to_csv(os.path.join(DATA_DIR, 'processed', 'corpus_by_page.csv'))  # optional, for reference
    df_by_sentence.to_csv(os.path.join(DATA_DIR, 'processed', 'corpus_by_sentence.csv'), index=False)

```
The file `words_to_replace.json` in the `config` folder is used for ad-hoc text cleaning.   
If using an AWS S3 bucket for storing PDFs, please set `from_s3_bucket=True`, so the `scraper` will be
```python
scraper = DocumentScraper(pdfs_folder, json_path, from_s3_bucket = True)
```
If you choose this option, the `pdfs_folder` can either start with `"s3://"` or omit this prefix. Please ensure you already have the 
correct credentials for accessing the S3 bucket you are pointing to, and that the path is correct.  
When running
`scraper.document_corpus_to_pandas_df()`, the json is deserialised into a python dictionary, and the corpus text will be
cleaned by replacing each key in this dictionary with its value. In order to modify and customize the content of this
json file, run the script `pdf2embeddings/json_creator.py` and adapt it as necessary.

Once you have created a file `corpus_by_sentence.csv`, you can embed the sentences in this file using your model of choice out of 
Word2Vec (with the option to include Tf-Idf weights), ELMo and BERT. For each model, sentence-level embeddings are 
generated. Where the original model would generate word-level embeddings, sentence-level embeddings have been created by 
averaging all the word embeddings of the respective sentence. The `embeddings_runner.py` script is an example of how you 
could run all 4 NLP models and save them separately. It imports the `embedder` module.

```python
import os
import json
import yaml
import logging.config
import pandas as pd
from pdf2embeddings.embedder import Embedder


models_to_be_run = [
    'Word2Vec_tfidf_weighted',  # comment out as needed
    'Word2Vec',
    'BERT',
    'ELMo',
]


if __name__ == '__main__':
    DATA_DIR = os.getenv('DATA_DIR')
    MODELS_DIR = os.getenv('MODELS_DIR')
    CONFIG_DIR = os.getenv('CONFIG_DIR')
    LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')

    with open(LOGGING_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

    with open(os.path.join(CONFIG_DIR, 'filenames.json'), 'r') as f:
        file_names = json.load(f)

    corpus_filename = "corpus_by_sentence.csv"
    corpus_by_sentence = pd.read_csv(os.path.join(DATA_DIR, "processed", corpus_filename))
    list_of_sentences = corpus_by_sentence['sentence'].values.tolist()
    print("Instantiating Embedder class.")
    embedder = Embedder(list_of_sentences)

    for model in models_to_be_run:
        print(f"Calculating {model} embeddings.")
        if model == 'Word2Vec_tfidf_weighted':
            sentence_embeddings, model_obj, tfidf_vectorizer = embedder.compute_word2vec_embeddings(tfidf_weights=True)
            embedder.save_model(tfidf_vectorizer, MODELS_DIR, file_names[model]['vectorizer_filename'])
            # the line above is specific to Word2Vec with TfIdf vectorizer and cannot be generalized to other models
        elif model == 'Word2Vec':
            sentence_embeddings, model_obj, _ = embedder.compute_word2vec_embeddings(tfidf_weights=False)
        elif model == 'BERT':
            bert_model = 'bert-base-nli-stsb-mean-tokens'  # This line is specific to BERT
            sentence_embeddings, model_obj = embedder.compute_bert_embeddings(bert_model)
        elif model == 'ELMo':
            sentence_embeddings, model_obj = embedder.compute_elmo_embeddings()
        else:
            raise KeyError(f'The model {model} is not recognized as input.')
        print(f"{model} embeddings calculated. Saving model.")
        embedder.save_embeddings(sentence_embeddings, MODELS_DIR, file_names[model]['embeddings_filename'])
        embedder.save_model(model_obj, MODELS_DIR, file_names[model]['model_filename'])
        print(f"{model} model saved. Saving .parquet file.")
        df = embedder.add_embeddings_to_corpus_df(
            os.path.join(DATA_DIR, "processed", corpus_filename), sentence_embeddings, file_names[model]['column_name']
        )
        embedder.df_to_parquet(df, os.path.join(DATA_DIR, "processed", file_names[model]['parquet_filename']))
        print(f"Parquet file saved. All steps done for the {model} model.")

```
Each model has been saved as a `.pickle` file in the `models` folder, each model's embeddings as a `.npy` file in the 
`models` folder, and each pd.DataFrame as a `.parquet` file in the `data/processed` folder. Each `.parquet` file 
contains the same data as the `corpus_by_sentence.csv` file previously saved, with an added column, representing the sentence embeddings for the 
chosen model. A separate `.parquet` has been saved for each model, although the user may modify the script above to save
all models' embeddings in the same `.parquet` file. The file names of the `.pickle`, `.npy` and `.parquet` files are
stored in the `filenames.json ` in the `config` folder. In order to modify and customize these names, run the script 
`pdf2embeddings/json_creator.py` and adapt it as necessary.

Finally, in order to search through your corpus of PDF files given a *user search query* (which can be a single word or a 
few words), run the `user_search_runner.py` script in the `scripts` folder, which imports the `process_user_queries` module:
```python
import os
import yaml
import json
import logging.config
from pdf2embeddings.process_user_queries import query_embeddings


if __name__ == '__main__':
    user_search_input = 'cell phone'
    model_name = 'BERT'  # change as appropriate
    DATA_DIR = os.getenv("DATA_DIR")
    CONFIG_DIR = os.getenv('CONFIG_DIR')
    MODELS_DIR = os.getenv("MODELS_DIR")
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    with open(LOGGING_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
    with open(os.path.join(CONFIG_DIR, 'filenames.json'), 'r') as f:
        file_names = json.load(f)

    tfidf_vectorizer = os.path.join(MODELS_DIR, "tfidf_vectorizer.pickle")

    model = os.path.join(MODELS_DIR, file_names[model_name]["model_filename"])  # this is optional for ELMo and BERT.
    trained_df_path = os.path.join(DATA_DIR, 'processed', file_names[model_name]["parquet_filename"])
    user_input_embedding, trained_df = query_embeddings(
        user_search_input, trained_df_path, file_names[model_name]["column_name"], model_name, model,
        distance_metric='cosine', tfidf_vectorizer=tfidf_vectorizer
    )
    # tfidf_vectorizer is not used (and optional) when model is not 'Word2Vec_TfIdf_weighted'
    if user_input_embedding.size and not trained_df.empty:  # they must not be empty
        print(trained_df.sort_values('metric_distance', ascending=True)[['sentence', 'metric_distance']].
              reset_index(drop=True).head(10))
```

At this point, the `user_input_embedding` is the embedding of the user search query, and `trained_df` is the 
pd.DataFrame containing a column with the metric distance between the user embedding and each individual sentence 
embedding in the corpus (default metric: cosine similarity). If you want to visualise the most similar sentences to the
user search query, you can simply sort the pd.DataFrame by its `metric_distance` column.
```python
print(trained_df.sort_values('metric_distance', ascending=True)[['sentence', 'metric_distance']].
              reset_index(drop=True)
```

<p><small>Project description adapted from the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
