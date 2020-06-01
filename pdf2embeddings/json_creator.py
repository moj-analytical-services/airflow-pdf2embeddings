import json
import os


def create_text_cleaning_json(folder: str) -> None:
    to_replace = {
        "\n": "",
        "Cat.": "Category",
        "cat.": "category",
        "Mr.": "Mr",
        "Mrs.": "Mrs",
        "Dr.": "Dr",
        "e.g.": "eg",
        "Op.": "Operational",
        "Cap.": "Capacity",
        "�": "",
        "\x0c": ""
    }
    with open(os.path.join(folder, 'words_to_replace.json'), 'w') as f:
        json.dump(to_replace, f)


def create_filenames_json(folder: str) -> None:
    filenames = {
        'Word2Vec': {
            "embeddings_filename": "w2v_embeddings.npy",
            "model_filename": "word2vec.pickle",
            "column_name": "Word2Vec",
            "parquet_filename": "corpus_by_sentence_with_Word2Vec.parquet"
        },
        'Word2Vec_tfidf_weighted': {
            "embeddings_filename": "w2v_with_tfidf_embeddings.npy",
            "model_filename": "word2vec_tfidf.pickle",
            "vectorizer_filename": "tfidf_vectorizer.pickle",
            "column_name": "Word2Vec_with_TfIdf_weights",
            "parquet_filename": "corpus_by_sentence_with_Word2Vec_TfIdf_weighted.parquet"
        },
        'ELMo': {
            "embeddings_filename": "elmo_embeddings.npy",
            "model_filename": "elmo_model.pickle",
            "column_name": "ELMo_layer_3",
            "parquet_filename": "corpus_by_sentence_with_ELMo.parquet"
        },
        'BERT': {
            "embeddings_filename": "bert_embeddings.npy",
            "model_filename": "bert_model.pickle",
            "column_name": "BERT",
            "parquet_filename": "corpus_by_sentence_with_BERT.parquet"
        }
    }
    with open(os.path.join(folder, 'filenames.json'), 'w') as f:
        json.dump(filenames, f)


if __name__ == '__main__':
    CONFIG_DIR = os.getenv('CONFIG_DIR')
    create_text_cleaning_json(CONFIG_DIR)
    create_filenames_json(CONFIG_DIR)
