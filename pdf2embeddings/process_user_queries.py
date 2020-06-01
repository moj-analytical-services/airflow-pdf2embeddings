import nltk
import pandas as pd
import numpy as np
import pickle
import logging
from scipy.spatial.distance import cdist
from gensim.models import Word2Vec
from allennlp.commands.elmo import ElmoEmbedder
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional, Union


logger = logging.getLogger(__name__)


def query_embeddings(
        user_input: str,
        trained_df: Union[str, pd.DataFrame],
        expected_embeddings_colname: str,
        model_name: str,
        model: Optional[Union[str, Word2Vec, ElmoEmbedder, SentenceTransformer]] = None,
        expected_sentence_colname: str = 'sentence',
        distance_metric: str = 'cosine',
        metric_colname: str = 'metric_distance',
        tfidf_vectorizer: Optional[Union[str, TfidfVectorizer]] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Takes a user input search, embeds it using the model of choice (available: Word2Vec with tf-idf weighting, Word2Vec
    without tf-idf weighting, ELMo and BERT), and compares this embedding against all the embeddings of a given corpus
    of sentences, calculating the similarity score. Embeddings must all be sentence-level.
    NOTE: this uses all models in testing mode. No training is performed and models are not updated to include user
    queries.
    :param user_input: the user search query
    :param trained_df: a pd.DataFrame, or a string pointing to a .parquet file which can be converted into a
           pd.DataFrame, containing at least two columns, one with the sentences in the corpus to be searched, and one
           with their respective embeddings. The model used to embed such sentences must be the same as the one given
           in model_name.
    :param expected_embeddings_colname: the column name, in the pd.DataFrame mentioned above, containing the embeddings.
    :param model_name: the model to use to embed the user query; available options: 'Word2Vec',
           'Word2Vec_TfIdf_weighted', 'ELMo', 'BERT'.
    :param model: the trained model used to embed the user query. It can be an instance of Word2Vec, ElmoEmbedder or
           SentenceTransformer, or a string pointing to a .pickle file representing the trained model of choice, to be
           loaded. This must be the same model chosen in model_name. Defaults: None. If None, the standard pre-trained
           ELMo or BERT models will be used. A model entry must be given if model_name is 'Word2Vec' or
           'Word2Vec_TfIdf_weighted', or else a FileNotFoundError will be raised.
    :param expected_sentence_colname: the column name, in the pd.DataFrame loaded from the .parquet file in
           trained_df_path, containing the sentences.
    :param distance_metric: the metric used to compare the user query embedding and the corpus embeddings and calculate
           the semantic similarity. Default: cosine similarity.
    :param metric_colname: the name of the column to be added to the trained_df containing the distance between the
           user input embedding and each sentence embedding, calculated using the distance_metric.
    :param tfidf_vectorizer: a TfidfVectorizer object, or a string pointing to a .pickle file representing a
           TfidfVectorizer object. Default: None. This only needs to be provided when model_name is
           'Word2Vec_TfIdf_weighted'.
    :return: user_input_embedding: the embedding of the user_input
             trained_df: a pd.DataFrame, which is the same as the one extracted from the .parquet file in
             trained_df_path, with an added column representing the distance between the user input embedding and each
             sentence embedding, calculated using the distance_metric.
    """

    if not user_input.strip():
        raise KeyError('User input cannot be empty.')
    assert model_name in ['Word2Vec', 'Word2Vec_TfIdf_weighted', 'ELMo', 'BERT'], \
        "Model name not recognized. Please choose one of ['Word2Vec', 'Word2Vec_TfIdf_weighted', 'ELMo', 'BERT']."
    assert isinstance(trained_df, pd.DataFrame) or '.parquet' in trained_df, \
        "trained_df_path should be a pd.DataFrame, or a string pointing to a .parquet file. Format not recognised."
    if '.parquet' in trained_df:
        logger.info(f'Loading DataFrame from {trained_df} file.')
        trained_df = pd.read_parquet(trained_df)
    assert expected_sentence_colname in trained_df.columns, \
        f"Please check your input trained DataFrame. It should contain a '{expected_sentence_colname}' column."
    assert expected_embeddings_colname in trained_df.columns, \
        f"Please check your input trained DataFrame. It should contain a '{expected_embeddings_colname}' column."
    assert model is None or isinstance(model, (Word2Vec, ElmoEmbedder, SentenceTransformer)) or '.pickle' in model, \
        "model should be None, or a string pointing to a .pickle file, or an instance of Word2Vec, ElmoEmbedder or" \
        "SentenceTransformer. Other file formats are not recognised."
    assert tfidf_vectorizer is None or isinstance(tfidf_vectorizer, TfidfVectorizer) or '.pickle' in tfidf_vectorizer,\
        "tfidf_vectorizer_path should be None, or a string pointing to a .pickle file, or an instance of" \
        "TfidfVectorizer. Other file formats are not recognised."

    if isinstance(model, str) and ".pickle" in model:
        logger.info(f'Loading model from {model} file.')
        model = pickle.load(open(model, "rb"))
        assert isinstance(model, (Word2Vec, ElmoEmbedder, SentenceTransformer)), \
            "The .pickle provided does not correspond to a Word2Vec, ElmoEmbedder or SentenceTransformer model."

    logger.info(f'Embedding user query using the {model_name} model...')
    if model_name in ['Word2Vec', 'Word2Vec_TfIdf_weighted']:
        if model is None:
            raise FileNotFoundError(
                "When using 'Word2Vec' or 'Word2Vec_TfIdf_weighted' model_name, the model variable must be provided "
                "and it cannot be left empty."
            )
        vocabs = [model.wv.vocab]
        word_to_tfidf_score = dict()
        tok_input = nltk.word_tokenize(user_input)

        if model_name == 'Word2Vec_TfIdf_weighted':
            assert tfidf_vectorizer is not None, \
                "When using the 'Word2Vec_TfIdf_weighted' model you must provide tfidf_vectorizer_path as a str " \
                "pointing to a .pickle."
            if isinstance(tfidf_vectorizer, str) and '.pickle' in tfidf_vectorizer:
                logger.info(f'Loading vectorizer from {tfidf_vectorizer} file.')
                tfidf_vectorizer = pickle.load(open(tfidf_vectorizer, "rb"))
            word_to_tfidf_score = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))
            vocabs.append(word_to_tfidf_score)

        words_embeddings = [
            model.wv.get_vector(word) * word_to_tfidf_score.get(word, 1)
            for word in tok_input
            if all(word in vocab for vocab in vocabs)
        ]
        if len(words_embeddings) == 0:
            logger.error(
                'None of the words inputted are in the Word2Vec vocabulary. Please change your input or try a '
                'different model, such as ELMo or BERT. Returning empty array and DataFrame.'
            )
            return np.array([]), pd.DataFrame()

        ignored_words = [word for word in tok_input if not all(word in vocab for vocab in vocabs)]
        # vocabulary is case sensitive. Need to keep case if word exists in both cases; if not,
        # need to lowercase/uppercase as appropriate to avoid a useful word being excluded. Not done yet.
        if len(ignored_words) != 0:
            logger.warning(
                f'The following words are not in the trained vocabulary and were therefore excluded from the '
                f'search: {ignored_words}'
            )

        user_input_embedding = np.mean(words_embeddings, axis=0)

    if model_name == 'ELMo':
        tok_input = nltk.word_tokenize(user_input)
        if model is None:
            logger.warning(
                'No ELMo model was provided. Defaulting to using an instance of allennlp.commands.elmo.ElmoEmbedder '
                'to embed the user query.'
            )
            model = ElmoEmbedder()
        words_embeddings = model.embed_sentence(tok_input)  # shape (3, len(tok_input), 1024)
        user_input_embedding = np.average(words_embeddings, axis=1)[2]

    if model_name == 'BERT':
        if model is None:
            logger.warning(
                'No BERT model was provided. Defaulting to using an instance of sentence_transformers.'
                'SentenceTransformer to embed the user query, with model \'bert-base-nli-stsb-mean-tokens\'.'
            )
            model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        user_input_embedding = model.encode([user_input])[0]
    logger.info('User query has been embedded.')

    trained_df[metric_colname] = cdist(
        list(trained_df[expected_embeddings_colname]), [user_input_embedding], metric=distance_metric
    )
    return user_input_embedding, trained_df