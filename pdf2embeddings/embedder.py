import numpy as np
import os
import pandas as pd
import logging
import nltk
import pickle
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from allennlp.commands.elmo import ElmoEmbedder
from sentence_transformers import SentenceTransformer
import multiprocessing
from typing import List, Tuple, Dict, Optional, Union


logger = logging.getLogger(__name__)


class Embedder:
    """
    This class provides methods for calculating sentence-level embeddings given a list of sentences (not tokenised)
    using Word2Vec (with an option to weight the embeddings by their tf-idf score), ELMo and BERT. It also provides
    static methods to save the embeddings and the models, and to add the embeddings as a column of a pd.DataFrame
    (useful if the DataFrame already contains another column in which each row correspond to one sentence in the list
    of sentences that have been embedded, in the same order, so one can quickly match each sentence with its embedding).
    Note that Word2Vec is the only model that is trained only on the list of sentences provided. For ELMo and BERT,
    we are making use of pre-trained models.
    """
    def __init__(self, list_of_sentences: List[str]) -> None:
        """
        :param list_of_sentences: a list where each item is a string containing a full sentence.
        """
        self.list_of_sentences = list_of_sentences
        logger.info(f'Initializing Embedder class with {len(self.list_of_sentences)} sentences.')
        assert_message = 'list_of_sentences must be a list of strings, where each string is a full sentence.'
        assert isinstance(self.list_of_sentences, list), assert_message
        assert all(isinstance(item, str) for item in self.list_of_sentences), assert_message

    def compute_word2vec_embeddings(
            self,
            tfidf_weights: bool = True,
            embedding_dim: int = 300,
            window: int = 5,
            min_count: int = 0,
            negative: int = 15,
            iterations: int = 10,
            workers: int = multiprocessing.cpu_count(),
            **kwargs
    ) -> Tuple[np.ndarray, Word2Vec, Optional[TfidfVectorizer]]:
        """
        Calculates the sentence-level embeddings of the sentences in list_of_sentences, using Word2Vec. If tfidf_weights
        is True, the word-level embeddings are multiplied with their tfidf score before calculating sentence-level
        embeddings.
        :param tfidf_weights: boolean. If True, the word embeddings are weighted with their tf-idf score.
        :param embedding_dim:
        :param window:
        :param min_count:
        :param negative:
        :param iterations:
        :param workers:
        :return: sentence_embeddings: a np.array containing sentence-level embeddings for each sentence in
                 list_of_sentences.
                 w2v: the Word2Vec model trained using list_of_sentences
                 tfidf_vectorizer: the TfidfVectorizer model. If tfidf_weights is False, this deafults to None.
        """

        tok_sentences = [nltk.word_tokenize(sentence) for sentence in self.list_of_sentences]
        # list of lists of strings (words)
        logger.info('Initializing Word2Vec.')
        w2v = Word2Vec(
            tok_sentences, size=embedding_dim, window=window, min_count=min_count, negative=negative, iter=iterations,
            workers=workers, **kwargs
        )
        word_to_tfidf_scores = dict()
        list_of_vocabs = [w2v.wv.vocab]
        tfidf_vectorizer = None

        if tfidf_weights:
            word_to_tfidf_scores, tfidf_vectorizer = self._compute_tfidf_weights(w2v)
            list_of_vocabs.append(word_to_tfidf_scores)

        sentence_embeddings = []

        for sentence in tok_sentences:
            words_embeddings = [
                w2v.wv.get_vector(word) * word_to_tfidf_scores.get(word, 1)
                for word in sentence if all([word in vocab for vocab in list_of_vocabs])
            ]
            # if statement is to exclude words that appear less than 'min_count' times in the w2v.wv.vocab (see Word2Vec
            # model) and, if tfidf_weights is True, also to exclude words that do not appear in word_to_tfidf_scores (to
            # note that this tfidf vocabulary is smaller than w2v.wv.vocab so some additional words are lost).

            mean_sentence_vector = np.mean(words_embeddings, axis=0)
            sentence_embeddings.append(mean_sentence_vector)
        sentence_embeddings = np.array(sentence_embeddings)  # shape (len(tok_sentences), embedding_dim)

        return sentence_embeddings, w2v, tfidf_vectorizer

    def _compute_tfidf_weights(self, w2v: Word2Vec) -> Tuple[Dict[str, float], TfidfVectorizer]:
        """
        Compute the tfidf weights given a list of sentences, if the words are in the given Word2Vec instance's
        vocabulary.
        :param w2v: a Word2Vec object (instance). Words that are not in the Word2Vec vocabulary will be ignores
        :return: word_to_tfidf_score: a dictionary with keys being the words in tf_idf_vectorizer (str) and the values
                                      being their corresponding tfidf scores (float).
                 tfidf_vectorizer: the TfidfVectorizer object, available for future re-use
        """
        logger.info('Initializing TfidfVectorizer.')
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, vocabulary=list(w2v.wv.vocab))
        # forcing the w2v vocabulary, as otherwise the tf-idf generated vocabulary will not match completely.
        tfidf_vectorizer.fit(self.list_of_sentences)
        word_to_tfidf_score = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))
        # Alternative 1:
        # word_to_tfidf_score = {
        #     item[0]: tfidf_vectorizer.idf_[item[1]] for item in tfidf_vectorizer.vocabulary_.items()
        # }
        # Alternative 2:
        # word_to_tfidf_score = defaultdict(lambda: max(tfidf_vectorizer.idf_),
        #    [(w, tfidf_vectorizer.idf_[i]) for w, i in tfidf_vectorizer.vocabulary_.items()])

        return word_to_tfidf_score, tfidf_vectorizer

    def compute_elmo_embeddings(
            self,
            options_file: str = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/"
                                "elmo_2x4096_512_2048cnn_2xhighway_options.json",
            weight_file: str = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/"
                               "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    ) -> Tuple[np.ndarray, ElmoEmbedder]:
        """
        Calculates the ELMo embeddings of the sentences within the corpus. Each embedding has 3 layers and 1024
        dimensions.

        :param options_file: from ELMo, the default options for the model
        :param weight_file: from ELMo, the pre-trained weights for the model
        :return: sentence_embeddings: a np.array containing sentence-level embeddings for each sentence in
                 list_of_sentences
                 elmo: the ElmoEmbedder model used to embed list_of_sentences
        """

        tok_sentences = [nltk.word_tokenize(sentence) for sentence in self.list_of_sentences]
        logger.info('Initializing ElmoEmbedder.')
        logger.debug(f'Parameters: \n options_file: {options_file} \n weight_file: {weight_file}')
        elmo = ElmoEmbedder(options_file, weight_file)

        word_embeddings = elmo.embed_sentences(tok_sentences, batch_size=16)  # this returns a generator such that
        # len(list(embedding_iter)) = len(tok_sentences)

        sentence_embeddings = []
        for embedding in word_embeddings:
            sentence_embedding = np.mean(embedding, axis=1)[2]  # taking last layer (ELMo layer is number 3, hence [2]).
            sentence_embeddings.append(sentence_embedding)
        sentence_embeddings = np.array(sentence_embeddings)  # shape (len(tok_sentences), 1024)

        return sentence_embeddings, elmo

    def compute_bert_embeddings(self, model: str) -> Tuple[np.ndarray, SentenceTransformer]:
        """
        Calculates the BERT embeddings of the sentences within the corpus. Each embedding has 768 dimensions.
        :param model: a string containing the BERT model to use. For a list of available models, see
               https://github.com/UKPLab/sentence-transformers#pretrained-models
        :return: sentence_embeddings: a np.array containing sentence-level embeddings for each sentence in
                 list_of_sentences.
                 bert: the SentenceTransformer BERT model used to embed list_of_sentences
        """
        logger.info('Initializing SentenceTransformer.')
        logger.debug(f'Model used is {model}.')
        bert = SentenceTransformer(model)
        sentence_embeddings = bert.encode(self.list_of_sentences)
        sentence_embeddings = np.array(sentence_embeddings)  # shape (len(self.list_of_sentences), 768)

        return sentence_embeddings, bert

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, folder: str, filename: str) -> None:
        assert '.npy' in filename, "Please ensure you are using a .npy format for saving the array with the embeddings."
        np.save(os.path.join(folder, filename), embeddings)

    @staticmethod
    def save_model(
            model: Union[Word2Vec, TfidfVectorizer, ElmoEmbedder, SentenceTransformer],
            folder: str,
            filename: str
    ) -> None:
        assert '.pickle' in filename, "Please ensure you are using a .pickle format for saving your model."
        pickle.dump(model, open(os.path.join(folder, filename), "wb"))

    @staticmethod
    def add_embeddings_to_corpus_df(
            input_path: Union[str, pd.DataFrame],
            sentence_embeddings: Union[str, np.ndarray],
            embeddings_column_name: str
    ) -> pd.DataFrame:
        """
        Add the given sentence-level embeddings (as an extra column) to a pd.DataFrame containing at least a column
        where each rows represents one sentence. The order (and length) of the sentences and the embeddings must be the
        same.
        :param input_path: the path to the source pd.DataFrame. Can be a pd.DataFrame directly, or a str pointing to
        a .parquet or .cvs file. No other formats are accepted.
        :param sentence_embeddings: path to the embeddings to be added. Can be a np.array or a str pointing to
        a .npy file. No other formats are accepted.
        :param embeddings_column_name: the column name of the added column containing the embeddings.
        :return: df: a copy of the pd.DataFrame specified in input_path with the added column containing the embeddings.
        """
        if isinstance(input_path, str) and ".parquet" in input_path:
            logger.info(f'Loading DataFrame from {input_path} file.')
            df = pd.read_parquet(input_path)
        elif isinstance(input_path, str) and ".csv" in input_path:
            logger.info(f'Loading DataFrame from {input_path} file.')
            df = pd.read_csv(input_path)
        elif isinstance(input_path, pd.DataFrame):
            df = input_path
        else:
            raise TypeError(
                "Please ensure your 'input path' is either a string pointing to either a .parquet or a .csv file, or a "
                "pandas.DataFrame. Other input formats are not recognised by this method."
            )

        if isinstance(sentence_embeddings, str) and ".npy" in sentence_embeddings:
            logger.info(f'Loading np.array from {sentence_embeddings} file.')
            sentence_embeddings = np.load(sentence_embeddings)
        elif isinstance(sentence_embeddings, np.ndarray):
            pass  # sentence_embeddings = sentence_embeddings
        else:
            raise TypeError(
                "Please ensure your 'sentence_embeddings' is either a .npy file or a numpy.array. Other input formats "
                "are not recognised by this method."
            )
        assert df.shape[0] == len(sentence_embeddings), \
            f"pd.DataFrame and embeddings could not be matched with lengths {df.shape[0]} and " \
            f"{len(sentence_embeddings)} respectively."

        df[embeddings_column_name] = list(sentence_embeddings)  # df expects a list input, not np.array
        return df

    @staticmethod
    def df_to_parquet(df: pd.DataFrame, output_filename: str) -> None:
        assert '.parquet' in output_filename, "Please ensure you are using a .parquet format for saving your model."
        df.to_parquet(output_filename + '.gzip', compression='gzip')
