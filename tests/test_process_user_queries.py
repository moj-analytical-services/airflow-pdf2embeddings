import os
import pytest
import logging
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from pdf2embeddings.process_user_queries import query_embeddings


def test_query_embeddings_with_word2vec_with_exact_query():
    embedding, trained_df = query_embeddings(
        "Michael went to the store to buy some eggs .",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'Word2Vec',
        'Word2Vec',
        os.path.join(os.getenv('FIXTURES_DIR'), 'word2vec.pickle')
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == "Michael went to the store to buy some eggs ."
    np.testing.assert_array_equal(embedding, trained_df['Word2Vec'][0])


def test_query_embeddings_with_word2vec_when_model_not_given_raises_error():
    with pytest.raises(FileNotFoundError):
        query_embeddings(
            "Michael went to the store to buy some eggs .",
            os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
            'Word2Vec',
            'Word2Vec'
        )


def test_query_embeddings_with_word2vec_tfidf_weighted_with_exact_query():
    embedding, trained_df = query_embeddings(
        "Michael went to the store to buy some eggs .",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'Word2Vec_with_TfIdf_weights',
        'Word2Vec_TfIdf_weighted',
        os.path.join(os.getenv('FIXTURES_DIR'), 'word2vec.pickle'),
        tfidf_vectorizer=os.path.join(os.getenv('FIXTURES_DIR'), 'tfidf_vectorizer.pickle')
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == "Michael went to the store to buy some eggs ."
    np.testing.assert_array_equal(embedding, trained_df['Word2Vec_with_TfIdf_weights'][0])


def test_query_embeddings_with_word2vec_raises_logger_error_when_all_words_out_of_vocabulary(caplog):
    with caplog.at_level(logging.ERROR):
        embedding, trained_df = query_embeddings(
            "Hello there how are you?",
            os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
            'Word2Vec',
            'Word2Vec',
            os.path.join(os.getenv('FIXTURES_DIR'), 'word2vec.pickle')
        )
    expected_log_message = \
        'None of the words inputted are in the Word2Vec vocabulary. Please change your input or try a different ' \
        'model, such as ELMo or BERT. Returning empty array and DataFrame.'
    # "The following words are not in the trained vocabulary and were therefore excluded from the search: " \
    # "['Hello', 'there', 'how', 'are', 'you', '?']"
    print(caplog.text)
    print(vars(caplog))
    assert expected_log_message in caplog.text
    np.testing.assert_array_equal(embedding, np.array([]))
    pdt.assert_frame_equal(trained_df, pd.DataFrame())


def test_query_embeddings_with_word2vec_raises_logger_warning_when_some_words_out_of_vocabulary(caplog):
    with caplog.at_level(logging.WARNING):
        embedding, trained_df = query_embeddings(
            "Hello Michael, this is a trial sentence!",
            os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
            'Word2Vec',
            'Word2Vec',
            os.path.join(os.getenv('FIXTURES_DIR'), 'word2vec.pickle')
        )
    expected_log_message = \
        "The following words are not in the trained vocabulary and were therefore excluded from the search: " \
        "['Hello', 'trial', '!']"
    assert expected_log_message in caplog.text


def test_query_embeddings_with_elmo_with_exact_query():
    embedding, trained_df = query_embeddings(
        "Michael went to the store to buy some eggs .",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'ELMo_layer_3',
        'ELMo'
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == "Michael went to the store to buy some eggs ."
    np.testing.assert_array_almost_equal(embedding, trained_df['ELMo_layer_3'][0])


def test_query_embeddings_with_bert_with_exact_query():
    embedding, trained_df = query_embeddings(
        "Michael went to the store to buy some eggs .",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'BERT',
        'BERT'
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == "Michael went to the store to buy some eggs ."
    np.testing.assert_array_almost_equal(embedding, trained_df['BERT'][0])


def test_query_embeddings_raises_error_when_input_is_empty():
    with pytest.raises(KeyError):
        query_embeddings(
            "  ",
            os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
            'BERT',
            'BERT'
        )


# The remaining tests below use a query 'New York' which does not correspond exactly to any of the sentences in the
# testing corpus. However, it should still be picked up as most similar to the sentence which contains 'New York' in it.
def test_query_embeddings_with_word2vec_with_non_exact_query():
    embedding, trained_df = query_embeddings(
        "New York",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'Word2Vec',
        'Word2Vec',
        os.path.join(os.getenv('FIXTURES_DIR'), 'word2vec.pickle')
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia :"
    with pytest.raises(AssertionError):
        # checks arrays are no longer almost equal as the query 'New York' doesn't exactly match any sentence
        np.testing.assert_array_almost_equal(embedding, trained_df['Word2Vec'][0])


def test_query_embeddings_with_word2vec_tfidf_weighted_with_non_exact_query():
    embedding, trained_df = query_embeddings(
        "New York",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'Word2Vec_with_TfIdf_weights',
        'Word2Vec_TfIdf_weighted',
        os.path.join(os.getenv('FIXTURES_DIR'), 'word2vec.pickle'),
        tfidf_vectorizer=os.path.join(os.getenv('FIXTURES_DIR'), 'tfidf_vectorizer.pickle')
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia :"
    with pytest.raises(AssertionError):
        # checks arrays are no longer almost equal as the query 'New York' doesn't exactly match any sentence
        np.testing.assert_array_almost_equal(embedding, trained_df['Word2Vec_with_TfIdf_weights'][0])


def test_query_embeddings_with_elmo_with_non_exact_query():
    embedding, trained_df = query_embeddings(
        "New York",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'ELMo_layer_3',
        'ELMo'
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia :"
    with pytest.raises(AssertionError):
        # checks arrays are no longer almost equal as the query 'New York' doesn't exactly match any sentence
        np.testing.assert_array_almost_equal(embedding, trained_df['ELMo_layer_3'][0])


def test_query_embeddings_with_bert_with_non_exact_query():
    embedding, trained_df = query_embeddings(
        "New York",
        os.path.join(os.getenv('FIXTURES_DIR'), 'full_df_with_embeddings.parquet.gzip'),
        'BERT',
        'BERT'
    )
    trained_df.sort_values('metric_distance', ascending=True, inplace=True)
    trained_df.reset_index(inplace=True, drop=True)
    assert trained_df['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia :"
    with pytest.raises(AssertionError):
        # checks arrays are no longer almost equal as the query 'New York' doesn't exactly match any sentence
        np.testing.assert_array_almost_equal(embedding, trained_df['BERT'][0])
