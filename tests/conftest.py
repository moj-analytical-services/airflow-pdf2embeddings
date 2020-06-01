import pytest
import os
import json
import numpy as np
import pandas as pd


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    monkeypatch.setenv("FIXTURES_DIR", os.path.abspath("tests/fixtures"))
    monkeypatch.setenv("CONFIG_DIR", os.path.abspath("config"))
    monkeypatch.setenv("PYTHONHASHSEED", "123")


@pytest.fixture
def list_of_sentences():
    return [
        "Michael went to the store to buy some eggs .",
        "Joel rolled down the street on his skateboard .",
        "test / this is a first sentence",
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia :",
    ]


@pytest.fixture
def text_cleaning_json():
    config_folder = os.getenv("CONFIG_DIR")
    with open(os.path.join(config_folder, 'words_to_replace.json'), 'r') as file:
        return json.load(file)


@pytest.fixture
def expected_tfidf_scores():
    fixtures_folder = os.getenv('FIXTURES_DIR')
    with open(os.path.join(fixtures_folder, 'expected_tfidf_scores.json')) as tfidf_scores:
        return json.load(tfidf_scores)


@pytest.fixture
def expected_w2v_embeddings_tfidf_true():
    fixtures_folder = os.getenv('FIXTURES_DIR')
    return np.load(os.path.join(fixtures_folder, 'expected_w2v_embeddings_tfidf_true.npy'))


@pytest.fixture
def expected_w2v_embeddings_tfidf_false():
    fixtures_folder = os.getenv('FIXTURES_DIR')
    return np.load(os.path.join(fixtures_folder, 'expected_w2v_embeddings_tfidf_false.npy'))


@pytest.fixture
def expected_elmo_embeddings():
    fixtures_folder = os.getenv('FIXTURES_DIR')
    return np.load(os.path.join(fixtures_folder, 'expected_elmo_embeddings.npy'))


@pytest.fixture
def expected_bert_embeddings():
    fixtures_folder = os.getenv('FIXTURES_DIR')
    return np.load(os.path.join(fixtures_folder, 'expected_bert_embeddings.npy'))


@pytest.fixture
def expected_df_with_embeddings():
    return pd.DataFrame({
        'dummy_sentences': ['First sentence.', 'Second sentence.', 'Third sentence.'],
        'dummy_embeddings': [np.array((1.0, 2.0, 3.0)), np.array((4.0, 5.0, 6.0)), np.array((7.0, 8.0, 9.0))]
    }).sort_index(axis=1, ascending=False)
