import os
import numpy as np
import pandas as pd
import warnings
import pandas.util.testing as pdt
from pdf2embeddings.embedder import Embedder
import pytest


class TestEmbedder:
    """For a fully reproducible run of the Word2Vec models, set PYTHONHASHSEED environment variable to 123."""
    def test_class_instantiation(self, list_of_sentences):
        embedder = Embedder(list_of_sentences)
        assert embedder.list_of_sentences == list_of_sentences

    def test_compute_word2vec_embeddings_when_tfidf_weights_is_false(
            self, list_of_sentences, expected_w2v_embeddings_tfidf_false
    ):
        assert os.getenv("PYTHONHASHSEED") == "123", \
            'Please set PYTHONHASHSEED environment variable to 123, or else the test will not be deterministically ' \
            'reproducible.'

        embedder = Embedder(list_of_sentences)
        embeddings, _, _ = embedder.compute_word2vec_embeddings(tfidf_weights=False, workers=1, seed=42, hashfxn=hash)
        assert len(list_of_sentences) == len(embeddings)
        assert embeddings.shape == (len(list_of_sentences), 300)
        np.testing.assert_array_equal(expected_w2v_embeddings_tfidf_false, embeddings)

    def test_compute_word2vec_embeddings_when_tfidf_weights_is_true(
            self, list_of_sentences, expected_tfidf_scores, expected_w2v_embeddings_tfidf_true
    ):
        assert os.getenv("PYTHONHASHSEED") == "123", \
            'Please set PYTHONHASHSEED environment variable to 123, or else the test will not be deterministically ' \
            'reproducible.'

        embedder = Embedder(list_of_sentences)
        embeddings, _, tfidf_vect = embedder.compute_word2vec_embeddings(
            tfidf_weights=True, workers=1, seed=42, hashfxn=hash
        )
        assert len(list_of_sentences) == len(embeddings)
        assert embeddings.shape == (len(list_of_sentences), 300)
        np.testing.assert_array_equal(expected_w2v_embeddings_tfidf_true, embeddings)
        assert dict(zip(tfidf_vect.get_feature_names(), tfidf_vect.idf_)) == expected_tfidf_scores

    def test_compute_word2vec_embeddings_is_empty_sentence_raises_error(self):
        embedder = Embedder([])
        with pytest.raises(RuntimeError):
            embedder.compute_word2vec_embeddings()

    def test_compute_word2vec_embeddings_contains_empty_sentence(self):
        embedder = Embedder(['Sentence one.', ''])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # avoids printing expected warning due to averaging an empty vector.
            embeddings, _, _ = embedder.compute_word2vec_embeddings()
        assert embeddings.shape == (2,)
        assert np.isnan(embeddings[1])

    def test_compute_elmo_embeddings(self, list_of_sentences, expected_elmo_embeddings):
        embedder = Embedder(list_of_sentences)
        embeddings, _ = embedder.compute_elmo_embeddings()
        assert len(list_of_sentences) == len(embeddings)
        assert embeddings.shape == (len(list_of_sentences), 1024)
        np.testing.assert_array_almost_equal(expected_elmo_embeddings, embeddings)

    def test_compute_elmo_embeddings_is_empty_sentence(self):
        embedder = Embedder([])
        embeddings, _ = embedder.compute_elmo_embeddings()
        assert len(embeddings) == 0
        np.testing.assert_array_equal(np.array([], dtype=np.float64), embeddings)

    def test_compute_elmo_embeddings_contains_empty_sentence(self):
        embedder = Embedder(['Sentence one.', ''])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # avoids printing expected warning due to averaging an empty vector.
            embeddings, _ = embedder.compute_elmo_embeddings()
        assert embeddings.shape == (2, 1024)

    def test_compute_bert_embeddings(self, list_of_sentences, expected_bert_embeddings):
        embedder = Embedder(list_of_sentences)
        embeddings, _ = embedder.compute_bert_embeddings(model='bert-base-nli-stsb-mean-tokens')
        assert len(list_of_sentences) == len(embeddings)
        assert embeddings.shape == (len(list_of_sentences), 768)
        np.testing.assert_array_almost_equal(expected_bert_embeddings, embeddings, decimal=5)

    def test_compute_bert_embeddings_is_empty_sentence(self):
        embedder = Embedder([])
        embeddings, _ = embedder.compute_bert_embeddings(model='bert-base-nli-stsb-mean-tokens')
        assert len(embeddings) == 0
        np.testing.assert_array_equal(np.array([], dtype=np.float64), embeddings)

    def test_compute_bert_embeddings_contains_empty_sentence(self):
        embedder = Embedder(['Sentence one.', ''])
        embeddings, _ = embedder.compute_bert_embeddings(model='bert-base-nli-stsb-mean-tokens')
        assert embeddings.shape == (2, 768)

    def test_add_embeddings_to_corpus_df_from_csv(self, list_of_sentences, expected_df_with_embeddings):
        # expected_df = pd.concat([pd.Series(list_of_sentences), pd.Series(expected_elmo_embeddings.tolist())], axis=1)
        embedder = Embedder(list_of_sentences)
        output_df = embedder.add_embeddings_to_corpus_df(
            os.path.join(os.getenv("FIXTURES_DIR"), 'dummy_sentences.csv'),
            np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))),
            'dummy_embeddings'
        )
        pdt.assert_frame_equal(expected_df_with_embeddings.sort_index(axis=1), output_df.sort_index(axis=1))

    def test_add_embeddings_to_corpus_df_from_parquet(self, list_of_sentences, expected_df_with_embeddings):
        # expected_df = pd.concat([pd.Series(list_of_sentences), pd.Series(expected_elmo_embeddings.tolist())], axis=1)
        embedder = Embedder(list_of_sentences)
        output_df = embedder.add_embeddings_to_corpus_df(
            os.path.join(os.getenv("FIXTURES_DIR"), 'dummy_sentences.parquet'),
            np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))),
            'dummy_embeddings'
        )
        pdt.assert_frame_equal(expected_df_with_embeddings.sort_index(axis=1), output_df.sort_index(axis=1))

    def test_add_embeddings_to_corpus_df_from_df(self, list_of_sentences, expected_df_with_embeddings):
        # expected_df = pd.concat([pd.Series(list_of_sentences), pd.Series(expected_elmo_embeddings.tolist())], axis=1)
        embedder = Embedder(list_of_sentences)
        output_df = embedder.add_embeddings_to_corpus_df(
            pd.DataFrame({'dummy_sentences': ['First sentence.', 'Second sentence.', 'Third sentence.']}),
            np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))),
            'dummy_embeddings'
        )
        pdt.assert_frame_equal(expected_df_with_embeddings.sort_index(axis=1), output_df.sort_index(axis=1))

    def test_add_embeddings_to_corpus_df_with_emb_from_npy(self, list_of_sentences, expected_df_with_embeddings):
        # expected_df = pd.concat([pd.Series(list_of_sentences), pd.Series(expected_elmo_embeddings.tolist())], axis=1)
        embedder = Embedder(list_of_sentences)
        output_df = embedder.add_embeddings_to_corpus_df(
            pd.DataFrame({'dummy_sentences': ['First sentence.', 'Second sentence.', 'Third sentence.']}),
            os.path.join(os.getenv("FIXTURES_DIR"), 'dummy_embeddings.npy'),
            'dummy_embeddings'
        )
        pdt.assert_frame_equal(expected_df_with_embeddings.sort_index(axis=1), output_df.sort_index(axis=1))

    def test_add_embeddings_to_corpus_df_from_txt_raises_error(self, list_of_sentences):
        # expected_df = pd.concat([pd.Series(list_of_sentences), pd.Series(expected_elmo_embeddings.tolist())], axis=1)
        embedder = Embedder(list_of_sentences)
        with pytest.raises(TypeError):
            embedder.add_embeddings_to_corpus_df(
                os.path.join(os.getenv("FIXTURES_DIR"), 'dummy_sentences.txt'),
                np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))),
                'dummy_embeddings'
            )

    def test_add_embeddings_to_corpus_df_with_emb_from_list_raises_error(self, list_of_sentences):
        # expected_df = pd.concat([pd.Series(list_of_sentences), pd.Series(expected_elmo_embeddings.tolist())], axis=1)
        embedder = Embedder(list_of_sentences)
        with pytest.raises(TypeError):
            embedder.add_embeddings_to_corpus_df(
                os.path.join(os.getenv("FIXTURES_DIR"), 'dummy_sentences.csv'),
                [(np.array((1.0, 2.0, 3.0)), np.array((4.0, 5.0, 6.0)), np.array((7.0, 8.0, 9.0)))],
                'dummy_embeddings'
            )
