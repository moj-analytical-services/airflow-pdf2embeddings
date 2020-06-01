import os
import numpy as np
from pdf2embeddings.scraper import DocumentScraper
from pdf2embeddings.arrange_text import CorpusGenerator
from pdf2embeddings.embedder import Embedder
from pdf2embeddings.process_user_queries import query_embeddings


def test_end_to_end_runner():
    scraper = DocumentScraper(
        os.getenv("FIXTURES_DIR"), os.path.join(os.getenv("FIXTURES_DIR"), 'words_to_replace.json')
    )
    df_by_page = scraper.document_corpus_to_pandas_df()
    generator = CorpusGenerator(df_by_page)
    df_by_sentence = generator.df_by_page_to_df_by_sentence()
    list_of_sentences = df_by_sentence['sentence'].values.tolist()
    assert list_of_sentences == [
        'Mr Michael went to the store to buy some eggs.',
        'Joel rolled down the street on his skateboard.',
        'test / this is a first sentence',
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia:"
    ]

    embedder = Embedder(list_of_sentences)
    models_to_be_run = ['Word2Vec_tfidf_weighted', 'Word2Vec', 'BERT', 'ELMo']
    for model in models_to_be_run:
        if model == 'Word2Vec_tfidf_weighted':
            sentence_embeddings, w2v_tfidf, tfidf_vectorizer = embedder.compute_word2vec_embeddings(tfidf_weights=True)
            df_by_sentence = embedder.add_embeddings_to_corpus_df(
                df_by_sentence, sentence_embeddings, 'Word2Vec_with_TfIdf_weights'
            )
        elif model == 'Word2Vec':
            sentence_embeddings, w2v, _ = embedder.compute_word2vec_embeddings(tfidf_weights=False)
            df_by_sentence = embedder.add_embeddings_to_corpus_df(df_by_sentence, sentence_embeddings, 'Word2Vec')
        elif model == 'BERT':
            bert_model = 'bert-base-nli-stsb-mean-tokens'  # This line is specific to BERT
            sentence_embeddings, bert = embedder.compute_bert_embeddings(bert_model)
            df_by_sentence = embedder.add_embeddings_to_corpus_df(df_by_sentence, sentence_embeddings, 'BERT')
        elif model == 'ELMo':
            sentence_embeddings, elmo = embedder.compute_elmo_embeddings()
            df_by_sentence = embedder.add_embeddings_to_corpus_df(df_by_sentence, sentence_embeddings, 'ELMo_layer_3')
        else:
            raise KeyError(f'The model {model} is not recognized as input.')

    w2v_emb, df_by_sentence = query_embeddings(
        list_of_sentences[0], df_by_sentence, 'Word2Vec', 'Word2Vec', w2v, metric_colname='w2v_distance_test1'
    )
    w2v_tfidf_emb, df_by_sentence = query_embeddings(
        list_of_sentences[0], df_by_sentence, 'Word2Vec_with_TfIdf_weights', 'Word2Vec_TfIdf_weighted', w2v_tfidf,
        metric_colname='w2v_tfidf_weighted_distance_test1', tfidf_vectorizer=tfidf_vectorizer
    )
    elmo_emb, df_by_sentence = query_embeddings(
        list_of_sentences[0], df_by_sentence, 'ELMo_layer_3', 'ELMo', elmo, metric_colname='elmo_distance_test1'
    )
    bert_emb, df_by_sentence = query_embeddings(
        list_of_sentences[0], df_by_sentence, 'BERT', 'BERT', bert, metric_colname='bert_distance_test1'
    )

    df_by_sentence.sort_values('w2v_distance_test1', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == "Mr Michael went to the store to buy some eggs."
    np.testing.assert_array_equal(w2v_emb, df_by_sentence['Word2Vec'][0])

    df_by_sentence.sort_values('w2v_tfidf_weighted_distance_test1', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == "Mr Michael went to the store to buy some eggs."
    np.testing.assert_array_equal(w2v_tfidf_emb, df_by_sentence['Word2Vec_with_TfIdf_weights'][0])

    df_by_sentence.sort_values('elmo_distance_test1', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == "Mr Michael went to the store to buy some eggs."
    # np.testing.assert_array_almost_equal(elmo_emb, df_by_sentence['ELMo_layer_3'][0])
    # This test does not work, see https://github.com/allenai/allennlp/issues/3995#

    df_by_sentence.sort_values('bert_distance_test1', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == "Mr Michael went to the store to buy some eggs."
    np.testing.assert_array_almost_equal(bert_emb, df_by_sentence['BERT'][0])

    w2v_emb, df_by_sentence = query_embeddings(
        "New York", df_by_sentence, 'Word2Vec', 'Word2Vec', w2v, metric_colname='w2v_distance_test2'
    )
    w2v_tfidf_emb, df_by_sentence = query_embeddings(
        "New York", df_by_sentence, 'Word2Vec_with_TfIdf_weights', 'Word2Vec_TfIdf_weighted', w2v_tfidf,
        metric_colname='w2v_tfidf_weighted_distance_test2', tfidf_vectorizer=tfidf_vectorizer
    )
    elmo_emb, df_by_sentence = query_embeddings(
        "New York", df_by_sentence, 'ELMo_layer_3', 'ELMo', elmo, metric_colname='elmo_distance_test2'
    )
    bert_emb, df_by_sentence = query_embeddings(
        "New York", df_by_sentence, 'BERT', 'BERT', bert, metric_colname='bert_distance_test2'
    )

    df_by_sentence.sort_values('w2v_distance_test2', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia:"

    df_by_sentence.sort_values('w2v_tfidf_weighted_distance_test2', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia:"

    df_by_sentence.sort_values('elmo_distance_test2', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia:"

    df_by_sentence.sort_values('bert_distance_test2', ascending=True, inplace=True)
    df_by_sentence.reset_index(inplace=True, drop=True)
    assert df_by_sentence['sentence'][0] == \
        "Take a look, then, at Tuesday's elections in New York City, New Jersey and Virginia:"
