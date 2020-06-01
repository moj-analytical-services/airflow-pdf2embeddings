import os
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from pdf2embeddings.arrange_text import CorpusGenerator


class TestCorpusGenerator:
    def test_class_instantiation(self):
        input_df = pd.DataFrame({'col_1': [1, 2, 3], 'col_2': [4, 5, 6]}, columns=['col_1', 'col_2'])
        corpus_gen = CorpusGenerator(input_df)
        pdt.assert_frame_equal(corpus_gen.df, input_df)

    def test_df_by_page_to_df_by_sentence(self):
        expected_df_by_sentence = pd.DataFrame(
            {
                'sentence': [
                    'Mr Michael went to the store to buy some eggs.',
                    'Joel rolled down the street on his skateboard.',
                    'test / this is a first sentence',
                    'Take a look, then, at Tuesday\'s elections in New York '
                    'City, New Jersey and Virginia:'
                ],
                'pdf_name': ['test_pdf_1', 'test_pdf_1', 'test_pdf_1', 'test_pdf_2'],
                'page_number': [1, 1, 2, 1]
            }
        )
        df_by_page = pd.DataFrame(
            {
                'test_pdf_1': [
                    'Mr Michael went to the store to buy some eggs. Joel rolled down the street on his skateboard.',
                    'test / this is a first sentence'
                ],
                'test_pdf_2': [
                    'Take a look, then, at Tuesday\'s elections in New York City, New Jersey and Virginia:',
                    np.nan
                ]

            }
        )
        corpus_gen = CorpusGenerator(df_by_page)
        df_by_sentence = corpus_gen.df_by_page_to_df_by_sentence()

        pdt.assert_frame_equal(expected_df_by_sentence.sort_index(axis=1), df_by_sentence.sort_index(axis=1))
        # sort_index needed to ensure column orders is the same (pd.DataFrame column order is normally not important and
        # there is no guarantee that the two DataFrames will have the same order, in which case the assertion would fail
