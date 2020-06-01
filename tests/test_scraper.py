import os
import numpy as np
import pandas as pd
from pdf2embeddings.scraper import DocumentScraper


class TestDocumentScraper:
    def test_class_instantiation(self, text_cleaning_json):
        scraper = DocumentScraper(
            os.getenv("FIXTURES_DIR"), os.path.join(os.getenv("FIXTURES_DIR"), 'words_to_replace.json')
        )
        assert scraper.pdf_folder == os.getenv("FIXTURES_DIR")
        assert scraper.open_json == text_cleaning_json

    def test_document_corpus_to_pandas_df(self):
        expected_scraped_df = pd.DataFrame(
            {'test_pdf_1': ['Mr Michael went to the store to buy some eggs. Joel rolled down the street on his '
                            'skateboard.', 'test / this is a first sentence'],
             'test_pdf_2': ['Take a look, then, at Tuesday\'s elections in New York City, New Jersey and Virginia:',
                            np.nan]}
        )
        scraper = DocumentScraper(
            os.getenv("FIXTURES_DIR"), os.path.join(os.getenv("FIXTURES_DIR"), 'words_to_replace.json')
        )
        scraped_df = scraper.document_corpus_to_pandas_df()
        pd.testing.assert_frame_equal(expected_scraped_df.sort_index(axis=1), scraped_df.sort_index(axis=1),
                                      check_index_type=False)
