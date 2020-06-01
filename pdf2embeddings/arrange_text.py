import pandas as pd
import numpy as np
import nltk
import logging


logger = logging.getLogger(__name__)


class CorpusGenerator:
    """
    This class contains methods for converting a "horizontal" pd.DataFrame or scraped text from files, where:
        - each column contains the text of a separate pdf file, and
        - each row within a column contains the text of one page within that pdf
    into a new "vertical" pd.DataFrame with only 3 columns:
        - 'sentence': one row per each sentence in the full corpus
        - 'pdf_name': the name of the PDF file each sentence is taken from
        - 'page_number': the page number where the sentence can be found within the file 'pdf_name'.
           NOTE: sentences that span across multiple pages have been split so no sentence can run across the next page.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        """
        :param df: the input pd.DataFrame as described in class docstring.
        """
        self.df = df

    def df_by_page_to_df_by_sentence(self) -> pd.DataFrame:
        """
        :return: df_by_sentence: the output pd.DataFrame as described in class docstring.
        """
        logger.info('Starting DataFrame conversion...')
        full_corpus_list = [
            {'sentence': sentence,
             'pdf_name': pdf_name,
             'page_number': page_i + 1}
            for pdf_name in self.df.columns
            for page_i, text in enumerate(self.df[pdf_name]) if isinstance(text, str)
            for sentence in nltk.sent_tokenize(text)
        ]
        # isinstance(text, str) is required to handle np.nan values: when a PDF has less pages than the longest
        # PDF, all rows corresponding to pages after the last page of that PDF are filled with np.nan.
        df_by_sentence = pd.DataFrame(full_corpus_list)
        df_by_sentence['sentence'].replace('', np.nan, inplace=True)
        df_by_sentence.dropna(axis=0, how='any', inplace=True)
        # this gets rid of lots of empty sentences when original file contains multiple dots '.....' (e.g. in table
        # content), which are empty because for every '.' a new sentence is generated.
        logger.info('New DataFrame created.')
        return df_by_sentence
