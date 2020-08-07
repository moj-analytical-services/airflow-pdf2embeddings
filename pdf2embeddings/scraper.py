import s3fs
import slate3k
import logging
from tqdm import tqdm
import os
import pandas as pd
import json
from typing import Tuple, Dict, Optional


logger = logging.getLogger(__name__)


class DocumentScraper:
    """
    This class has methods for scraping all the files with '.pdf' extension within a given folder, cleaning the text,
    and generating a pd.DataFrame where:
        - each column contains the text of a separate pdf file (column name is pdf title without extension), and
        - each row within a column contains the text of one page within that pdf.
    If different pdf files have different number of pages, any empty rows at the bottom of a column are filled with nan.
    """
    def __init__(self, pdf_folder: str, json_filename: Optional[str] = None) -> None:
        """
        :param pdf_folder: path to folder containing pdf files to be scraped
        :param json_filename: full path of the json file created by the module json_creator.py. This json file
               contains dictionary of words to replace (e.g. Dr. --> Dr), used for text cleaning. Defaults to None, in
               which case no ad-hoc text cleaning will be performed.
        """
        self.pdf_folder = pdf_folder
        self.open_json = self._read_config(json_filename)
        self.from_s3_bucket = None

    @staticmethod
    def _read_config(json_filename: Optional[str]) -> Dict[str, str]:
        """
        :param json_filename: json filename to be deserialized.
        :return: the dictionary from json object. If json_filename is None, and empty dictionary will be returned.
        """
        if json_filename is None:
            logger.warning('No .json file for text cleaning was provided. Ad-hoc text cleaning will not be performed.')
            return dict()
        logger.info(f'Reading {json_filename} file for text cleaning.')
        assert '.json' in json_filename, 'The json_filename provided does not correspond to a .json file.'
        with open(json_filename, 'r') as file:
            return json.load(file)

    def _text_to_series_of_pages(self, pdf_name: str) -> Tuple[pd.Series, int]:
        """
        :param pdf_name: full name of pdf (including .pdf extension) to be scraped and converted into a pd.Series
        :return: document_series: a pd.Series where each row contains the text of one pdf page.
                 num_pages: int, the number of pages of the input pdf file
        """
        assert pdf_name.endswith('.pdf'), 'Input file is not in .pdf format. The file cannot be processed.'
        document_series = pd.Series()
        if not self.from_s3_bucket:
            pdf = open(os.path.join(self.pdf_folder, pdf_name), 'rb')
        if self.from_s3_bucket:
            pdf = s3fs.S3FileSystem().open(pdf_name, 'rb')  # no need to join with self.pdf_folder as s3fs includes that
        pdf_reader = slate3k.PDF(pdf)
        num_pages = len(pdf_reader)
        for i, page in enumerate(pdf_reader):
            logger.debug(f'Reading page {i+1} of PDF file {pdf_name}')
            page_text = self._clean_text(page)
            page_series = pd.Series(page_text)
            document_series = document_series.append(page_series, ignore_index=True)
        pdf.close()

        return document_series, num_pages

    def _clean_text(self, text: str) -> str:
        """
        :param text: the text to be cleaned. This replaces certain words based on the dict self.open_json
        :return: text: the cleaned text.
        """
        for k, v in self.open_json.items():
            text = text.replace(k, v)
        text = text.strip()
        return text

    def document_corpus_to_pandas_df(self) -> pd.DataFrame:
        """
        This method can be called by the user to generate the final pd.DataFrame as described in class docstring.
        :return: df: a pd.DataFrame. See class docstring.
        """
        df = pd.DataFrame()
        try:
            pdf_list = [pdf for pdf in os.listdir(self.pdf_folder) if pdf.endswith('.pdf')]  # excluding non .pdf files
            not_pdf_list = [pdf for pdf in os.listdir(self.pdf_folder) if not pdf.endswith('.pdf')]
            self.from_s3_bucket = False
        except FileNotFoundError:
            try:
                pdf_list = [pdf for pdf in s3fs.S3FileSystem().ls(self.pdf_folder) if pdf.endswith('.pdf')]
                not_pdf_list = [pdf for pdf in s3fs.S3FileSystem().ls(self.pdf_folder) if not pdf.endswith('.pdf')]
                self.from_s3_bucket = True
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"{err}. We also tried to look for an S3 bucket path but could not find any. Other types of cloud "
                    f"storage are not natively supported by pdf2emb_nlp."
                )
        if len(not_pdf_list) > 0:
            logger.warning(
                f'\nThe following files were present in the directory {self.pdf_folder}, but were not scraped as they '
                f'are not in .pdf format: \n{not_pdf_list}'
            )
        logger.info('Starting scraping PDFs...')
        for i, file in enumerate(tqdm(sorted(pdf_list))):
            # sorted is so pdfs are extracted in alphabetic order, and to make testing more robust.
            series, num_pages = self._text_to_series_of_pages(file)
            logger.info(f"Reading PDF file {i + 1} out of {len(pdf_list)}: \"{file}\", number of pages: {num_pages}")
            if isinstance(series, pd.Series):
                series.rename(file.replace('.pdf', ''), inplace=True)
                df = pd.concat([df, series], axis=1)
        return df
