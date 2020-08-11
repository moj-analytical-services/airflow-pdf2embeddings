import s3fs
import slate3k
import logging
from tqdm import tqdm
from boto3 import Session
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
    It also offers support for folders stored in the cloud (AWS S3 buckets only).
    """
    def __init__(self, pdf_folder: str, json_filename: Optional[str] = None, from_s3_bucket: bool = False) -> None:
        """
        :param pdf_folder: path to the folder containing pdf files to be scraped. Can also be an S3 bucket (see below).
        :param json_filename: full path of the json file created by the module json_creator.py. This json file
               contains dictionary of words to replace (e.g. Dr. --> Dr), used for text cleaning. Defaults to None, in
               which case no ad-hoc text cleaning will be performed.
        :param from_s3_bucket: a boolean specifying whether to scrape the PDFs from a folder located in an AWS S3
               bucket. If set to True, the path can either start with "s3://" or omit this prefix. Default: False.
        """
        self.pdf_folder = pdf_folder
        self.open_json = self._read_config(json_filename)
        self.from_s3_bucket = from_s3_bucket

        if self.from_s3_bucket:
            assert Session().get_credentials() is not None, "You do not have any valid credentials to access AWS S3."
            assert s3fs.S3FileSystem().isdir(pdf_folder), \
                f"The directory you specified, {pdf_folder} does not seem to be a valid S3 path you have access to."
            logger.warning("AWS S3 bucket detected: PDFs will be scraped from S3 bucket rather than local storage.")
        if not self.from_s3_bucket and not os.path.isdir(pdf_folder):
            raise FileNotFoundError(
                f"No such directory: {pdf_folder}. If you intended to read PDFs from an S3 bucket please set "
                f"from_s3_bucket = True when instantiating DocumentScraper."
            )

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
        else:
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
        dir_list = os.listdir(self.pdf_folder) if not self.from_s3_bucket else s3fs.S3FileSystem().ls(self.pdf_folder)
        pdf_list = [pdf for pdf in dir_list if pdf.endswith('.pdf')]  # excluding non .pdf files
        not_pdf_list = [pdf for pdf in dir_list if not pdf.endswith('.pdf')]
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