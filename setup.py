import os
from setuptools import find_packages, setup
from pdf2embeddings.__init__ import __version__


def read(*paths):
    """Builds a file path from *paths and returns the content."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()


setup(
    name='pdf2embeddings',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=__version__,
    description='NLP tool for scraping text from a corpus of PDF files, embedding the sentences in the text and '
                'finding semantically similar sentences to a given search query.',
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url='https://github.com/moj-analytical-services/airflow-pdf2embeddings',
    download_url='https://pypi.python.org/pypi/pdf2embeddings',
    tests_require=["pytest", "flaky"],
    include_package_data=True,
    python_requires=">=3.6",
    license='MIT',
    install_requires=[
        "allennlp==0.9.0",
        "boto3~=1.10.34",
        "gensim==3.8.1",
        "nltk==3.4.5",
        "numpy==1.18.2",
        "pandas==0.25.3",
        "ply==3.11",
        "pyarrow==0.16.0",
        "pytest==5.4.1",
        "scikit-learn==0.22.1",
        "scipy==1.4.1",
        "sentence-transformers==0.2.5.1",
        "slate3k==0.5.3",
        "typing==3.7.4.1",
        "tqdm==4.45.0",
        "s3fs~=0.4.2",
    ],
    author='moj-analytical-services',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ]
)
