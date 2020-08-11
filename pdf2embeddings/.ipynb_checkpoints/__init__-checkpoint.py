try:
    from pdf2embeddings import scraper, arrange_text, embedder, process_user_queries
except ModuleNotFoundError as err:
    print(f"You have not imported all the required packages. \n{err}")
__version__ = '0.2.4'
