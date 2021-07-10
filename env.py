import os
import nltk


def install_nltk():
    # Download NLTK model data (you need to do this once)
    path = 'data/book'
    if not os.path.exists(path): 
        nltk.download("book", download_dir=path)
    else:
        nltk.data.path = [path]
        print("already exist.")

if __name__ == "__main__":
    install_nltk()