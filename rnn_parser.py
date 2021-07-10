import csv
from env import install_nltk
import nltk
import itertools
from contstants import vocabulary_size, unknown_token, sentence_start_token, sentence_end_token


def parser() -> tuple:
    print("Reading CSV file...")
    with open('data/reddit-comments-2015-08.csv', 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        # Split full comments into sentences
        # https://stackoverflow.com/questions/5239856/asterisk-in-function-call
        sentences = itertools.chain(
            *[nltk.sent_tokenize(row[0].lower()) for row in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (
        vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [
            w if w in word_to_index else unknown_token for w in sent]

    return (index_to_word, word_to_index, tokenized_sentences)


if __name__ == "__main__":
    install_nltk()
    index_to_word, word_to_index, tokenized_sentences = parser()
    print(tokenized_sentences[0])
    #print("\nExample sentence: '%s'" % sentences[0])
    print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])