import nltk
import re
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer


_NGRAM = 3


def get_word_ngrams(_text, ngram_size=_NGRAM):
    pattern = r'[a-zA-Zа-яА-ЯёЁЇїІіЄєҐґІіЎў]+'
    _tokenizer = RegexpTokenizer(pattern)
    _ngrams = []
    tokens = _tokenizer.tokenize(_text)
    for token in tokens:
        ngrams_list = list(ngrams(token.lower(), ngram_size, pad_left=True, pad_right=True))
        for ngram in ngrams_list:
            _ngrams.append(''.join([item if item is not None else '_' for item in ngram]))
    return _ngrams


def get_sentence_ngrams(_text, ngram=_NGRAM):
    pattern = r'[^a-zA-Zа-яА-Я]+'
    return list(ngrams(re.sub(pattern, ' ', _text).split(), ngram))


def custom_word_tokenize(_text):
    pattern = r'[a-zA-Zа-яА-Я]+'
    return re.findall(pattern, _text)


def custom_sentence_tokenize(_text):
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    return re.sub(pattern, '\n', _text).splitlines()


def sentence_tokenize(_text):
    return sent_tokenize(_text)


def word_tokenize(_text):
    return nltk.word_tokenize(_text)
