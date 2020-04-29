import pandas as pd
import text_mining.tokenization.tokenizer as tk


def get_ngram_frequency(_ngrams):
    _series = pd.Series()
    for _ngram in _ngrams:
        if _ngram not in _series:
            _series[_ngram] = 1
        else:
            _series[_ngram] += 1
    return _series.sort_values(ascending=False)


def get_distance(_pretrained_series, _current_series):
    _distance = 0
    for ngram in _current_series.index:
        if ngram not in _pretrained_series.index:
            _distance += abs(_current_series.size - _current_series[ngram])
        else:
            _distance += abs(_pretrained_series[ngram] - _current_series[ngram])
    return _distance


def train_models(_languages, pretrainded_head=500):
    _pretrainded_models = {}
    for _language in _languages:
        _corpus = ''
        for i in range(1, 4):
            with open('docs/{}/{}{}.txt'.format(_language, _language, i), encoding='utf-8') as _f:
                _corpus += _f.read().rstrip()
        _ngrams = tk.get_word_ngrams(_corpus)
        _pretrained = get_ngram_frequency(_ngrams).head(pretrainded_head)
        _pretrained.to_csv('{}.csv'.format(_language), header=False)
        _pretrainded_models[_language] = _pretrained
    return _pretrainded_models


def read_models(_languages):
    _pretrainded_models = {}
    for _language in _languages:
        _pretrained = pd.read_csv('{}.csv'.format(_language), index_col=0, header=None).iloc[:, 0]
        _pretrainded_models[_language] = _pretrained
    return _pretrainded_models


def get_score(_pretrainded_models, _current_model, n=3):
    _result = {}
    _total_distance = 0
    for _language, _model in _pretrainded_models.items():
        _distance = get_distance(_model, _current_model)
        _result[_language] = _distance
        _total_distance += _distance
    for _item in _result.keys():
        _result[_item] = abs(_result[_item] - int(_total_distance / n))
    return _result


languages = ['rus', 'blr', 'ukr']
# pretrainded_models = train_models(languages)
pretrainded_models = read_models(languages)


with open('docs/text.txt', encoding='utf-8') as f:
    text = f.read().rstrip()
current_model = tk.get_word_ngrams(text)
current_model_frequency = get_ngram_frequency(current_model)

score = get_score(pretrainded_models, current_model_frequency)
print('Score: {}'.format(score))
print('Text in {} language'.format([k for k in score if score[k] == max(score.values())][0]))
