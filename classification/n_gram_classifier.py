import numpy as np
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


def train_models(_languages, pretrainded_head=5000):
    _pretrainded_models = {}
    for _language in _languages:
        _corpus = ''
        for _i in range(1, 4):
            with open('docs/{}/{}{}.txt'.format(_language, _language, _i), encoding='utf-8') as _f:
                _corpus += _f.read().rstrip()
        _ngrams = tk.get_word_ngrams(_corpus)
        _pretrained = get_ngram_frequency(_ngrams).head(pretrainded_head)
        _pretrained.to_csv('{}.csv'.format(_language), header=False)
        _pretrainded_models[_language] = _pretrained
    return _pretrainded_models


def read_models(_languages, _trained_set):
    _pretrainded_models = {}
    for _language in _languages:
        _pretrained = pd.read_csv('{}_{}.csv'.format(_language, _trained_set), index_col=0, header=None).iloc[:, 0]
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


def main():
    languages = ['rus', 'blr', 'ukr']
    # pretrainded_models = train_models(languages)
    pretrainded_models = read_models(languages, 500)

    df = pd.DataFrame(np.zeros([len(languages), len(languages)]),
                      index=languages,
                      columns=languages)
    labels = pd.read_csv('docs/y.csv').set_index('name')

    print('Count text dataframe: \n{}'.format(labels.loc[:, 'language'].value_counts()))
    for i in range(1, 12):
        with open('docs/text{}.txt'.format(i), encoding='utf-8') as f:
            text = f.read().rstrip()
        current_model = tk.get_word_ngrams(text)
        current_model_frequency = get_ngram_frequency(current_model)
        score = get_score(pretrainded_models, current_model_frequency)
        predicted_language = [k for k in score if score[k] == max(score.values())][0]
        actual_language = labels.loc['text{}.txt'.format(i), 'language']
        print('Predicted: text{} in {} language'.format(i, predicted_language))
        print('Predicted score: {}'.format(score))
        print('Actual: text{} in {} language'.format(i, actual_language))
        if predicted_language is actual_language:
            df.loc[predicted_language][predicted_language] += 1
        else:
            df.loc[predicted_language][actual_language] += 1
    print('Resulted dataframe: \n{}'.format(df))
    with np.errstate(divide='ignore'):
        for language in languages:
            recall = df.loc[language][language] / sum(df[language])
            precision = df.loc[language][language] / sum(df.loc[language])
            f1_score = 2 * precision * recall / (precision + recall)
            print('Recall {}-class: {}'.format(language, recall))
            print('Precision {}-class: {}'.format(language, precision))
            print('F1-score {}-class: {}'.format(language, f1_score))
        true_positive = sum(np.diag(df))
        total_without_true_positive = df.mask(np.eye(3, dtype=bool)).fillna(0.0).values.sum()
        accuracy = true_positive / (total_without_true_positive + true_positive)
        print('Accuracy: {}'.format(accuracy))


# main()
