import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.cluster.hierarchy import linkage

import text_mining.tf_idf.lemmatizer as lm
import text_mining.helpers.helpers as hp
import text_mining.tokenization.tokenizer as tk
import text_mining.vectorization.metrics as wm


def _get_word_set(_corpus):
    _word_set = []
    for _document in _corpus:
        for _sentence in _document:
            for _word in list(set(_sentence)):
                if _word not in _word_set:
                    _word_set.append(_word)
    return _word_set


def _get_document_set(_corpus):
    _document_set = []
    for i in range(len(_corpus)):
        _document_set.append('document{}'.format(i))
    return _document_set


def _get_word_context_matrix(_corpus, _word_set, _window=5):
    word_count = len(_word_set)
    word_context_matrix = pd.DataFrame(np.zeros([word_count, word_count]), 
                                       columns=_word_set, 
                                       index=_word_set)
    for _document in _corpus:
        for _sentence in _document:
            for i, _word in enumerate(_sentence):
                for j in range(max(i - _window, 0), min(i + _window, len(_sentence))):
                    word_context_matrix[_word][_sentence[j]] += 1
    return word_context_matrix


def _get_document_context_matrix(_corpus, _word_set, _document_set, _window=5):
    word_count = len(_word_set)
    document_count = len(_corpus)
    document_context_matrix = pd.DataFrame(np.zeros([document_count, word_count]),
                                           columns=_word_set,
                                           index=_document_set)
    for i, _document in enumerate(_corpus):
        for _sentence in _document:
            for _word in _sentence:
                document_context_matrix.iloc[i][_word] += 1
    return document_context_matrix


def _get_pmi(df, ppmi=True):
    col_totals = df.sum(axis=0)
    row_totals = df.sum(axis=1)
    sum_value = col_totals.sum()
    expected = np.outer(row_totals, col_totals) / sum_value
    df = df / expected  # P(X_(ij)) ~ df / P(X_(i*)) * P(X_(*j))
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0
    if ppmi:
        df[df < 0] = 0.0
    return normalize_dataframe(df)


def get_linkages(_indeces, _words, _ppmi_df):
    cosine_weights = np.apply_along_axis(wm.get_cosine_similarity_linkage, 0, _indeces, _words, _ppmi_df)
    jaccard_weights = np.apply_along_axis(wm.get_weighted_jaccard_similarity_linkage, 0, _indeces, _words, _ppmi_df)
    kullback_leibler_weights = np.apply_along_axis(wm.get_kullback_leibler_divergence_linkage, 0, _indeces, _words, _ppmi_df,
                                                   positive=False)
    jensen_shannon_weights = np.apply_along_axis(wm.get_jensen_shannon_divergence_linkage, 0, _indeces, _words, _ppmi_df,
                                                 positive=True)
    return [linkage(cosine_weights, 'ward'),
            linkage(jaccard_weights, 'ward'),
            linkage(kullback_leibler_weights, 'ward'),
            linkage(jensen_shannon_weights, 'ward')]


def _draw_matrix_heatmap(df, _word_set):
    figure(figsize=(15, 10))
    df = pd.DataFrame(df,
                      index=_word_set,
                      columns=_word_set)
    sns.heatmap(df,
                annot=True,
                xticklabels=_word_set,
                yticklabels=_word_set,
                linewidths=.5,
                linecolor='black',
                cmap="Blues")
    plt.yticks(rotation=0)
    plt.title('Word-Context Matrix')
    plt.show()


def draw(_wordsim_pairs, _ppmi_df, _word_context_df, _word_set):
    # All-words dendogram
    hp.plot_dendrogram(linkage(_ppmi_df, 'ward'), 'Weighted matrix', _ppmi_df.columns)
    # Wordsim-pairs dendogram
    indeces = np.triu_indices(len(_ppmi_df.columns), 1)
    linkages = get_linkages(indeces, _wordsim_pairs, _ppmi_df)
    titles = ['Cosine', 'Jaccard', 'Kullback-Leibler', 'Jensen-Shannon']
    hp.plot_dendrogram(linkages, titles, _wordsim_pairs)
    # Heatmap word-context matrix
    _draw_matrix_heatmap(_word_context_df, _word_set)


def compare_word(_df, _w1, _w2, base=False, word2vec=False):
    if base:
        _w1_df, _w2_df = _df[_w1], _df[_w2]
    elif word2vec:
        pass
    else:
        _w1_df, _w2_df = _df.loc[:, _w1], _df.loc[:, _w2]

    wordsim353 = pandas.read_csv('docs/wordsim353.csv')
    wordsim353['Human (Mean)'] = normalize_series(wordsim353['Human (Mean)'])
    print('{} - {}'.format(_w1, _w2))
    wordsim = wordsim353[(wordsim353['Word 1'] == _w1) & (wordsim353['Word 2'] == _w2) |
                         (wordsim353['Word 1'] == _w2) & (wordsim353['Word 2'] == _w1)]
    print(wordsim)
    cosine = wm.get_cosine_similarity(_w1_df, _w2_df)
    print('Cosine: {}'.format(cosine))
    jaccard = wm.get_weighted_jaccard_similarity(_w1_df, _w2_df)
    print('Jaccard: {}'.format(jaccard))
    kullback_leibler = wm.get_kullback_leibler_divergence(_w1_df, _w2_df, positive=False)
    print('Kullback-Leibler: {}'.format(kullback_leibler))
    jensen_shannon = wm.get_jensen_shannon_divergence(_w1_df, _w2_df, positive=True)
    print('Jensen-Shannon: {}'.format(jensen_shannon))
    result = {'wordsim353': wordsim.iloc[0]['Human (Mean)'], 'cosine': cosine, 'jaccard': jaccard,
              'kullback_leibler': kullback_leibler, 'jensen_shannon': jensen_shannon}
    return result


def get_pearson_coef(_x, _y):
    x_avg = sum(_x) / len(_x)
    y_avg = sum(_y) / len(_y)
    numerator, denominator_x, denominator_y = 0, 0, 0
    for i in range(len(_x)):
        numerator += (_x[i] - x_avg) * (_y[i] - y_avg)
        denominator_x += (_x[i] - x_avg)**2
        denominator_y += (_y[i] - y_avg)**2
    return numerator / (denominator_x * denominator_y)**0.5


def normalize_dataframe(_df):
    _result = _df.copy()
    for _feature_name in _df.columns:
        _max_value = _df[_feature_name].max()
        _min_value = _df[_feature_name].min()
        _result[_feature_name] = (_df[_feature_name] - _min_value) / (_max_value - _min_value)
    return _result


def normalize_series(_series, min_max=True):
    if min_max:
        return (_series - _series.min()) / (_series.max() - _series.min())
    else:
        return (_series - _series.mean()) / _series.std()


def plsa(_corpus, _word_set, _document_set, number_of_topics=5, max_iterations=1):
    word_count = len(_word_set)
    document_count = len(_document_set)

    document_context_matrix = _get_document_context_matrix(_corpus, _word_set, _document_set)
    document_topic_probability = normalize_dataframe(
        pd.DataFrame(np.random.random(size=(document_count, number_of_topics)), index=_document_set))
    topic_word_probability = normalize_dataframe(pd.DataFrame(np.random.random(size=(number_of_topics, word_count)),
                                                              columns=_word_set))
    topic_probability = np.zeros([document_count, word_count, number_of_topics])

    # E-Step
    for iteration in range(max_iterations):
        for _i, _doc in enumerate(_document_set):
            for _j, _word in enumerate(_word_set):
                probability = document_topic_probability.loc[_doc, :] * topic_word_probability.loc[:, _word]
                topic_probability[_i][_j] = normalize_series(probability)
    # M-Step
    for _topic in range(number_of_topics):
        for _j, _word in enumerate(_word_set):
            result = 0
            for _i, _doc in enumerate(_document_set):
                result += document_context_matrix.loc[_doc][_word] * topic_probability[_i, _j, _topic]
            topic_word_probability.loc[_topic][_word] = result
        topic_word_probability.loc[_topic] = normalize_series(topic_word_probability.loc[_topic])
    for _i, _doc in enumerate(_document_set):
        for _topic in range(number_of_topics):
            result = 0
            for _j, _word in enumerate(_word_set):
                result += document_context_matrix.loc[_doc][_word] * topic_probability[_i, _j, _topic]
            document_topic_probability.loc[_doc][_topic] = result
        document_topic_probability.loc[_doc] = normalize_series(document_topic_probability.loc[_doc])
    return topic_word_probability


def show(_result, _method):
    wordsim_values, cosine_values, jaccard_values, kullback_leibler_values, jensen_shannon_values = [], [], [], [], []
    for dict_item in _result:
        wordsim_values.append(dict_item['wordsim353'])
        cosine_values.append(dict_item['cosine'])
        jaccard_values.append(dict_item['jaccard'])
        kullback_leibler_values.append(dict_item['kullback_leibler'])
        jensen_shannon_values.append(dict_item['jensen_shannon'])
    print(_method)
    print('Pearson coef (wordsim and cosine): {}'.format(get_pearson_coef(wordsim_values, cosine_values)))
    print('Pearson coef (wordsim and jaccard_values): {}'.format(get_pearson_coef(wordsim_values, jaccard_values)))
    print('Pearson coef (wordsim and kullback_leibler_values): {}'.format(get_pearson_coef(wordsim_values,
                                                                                           kullback_leibler_values)))
    print('Pearson coef (wordsim and jensen_shannon_values): {}'.format(get_pearson_coef(wordsim_values,
                                                                                         jensen_shannon_values)))


def main():
    dict_files = {('bank', 'money'): 9, ('book', 'paper'): 8, ('coffee', 'cup'): 8, ('computer', 'internet'): 8,
                  ('doctor', 'nurse'): 8, ('dollar', 'loss'): 8, ('football', 'soccer'): 8, ('phone', 'cell'): 8,
                  ('software', 'computer'): 8, ('tiger', 'cat'): 11}
    result_base, result_plsa, result_word2vec = [], [], []
    for group_names, count in dict_files.items():
        first_word, second_word = group_names
        corpus = hp.get_corpus(first_word, count)
        corpus = [[lm.lemmatize_sentence(sentence) for sentence in tk.custom_sentence_tokenize(doc)] for doc in corpus]

        word_set = _get_word_set(corpus)
        document_set = _get_document_set(corpus)

        result_plsa.append(compare_word(plsa(corpus, word_set, document_set), first_word, second_word))

        word_context_df = _get_word_context_matrix(corpus, word_set)
        ppmi_df = _get_pmi(word_context_df)
        result_base.append(compare_word(ppmi_df, first_word, second_word, base=True))

        # wordsim_pairs = [(first_word, second_word)]
        # draw(wordsim_pairs, ppmi_df, word_context_df, word_set)

    show(result_base, 'BASE METHOD')
    show(result_plsa, 'PLSA METHOD')
    # show(result_word2vec, 'WORD2VEC METHOD')


main()
