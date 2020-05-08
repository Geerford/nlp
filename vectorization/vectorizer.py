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
    return df


def get_linkages(_indeces, _words, _ppmi_df):
    cosine_weights = np.apply_along_axis(wm.get_cosine_similarity, 0, _indeces, _words, _ppmi_df)
    jaccard_weights = np.apply_along_axis(wm.get_weighted_jaccard_similarity, 0, _indeces, _words, _ppmi_df)
    kullback_leibler_weights = np.apply_along_axis(wm.get_kullback_leibler_divergence, 0, _indeces, _words, _ppmi_df,
                                                   positive=False)
    jensen_shannon_weights = np.apply_along_axis(wm.get_jensen_shannon_divergence, 0, _indeces, _words, _ppmi_df,
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


def compare_word(_ppmi_df, w1, w2):
    wordsim353 = pandas.read_csv('docs/wordsim353.csv')
    print('{} - {}'.format(w1, w2))
    wordsim = wordsim353[(wordsim353['Word 1'] == w1) & (wordsim353['Word 2'] == w2) |
                         (wordsim353['Word 1'] == w2) & (wordsim353['Word 2'] == w1)]
    print(wordsim)
    cosine = wm._get_cosine_similarity(_ppmi_df[w1], _ppmi_df[w2])
    print('Cosine: {}'.format(cosine))
    jaccard = wm._get_weighted_jaccard_similarity(_ppmi_df[w1], _ppmi_df[w2])
    print('Jaccard: {}'.format(jaccard))
    kullback_leibler = wm._get_kullback_leibler_divergence(_ppmi_df[w1], _ppmi_df[w2], positive=False)
    print('Kullback-Leibler: {}'.format(kullback_leibler))
    jensen_shannon = wm._get_jensen_shannon_divergence(_ppmi_df[w1], _ppmi_df[w2], positive=True)
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


def main():
    dict_files = {('bank', 'money'): 9, ('book', 'paper'): 8, ('coffee', 'cup'): 8, ('computer', 'internet'): 8,
                  ('doctor', 'nurse'): 8, ('dollar', 'loss'): 8, ('football', 'soccer'): 8, ('phone', 'cell'): 8,
                  ('software', 'computer'): 8, ('tiger', 'cat'): 11}
    result = []
    for group_names, count in dict_files.items():
        first_word, second_word = group_names
        corpus = hp.get_corpus(first_word, count)
        corpus = [[lm.lemmatize_sentence(sentence) for sentence in tk.custom_sentence_tokenize(doc)] for doc in corpus]
        word_set = _get_word_set(corpus)
        word_context_df = _get_word_context_matrix(corpus, word_set)
        ppmi_df = _get_pmi(word_context_df)
        result.append(compare_word(ppmi_df, first_word, second_word))
        # wordsim_pairs = [(first_word, second_word)]
        # draw(wordsim_pairs, ppmi_df, word_context_df, word_set)

    wordsim_values, cosine_values, jaccard_values, kullback_leibler_values, jensen_shannon_values = [], [], [], [], []

    for dict_item in result:
        wordsim_values.append(dict_item['wordsim353'])
        cosine_values.append(dict_item['cosine'])
        jaccard_values.append(dict_item['jaccard'])
        kullback_leibler_values.append(dict_item['kullback_leibler'])
        jensen_shannon_values.append(dict_item['jensen_shannon'])
    print('###################')
    print('Pearson coef (wordsim and cosine): {}'.format(get_pearson_coef(wordsim_values, cosine_values)))
    print('Pearson coef (wordsim and jaccard_values): {}'.format(get_pearson_coef(wordsim_values, jaccard_values)))
    print('Pearson coef (wordsim and kullback_leibler_values): {}'.format(get_pearson_coef(wordsim_values,
                                                                                           kullback_leibler_values)))
    print('Pearson coef (wordsim and jensen_shannon_values): {}'.format(get_pearson_coef(wordsim_values,
                                                                                         jensen_shannon_values)))


main()
