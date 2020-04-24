import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.cluster.hierarchy import linkage, dendrogram

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
    print(wordsim353[(wordsim353['Word 1'] == w1) & (wordsim353['Word 2'] == w2)])
    print('Cosine: {}'.format(wm._get_cosine_similarity(_ppmi_df[w1], _ppmi_df[w2])))
    print('Jaccard: {}'.format(wm._get_weighted_jaccard_similarity(_ppmi_df[w1], _ppmi_df[w2])))
    print('Kullback-Leibler: {}'.format(wm._get_kullback_leibler_divergence(_ppmi_df[w1], _ppmi_df[w2], positive=False)))
    print('Jensen-Shannon: {}'.format(wm._get_jensen_shannon_divergence(_ppmi_df[w1], _ppmi_df[w2], positive=True)))


def main():
    corpus = hp.get_corpus('tiger', n=11)
    corpus = [[lm.lemmatize_sentence(sentence) for sentence in tk.custom_sentence_tokenize(doc)] for doc in corpus]
    word_set = _get_word_set(corpus)
    word_context_df = _get_word_context_matrix(corpus, word_set)
    ppmi_df = _get_pmi(word_context_df)
    compare_word(ppmi_df, 'tiger', 'cat')
    # wordsim_pairs = [('tiger', 'cat')]
    # draw(wordsim_pairs, ppmi_df, word_context_df, word_set)


main()
