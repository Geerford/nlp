import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from nltk.corpus import wordnet as wn


def get_lower_case(_a, _b):
    return [i.lower() for i in _a], [i.lower() for i in _b]


def get_matches(_a, _b, _scan_window, i, j, j_count, m, t_matches):
    if j >= len(_b):
        j = len(_b) - 1
    j_count += 1
    if _a[i] == _b[j] and i != j and i - j < _scan_window:
        m += 1
        t_matches.append([i, j])
    return j, j_count, m


def get_scan_window(_a, _b):
    return round(max(len(_a), len(_b)) / 2) - 1


def get_prefix_length(_a, _b):
    _a, _b = get_lower_case(_a, _b)
    count = 0
    recommended_value = 4
    for i in range(len(_a)):
        if i < len(_b) and _a[i] == _b[i]:
            count += 1
    return min(count, recommended_value)


def get_stopwords(rus=False):
    if rus:
        stopwords = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
                     'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
                     'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
                     'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
                     'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
                     'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
                     'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
                     'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
                     'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой',
                     'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
                     'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
                     'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между',
                     'кем', 'кому']

    else:
        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                     'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                     'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                     "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                     "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                     'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                     'won', "won't", 'wouldn', "wouldn't", 'u', 'would']
    return stopwords


def tag_map(_tag):
    if _tag is 'J':
        return wn.ADJ
    elif _tag is 'V':
        return wn.VERB
    elif _tag is 'R':
        return wn.ADV
    else:
        return wn.NOUN


def plot_dendrogram(linkages, title, words):
    template_title = 'Hierarchical Clustering Dendrogram - {}'
    titles = []
    if isinstance(linkages, list):
        for i in range(len(linkages)):
            titles.append(template_title.format(title[i]))
            dendrogram(linkages[i],
                       labels=np.array(words),
                       orientation="right",
                       leaf_font_size=8)
            plt.title(titles[i])
            plt.show()
    else:
        titles.append(template_title.format(title))
        dendrogram(linkages,
                   labels=np.array(words),
                   orientation="right",
                   leaf_font_size=8)
        plt.title(titles)
        plt.show()

def get_corpus(_group_name, n):
    corpus = []
    for i in range(1, n + 1):
        with open('docs/{}{}.txt'.format(_group_name, i), encoding='utf-8') as f:
            corpus.append(f.read().rstrip())
    return corpus
