import text_mining.tokenization.tokenizer as tk
import text_mining.tf_idf.lemmatizer as lm
import text_mining.tf_idf.tf_idf as tf
import text_mining.helpers.helpers as hp


def _get_word_tfidf_series(_word, _df):
    return _df[_word]


def _analyse_corpus(_group_name, n):
    corpus = hp.get_corpus(_group_name, n)
    corpus = [[lm.lemmatize_sentence(sentence) for sentence in tk.custom_sentence_tokenize(doc)] for doc in corpus]
    print('Lemmatized corpus: \n{}'.format(corpus))
    df = tf.get_custom_dataframe_tf_idf(corpus)
    print('Custom tf-idf: \n{}'.format(df))
    skilearn_df = tf.get_skilearn_dataframe_tf_idf(corpus)
    print('Skilearn tf-idf: \n{}'.format(skilearn_df))
    return df


text_eng = "The plumage of the house sparrow is mostly different shades of grey and brown. The sexes exhibit strong " \
           "dimorphism: the female is mostly buffish above and below, while the male has boldly coloured head markings, " \
           "a reddish back, and grey underparts. The male has a dark grey crown from the top of its bill to its back, and " \
           "chestnut brown flanking its crown on the sides of its head. It has black around its bill, on its throat, and " \
           "on the spaces between its bill and eyes (lores). It has a small white stripe between the lores and crown and " \
           "small white spots immediately behind the eyes (postoculars), with black patches below and above them. The " \
           "underparts are pale grey or white, as are the cheeks, ear coverts, and stripes at the base of the head. The " \
           "upper back and mantle are a warm brown, with broad black streaks, while the lower back, rump and upper tail " \
           "coverts are greyish brown. The male is duller in fresh nonbreeding plumage, with whitish tips on many " \
           "feathers. Wear and preening expose many of the bright brown and black markings, including most of the black " \
           "throat and chest patch, called the 'bib' or 'badge'. The badge is variable in width and general size, and may " \
           "signal social status or fitness. This hypothesis has led to a 'veritable cottage industry' of studies, which " \
           "have only conclusively shown that patches increase in size with age. The male's bill is black in the breeding " \
           "season and dark grey during the rest of the year. The female has no black markings or grey crown. Its " \
           "upperparts and head are brown with darker streaks around the mantle and a distinct pale supercilium. Its " \
           "underparts are pale grey-brown. The female's bill is brownish-grey and becomes darker in breeding plumage " \
           "approaching the black of the male's bill. Juveniles are similar to the adult female, but deeper brown below " \
           "and paler above, with paler and less defined supercilia. Juveniles have broader buff feather edges, and tend " \
           "to have looser, scruffier plumage, like moulting adults. Juvenile males tend to have darker throats and white " \
           "postoculars like adult males, while juvenile females tend to have white throats. However, juveniles cannot be " \
           "reliably sexed by plumage: some juvenile males lack any markings of the adult male, and some juvenile females " \
           "have male features. The bills of young birds are light yellow to straw, paler than the female's bill. Immature " \
           "males have paler versions of the adult male's markings, which can be very indistinct in fresh plumage. By " \
           "their first breeding season, young birds generally are indistinguishable from other adults, though they may " \
           "still be paler during their first year."
text_eng1 = 'Some variation is seen in the 12 subspecies of house sparrows, which are divided into two groups, the ' \
            'Oriental P. d. indicus group, and the Palaearctic P. d. domesticus group. Birds of the P. d. domesticus ' \
            'group have grey cheeks, while P. d. indicus group birds have white cheeks, as well as bright colouration ' \
            'on the crown, a smaller bill, and a longer black bib.[19] The subspecies P. d. tingitanus differs little ' \
            'from the nominate subspecies, except in the worn breeding plumage of the male, in which the head is ' \
            'speckled with black and underparts are paler.[20] P. d. balearoibericus is slightly paler than the ' \
            'nominate, but darker than P. d. bibilicus.[21] P. d. bibilicus is paler than most subspecies, but has the' \
            ' grey cheeks of P. d. domesticus group birds. The similar P. d. persicus is paler and smaller, and P. d. ' \
            'niloticus is nearly identical but smaller. Of the less widespread P. d. indicus group subspecies, P. d. ' \
            'hyrcanus is larger than P. d. indicus, P. d. hufufae is paler, P. d. bactrianus is larger and paler, and ' \
            'P. d. parkini is larger and darker with more black on the breast than any other subspecies.'
text_rus = 'Прежде область обитания домового воробья ограничивалась Северной Европой. Впоследствии широко ' \
           'распространился в Европе и Азии (за исключением Арктики, северо-восточных, юго-восточных и центральных ' \
           'районов Азии), а также в Северной и Восточной Африке, Сенегале, Малой Азии, на Аравийском полуострове и ' \
           'острове Ява. В Италии обитает близкий вид - итальянский воробей (Passer italiae). Начиная с XX века был ' \
           'завезён в разные страны, широко там расселился и в настоящее время, кроме указанных выше мест, обитает ' \
           'также в Южной Африке, Австралии, Новой Зеландии, Северной и Южной Америке и на многих островах. Почти ' \
           'повсеместно является оседлой птицей, лишь из самых северных частей ареала на зиму откочевывает к югу (до ' \
           '1000 км), а из Средней Азии улетает в Переднюю Азию и Индию. Следуя за жильём человека, проник далеко ' \
           'на север в несвойственную для него зону лесотундры и даже тундры — до Мурманской области, устья Печоры, ' \
           'севера Якутии.'


# corpus_test = [text_eng, text_eng1]
# print(corpus_test)

df1 = _analyse_corpus('python', n=10)
df2 = _analyse_corpus('national_geographic', n=10)
# _analyse_corpus('simple_text')

# Compare dictionaries
print('##############')
columns_intersection = df1.columns.intersection(df2.columns)
columns_difference = df1.columns.difference(df2.columns)

epsilon = 0.04

df1_same = df1[columns_intersection].sum(axis=0).where(lambda x: x > epsilon).dropna()
df2_same = df2[columns_intersection].sum(axis=0).where(lambda x: x > epsilon).dropna()
print(df1_same)
print(df2_same)
print('##############')
df1_different = df1[columns_difference].sum(axis=0).where(lambda x: x > epsilon).dropna()
df2_different = df2.drop(columns_intersection, axis=1).sum(axis=0).where(lambda x: x > epsilon).dropna()
print(df1_different)
print(df2_different)


# print(df1.drop(df1.columns.difference(df2.columns), axis=1))
# print(type(df_intersection))
# print(df2.drop(df_intersection))
