import DB, Tree

MIN_TYPES = 0
MAX_TYPES = 13 #allways + 1


def get_data_id(data_name):
    data ={ 'Futbol':       0,
              'Baloncesto': 1,
              'Golf':       2,
              'Boxeo':      3,
              'Judo':       4,
              'Balonmano':  5,
              'Tenis':      6,
              'Ciclismo':   7,
              'Atletismo':  8,
              'Voleibol':   9,
              'Rugby':      10,
              'Motociclismo':     11,       #formula1
              'Futbol americano': 12,}

    return data[data_name.lower()]

def get_data_name(data_id):
    data ={ 0: 'Futbol',
              1: 'Baloncesto',
              2: 'Golf',
              3: 'Boxeo',
              4: 'Judo',
              5: 'Balonmano',
              6: 'Tenis',
              7: 'Ciclismo',
              8: 'Atletismo',
              9: 'Voleibol',
              10: 'Rugby',
              11: 'Motociclismo',
              12: 'Futbol americano'}

    return data[data_id]

def get_data_id_lower(data_name):
    data ={ 'futbol':       0,
              'baloncesto': 1,
              'golf':       2,
              'boxeo':      3,
              'judo':       4,
              'balonmano':  5,
              'tenis':      6,
              'ciclismo':   7,
              'atletismo':  8,
              'voleibol':   9,
              'rugby':      10,
              'motociclismo':     11, #formula1
              'futbol americano': 12,}

    return data[data_name.lower()]

def get_sentiment_id(data_name):
    data ={   'alegria':    0,
              'amor':       1,
              'enfado':     2,
              'miedo':      3,
              'sorpresa':   4,
              'tristeza':   5
            }
    return data[data_name.lower()]

def get_sentiment_name(data_id):
    data ={   0: 'alegria',
              1: 'amor',
              2: 'enfado',
              3: 'miedo',
              4: 'sorpresa',
              5: 'tristeza'
            }

    return data[data_id]


class WikiData:
    def __init__(self, text, word_list, top_words, top_words_percentages, type_name):
        self.text = text
        self.word_list = word_list
        self.top_words = top_words
        self.top_words_percentages = top_words_percentages
        self.url = None
        self.type_name = type_name

    def getTopWordsString(self):
        text = ''
        for word in self.top_words:
            text += word + ' '
        return text


class Dictionary:
    def __init__(self, type, key_words, secondary_words, excluding_Words):
        self.type_name = type
        self.key_words = key_words
        self.secondary_words = secondary_words
        self.excluding_words = excluding_Words

class DictionaryTree:
    def __init__(self, type, key_words, secondary_words, excluding_words):
        self.type_name = type
        key_words_tree = Tree.AVLTree()
        self.key_words = key_words_tree.insert_array(key_words)
        secondary_words_tree = Tree.AVLTree()
        self.secondary_words = secondary_words_tree.insert_array(secondary_words)
        excluding_Words_tree = Tree.AVLTree()
        self.excluding_words = excluding_Words_tree.insert_array(excluding_words)

class Tweet:
    def __init__(self, text, length, date, source, likes, retweets, stemmed):
        self.text = text
        self.length = length
        self.date = date
        self.source = source
        self.likes = likes
        self.retweets = retweets
        self.sa = 'NEUTRAL'
        self.stemmed = stemmed
        self.theme = 'SPORT'

class SA:
    def __init__(self, alegria, amor, enfado, miedo, sorpresa, tristeza):
        self.alegria = alegria
        self.amor = amor
        self.enfado = enfado
        self.miedo = miedo
        self.sorpresa = sorpresa
        self.tristeza = tristeza

class Sentiment:
    def __init__(self, sentiment, words_list):
        self.sentiment = sentiment
        self.words_list = words_list

class User:
    def __init__(self, description, verified, followers_count):
        self.description = description
        self.verified = verified
        self.followers_count = followers_count


def get_empty_words_from_file():
    # Palabras que no se tendran en cuenta:
    fichero = open ('./data/palabras_vacias.txt',"r", encoding="utf-8")
    empty_words = fichero.read().lower()
    empty_words = empty_words.split()
    return empty_words

def delete_text_punctuation(text):
    import re, string
    # elimina signos de puntuacion y numeros
    text = re.sub('[%s,\d]' % re.escape(string.punctuation), ' ', text)
    # quitar tildes LO MALO ES QUE BORRA LAS Ñ
    #text = ''.join((c for c in unicodedata.normalize('NFD', natacion) if unicodedata.category(c) != 'Mn'))
    return text

def delete_empty_words(word_list):
    empty_words = DB.GET_empty_words_from_DB()
    # Remove empty words
    for i in word_list[:]:
        if len(i) < 2:
            word_list.remove(i)
        else:
            if i in empty_words:
                word_list.remove(i)

    return  word_list


# funcion de extraccion de raices lexicales
def stem(word):
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer('spanish')
    return stemmer.stem(word)

# funcion de extraccion de raices lexicales sobre una lista
def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stem(item))
    return stemmed

def translate(text, lang_out):
    #OPCION 1
    # import goslate
    # gs = goslate.Goslate()
    # tweets[0].text = gs.translate(tweets[0].text, lang)

    #OPCION 2
    # from translate import translator
    # tweets[0].text = translator(lang, 'es', tweets[0].text)

    #OPCION 3
    from textblob import TextBlob
    transltator = TextBlob(text)
    lang_in = transltator.detect_language()
    return str(transltator.translate(to=lang_out))


