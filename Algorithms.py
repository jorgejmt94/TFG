# my files
import Utils, DB, DataFromInternet, Tree


def palabras_repetidas_wikipedia(text):
    import heapq
    print('--------------PALABRAS_REPETIDAS WIKIPEDIA--------------')
    # tratamiento del texto a clasificar (buscar top_words, etc)
    text2_list = Utils.delete_empty_words(Utils.delete_text_punctuation(text.lower()).split())
    text2_top_words, text2_top_words_percentage = DataFromInternet.get_top_words(text2_list, 1)
    text_to_clasyfy = Utils.WikiData(
        text,  # the text
        text2_list,  # the text in a list
        text2_top_words,  # the top words
        text2_top_words_percentage,
        None  # the type to know
    )

    data = DB.GET_wikipedia_data_from_DB()
    n_repetidas = []
    i=0
    for type in data:
        n_repetidas.append(0)
        for type_word in type.top_words:
            for word  in text_to_clasyfy.top_words:
                if word == type_word:
                    #print(type.type_name, "->" ,word)
                    n_repetidas[i] += ( text_to_clasyfy.top_words_percentages[text_to_clasyfy.top_words.index(word)] + type.top_words_percentages[type.top_words.index(type_word)] ) #TODO:review this
        i += 1
    print("| Resultado palabras_repetidas_wikipedia:")
    max_values = heapq.nlargest(2, n_repetidas) #se escoge las dos mas altas

    i = 0
    for _ in max_values:
        print("|")
        print("v", "%0.2f" %max_values[i], "puntos -> ", Utils.get_data_name(n_repetidas.index(max_values[i])))
        i += 1


    print('---------------------------------------------------------')

def palabras_repetidas_dictionary(text_to_classyfy):
    import string, re, heapq
    n_repetidas_key = []
    n_repetidas_secondary = []
    words_repetidas_key = []
    words_repetidas_secondary = []
    i=0
    #print('--------------PALABRAS_REPETIDAS--------------')

    text_to_classyfy = re.sub('[%s,\d]' % re.escape(string.punctuation), ' ', text_to_classyfy).lower().split()
    # GET THE DICTIONARY
    dictionary = DB.GET_dictionary_from_DB()
    # THE ALGORITHM
    for type in dictionary:
        n_repetidas_key.append(0)
        n_repetidas_secondary.append(0)
        for type_word in type.key_words:

            for word  in text_to_classyfy:
                if Utils.stem(word.lower()) == Utils.stem(type_word.lower()):
                    #print('key:',type.type_name,"->",word)
                    n_repetidas_key[i] += 1
                    words_repetidas_key.append(word)
        for type_word in type.secondary_words:
            for word  in text_to_classyfy:
                if Utils.stem(word.lower()) == Utils.stem(type_word.lower()):
                    #print('secondary',type.type_name,"->",word)
                    n_repetidas_secondary[i] += 1
                    words_repetidas_secondary.append(word)

        i += 1

    i = 0
    #print the result
    #print("--------------------Resultado palabras_repetidas_dictionary:--------------------")
    max_values_key = heapq.nlargest(1, n_repetidas_key) #se escoge las dos mas altas
    max_values_secondary = heapq.nlargest(1, n_repetidas_secondary) #se escoge las dos mas altas
    i = 0
    if max_values_key[i] != 0:
        #words_repetidas_key = set(words_repetidas_key)

        print('Segun primary words:', Utils.get_data_name(n_repetidas_key.index(max_values_key[0])), 'con las palabras repetidas:', words_repetidas_key)
    else:
        print('Ninguna key word encontrada')
    # for _ in max_values_key:
    #     print("|")
    #     print("v", "%0.2f" %max_values_key[i], "puntos -> ", Utils.get_data_name(n_repetidas_key.index(max_values_key[i])))
    #     i += 1
    # i = 0
    if max_values_secondary[i] != 0:
        #words_repetidas_secondary = set(words_repetidas_secondary)

        print('Segun secondary words:', Utils.get_data_name(n_repetidas_secondary.index(max_values_secondary[0])), 'con las palabras repetidas:', words_repetidas_secondary)
    else:
        print('Ninguna secondary word encontrada')
    # for _ in max_values_secondary:
    #     print("|")
    #     print("v", "%0.2f" %max_values_secondary[i], "puntos -> ", Utils.get_data_name(n_repetidas_secondary.index(max_values_secondary[i])))
    #     i += 1
    #
    # print('-------------------------------------------------------------------------------')

def palabras_repetidas_dictionary_with_tree(text_to_classify):
    import heapq

    mongo_dictionary = DB.GET_dictionary_from_DB() #from mongodb

    dictionary = []
    #como estan guardados por orden podemos cogerlos tal cual
    for i in range(Utils.MIN_TYPES, Utils.MAX_TYPES):
        key_words_tree = Tree.AVLTree()
        key_words_tree.insert_array(mongo_dictionary[i].key_words)
        secondary_words_tree = Tree.AVLTree()
        secondary_words_tree.insert_array(mongo_dictionary[i].secondary_words)
        excluding_words_tree = Tree.AVLTree()
        excluding_words_tree.insert_array(mongo_dictionary[i].excluding_words)
        dictionary.append(Utils.Dictionary(
            mongo_dictionary[i].type_name,
            key_words_tree,
            secondary_words_tree,
            excluding_words_tree
        ))


    # print('Write a text: ')
    # text_to_classify = input().lower()
    text_to_classify = Utils.delete_text_punctuation(text_to_classify)
    #the algorithm
    #while text_to_classify != '1' and text_to_classify != 'exit':
    key_words_value = []
    secondary_words_value = []
    excluding_words_value = []
    for sport in dictionary:
        #print('----------------------------------------------------------------------',sport.type_name)

        key_words_value.append(sport.key_words.find_words_in_text(text_to_classify, word_mark= 1))

        secondary_words_value.append(sport.secondary_words.find_words_in_text(text_to_classify, 1))

        excluding_words_value.append(sport.excluding_words.find_words_in_text(text_to_classify, word_mark=1))

    #print(key_words_value)
    #print(secondary_words_value[:])
    #print(excluding_words_value[:])
    i = 0
    for exclude_value in excluding_words_value:
        key_words_value[i] -= exclude_value
        secondary_words_value[i] -= exclude_value*2
        i+=1
    max_values_key = heapq.nlargest(1, key_words_value)  # se escoge las dos mas altas
    max_values_secondary = heapq.nlargest(3, secondary_words_value)  # se escoge las dos mas altas

    i = 0
    # print('MAX VALUES KEY')
    if max_values_key[i] != 0:
        #words_repetidas_key = set(words_repetidas_key)

        print('Segun primary words:', Utils.get_data_name(key_words_value.index(max_values_key[0])))
    else:
        print('Ninguna key word encontrada')
    # for _ in max_values_key:
    #     print("->", max_values_key[i], "puntos -> ",
    #           Utils.get_data_name(key_words_value.index(max_values_key[i])))
    #     i += 1
    # i = 0
    #print('MAX VALUES SECONDARY')
    if max_values_secondary[i] != 0:
        #words_repetidas_secondary = set(words_repetidas_secondary)

        print('Segun secondary words:', Utils.get_data_name(secondary_words_value.index(max_values_secondary[0])))
    else:
        print('Ninguna secondary word encontrada')
    # ok = 1
    # for _ in max_values_secondary:
    #     if ok == 1:
    #         print("|")
    #         print("v", "%0.2f" % max_values_secondary[i], "puntos -> ",
    #               Utils.get_data_name(secondary_words_value.index(max_values_secondary[i])))
    #     if i > 0:
    #         if Utils.get_data_name(secondary_words_value.index(max_values_secondary[i])) == Utils.get_data_name(secondary_words_value.index(max_values_secondary[i-1])):
    #             ok = 0
    #     i += 1

    # print('Write a text: ')
    # input_value = input().lower()
    # input_value = Utils.delete_text_punctuation(input_value)

def text_classification_with_naive_bayes(text):
    from textblob.classifiers import NaiveBayesClassifier
    #key words
    dictionary = DB.GET_dictionary_from_DB()
    train = []

    for type in dictionary:
        for word in type.key_words:
            to_add = (word.lower(), Utils.get_data_id_lower(type.type_name))
            train.append(to_add)
    cl = NaiveBayesClassifier(train)


    result = cl.classify(text.lower())
    print('Según key words:',Utils.get_data_name(result))

    #prob_dist = cl.prob_classify(0)
    # prob_dist.max()
    # print(round(prob_dist.prob(0), 12))

    #secondary words
    dictionary = DB.GET_dictionary_from_DB()
    train = []
    to_add_list=[]
    aux=[]
    import random
    #en las secondary words cogeremos un valor aleatorio de palabras para equilibrar (sino siempre dice que es futbol)
    for type in dictionary:
        for word in type.secondary_words:
            to_add_list.append((word.lower(), type.type_name))
        aux=random.sample(to_add_list,50)#cogemos 50 al azar
        to_add_list = []
        for add in aux:
            train.append(add)

    cl = NaiveBayesClassifier(train)
    result = cl.classify(text.lower())
    print('Según secondary words:',result)

    # from textblob import TextBlob
    # blob = TextBlob("baloncesto", classifier=cl)
    # print(blob.classify())



def text_classification_with_(text):

    #key words
    dictionary = DB.GET_dictionary_from_DB()
    from gensim.models import Word2Vec

    for type in dictionary:
        for word in type.secondary_words:
            to_add = (word, type.type_name)





def analize_sentiment_with_vaderSentiment(text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    if vs['compound'] > 0:
        return 'POSITIVE'
    elif vs['compound'] < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def analize_sentiment_with_textBlob(text):
    from textblob import TextBlob
    #TextBlob.translate(to='es')
    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return 'POSITIVE'
    elif analysis.sentiment.polarity == 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def analize_sentiment_with_dictionary(text):
    import heapq

    n_repetidas = []
    words_repetidas = []

    i=0

    text_to_classyfy = text.split()

    # GET THE DICTIONARY
    dictionary = DB.GET_SA_from_DB()

    #recorro la lista de palabras de cada sentimiento
    for type in dictionary:
        n_repetidas.append(0)
        for type_word in type.words_list:
            #comparo cada palabra del texto con una palabra de la lista de sentimientos
            for text_word  in text_to_classyfy:

                if Utils.stem(text_word.lower()) == Utils.stem(type_word.lower()):
                    n_repetidas[i] += 1
                    words_repetidas.append(text_word)

        i += 1

    words_repetidas = set(words_repetidas)
    i=0
    #print("\n--------------------Resultado sentiment_Analysis_via_dictionary:--------------------")
    #print('Texto a clasificar:', text)
    max_values = heapq.nlargest(1, n_repetidas)  # se escoge las dos mas altas
    if max_values[0] != 0:
        # print('Frase clasificada con el sentimiento ->',
        #   Utils.get_sentiment_name(n_repetidas.index(max_values[i])),
        #   '<- encontrada/s', max_values[i], 'repeticiones de palabras en el diccionario.')
        print('Frase clasificada con el sentimiento ->',
              Utils.get_sentiment_name(n_repetidas.index(max_values[i])),
              '<- encontrada/s', words_repetidas)
    else:
        print('Ninguna palabra encontrada dentro del diccionario')

def analize_sentiment_with_naive_bayes(text):
    from textblob.classifiers import NaiveBayesClassifier
    #key words
    dictionary = DB.GET_SA_from_DB()
    train = []

    for type in dictionary:
        for word in type.words_list:
            to_add = (word.lower(), type.sentiment)
            train.append(to_add)
    cl = NaiveBayesClassifier(train)

    # prob_dist = cl.prob_classify(0)
    # prob_dist.max()
    # print(round(prob_dist.prob(0), 12))

    result = cl.classify(text.lower())
    print('Según key words:',result)

