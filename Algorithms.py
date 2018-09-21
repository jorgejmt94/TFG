# my files
import Utils, DB, DataFromInternet, Tree, Twitter


''''
Clasificacion por tematica
'''''
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
    #import time
    #start = time.time()

    import string, re, heapq
    n_repetidas_key,     n_repetidas_secondary,     n_words_exluding  = [], [], []
    words_repetidas_key, words_repetidas_secondary, words_exluding    = [], [], []

    i=0
    #print('--------------PALABRAS_REPETIDAS--------------')

    text_to_classyfy_list = re.sub('[%s,\d]' % re.escape(string.punctuation), ' ', text_to_classyfy).lower().split()
    text_to_classyfy_list = Utils.delete_empty_words(text_to_classyfy_list)
    print('Tweet a tratar:',text_to_classyfy_list)
    print()
    print(" ---------------------- Dictionary --------------------------- ")
    # GET THE DICTIONARY
    dictionary = DB.GET_dictionary_from_DB()
    # THE ALGORITHM
    for type in dictionary:
        n_repetidas_key.append(0)
        n_repetidas_secondary.append(0)
        n_words_exluding.append(0)
        aux = []
        for type_word in type.key_words:
            for word  in text_to_classyfy_list:
                if Utils.stem(word.lower()) == Utils.stem(type_word.lower()):
                    n_repetidas_key[i] += 1
                    aux.append(word)
        words_repetidas_key.append(aux)
        aux = []
        for type_word in type.secondary_words:
            for word  in text_to_classyfy_list:
                if Utils.stem(word.lower()) == Utils.stem(type_word.lower()):
                    n_repetidas_secondary[i] += 1
                    aux.append(word)
        words_repetidas_secondary.append(aux)
        aux = []
        for type_word in type.excluding_words:
            for word  in text_to_classyfy_list:
                if Utils.stem(word.lower()) == Utils.stem(type_word.lower()):
                    n_words_exluding[i] += 1
                    aux.append(word)
        words_exluding.append(aux)

        i += 1

    i = 0
    #print the result
    #print("--------------------Resultado palabras_repetidas_dictionary:--------------------")
    max_values_key = heapq.nlargest(1, n_repetidas_key) #se escoge las dos mas altas
    max_values_secondary = heapq.nlargest(1, n_repetidas_secondary) #se escoge las dos mas altas

    i = 0
    if max_values_key[i] != 0:
        #words_repetidas_key = set(words_repetidas_key)

        print('Segun palabras primarias:    ', Utils.get_data_name(n_repetidas_key.index(max_values_key[0])), 'con las palabras repetidas:', words_repetidas_key[n_repetidas_key.index(max_values_key[0])])
    else:
        print('Ninguna key word encontrada')
    # for _ in max_values_key:
    #     print("|")
    #     print("v", "%0.2f" %max_values_key[i], "puntos -> ", Utils.get_data_name(n_repetidas_key.index(max_values_key[i])))
    #     i += 1
    # i = 0
    if max_values_secondary[i] != 0:
        #words_repetidas_secondary = set(words_repetidas_secondary)

        print('Segun palabras secundarias:  ', Utils.get_data_name(n_repetidas_secondary.index(max_values_secondary[0])), 'con las palabras repetidas:', words_repetidas_secondary[n_repetidas_secondary.index(max_values_secondary[0])])
    else:
        print('Ninguna secondary word encontrada')



    # for _ in max_values_secondary:
    #     print("|")
    #     print("v", "%0.2f" %max_values_secondary[i], "puntos -> ", Utils.get_data_name(n_repetidas_secondary.index(max_values_secondary[i])))
    #     i += 1
    #
    i = 0
    final_value=[]
    for i in range(0,13):
        final_value.append(0)
        final_value[i] = n_repetidas_key[i]*1 + n_repetidas_secondary[i]*0.25 - n_words_exluding[i]*1.5
    max_value_key = heapq.nlargest(1, final_value) #se escoge las dos mas altas
    if max_value_key[0] == 0:
        print('El texto no se pudo clasificar.')
    else:
        print('Aplicando la formula final:  ', Utils.get_data_name(final_value.index(max_value_key[0])),'con',max_value_key[0], 'puntos')

    # end = time.time()
    # print('Ha tardado:',end - start,'seg')

    # print('-------------------------------------------------------------------------------')

def palabras_repetidas_dictionary_with_tree(text_to_classify):
    import heapq
    #import time
    print()
    print(" -------------------- Dictionary&Tree ----------------------- ")
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
    found_1words,found_2words,found_exwords = [],[],[]
    #start = time.time()

    for sport in dictionary:
        #print('----------------------------------------------------------------------',sport.type_name)
        value, words = sport.key_words.find_words_in_text(text_to_classify, word_mark = 1)
        key_words_value.append(value)
        found_1words.append(words)

        value, words = sport.secondary_words.find_words_in_text(text_to_classify, word_mark=0.25)
        secondary_words_value.append(value)
        found_2words.append(words)

        value, words = sport.excluding_words.find_words_in_text(text_to_classify, word_mark=1.5)
        excluding_words_value.append(value*-1)
        found_exwords.append(words)

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

        print('Segun primary words:', Utils.get_data_name(key_words_value.index(max_values_key[0])), found_1words[key_words_value.index(max_values_key[0])])
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

        print('Segun secondary words:', Utils.get_data_name(secondary_words_value.index(max_values_secondary[0])), found_2words[secondary_words_value.index(max_values_secondary[0])])
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
    # end = time.time()
    # print('Ha tardo:',end - start,'seg')

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

def text_classification_without_dictionary():
    from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import pandas, xgboost
    from keras import layers, models, optimizers

    data = open('./data/text_examples.txt', encoding="utf-8").read()

    labels, texts = [], []
    # clabel, ctext = [], []
    #
    # for i, line in enumerate(data.split("\n")):
    #
    #      content = line.split("$")
    #      labels.append(content[0])
    #      #texts.append(line)
    #      texts.append(content[1])
    #
    #  # create a dataframe using texts and lables
    # ctrainDF = pandas.DataFrame()
    # ctrainDF['text'] = texts
    # ctrainDF['label'] = labels #[0,1,2,3,4,5,6,7,8,9,10,11,12,7,6,0,0,2]
    i = 0
    year = 2018
    n_tweets_aux = n_tweets = 1000/13 #coge tweets de 100 en 100
    show=True
    print('Descargando tweets...')
    for sport_n in range(0, 13):
        while n_tweets_aux>0:
            data = Twitter.get_twees_by_hashtag(Utils.get_data_name(sport_n), n_tweets=n_tweets, year=year,show=show)
            for tweet in data:
                labels.append(sport_n)
                texts.append(Twitter.clean_tweet_2(tweet).text)
            year=year-1
            n_tweets_aux = n_tweets_aux-100#/13
            show=False
        n_tweets_aux = n_tweets
        year=2018
        show=True

    print('Descarga de tweets finalizada!', len(texts))
    print(len(labels))
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.8, random_state=42)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])

    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect.transform(train_x)
    xvalid_count = count_vect.transform(valid_x)

    # # word level tf-idf
    # tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    # tfidf_vect.fit(trainDF['text'])
    # xtrain_tfidf = tfidf_vect.transform(train_x)
    # xvalid_tfidf = tfidf_vect.transform(valid_x)

    # # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

    # # characters level tf-idf
    # tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
    #                                          max_features=5000)
    # tfidf_vect_ngram_chars.fit(trainDF['text'])
    # xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
    # xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

    def train_model(classifier, feature_vector_train, label, feature_vector_valid):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        #print ("Confusion Matrix:\n",metrics.confusion_matrix(predictions, valid_y))

        return metrics.accuracy_score(predictions, valid_y)

    print()
    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
    print("Naive Bayes, Count Vectors:                              ", accuracy)
    # Naive Bayes on Word Level TF IDF Vectors
    # accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
    # print("NB, WordLevel TF-IDF: ", accuracy)
    # # Naive Bayes on Ngram Level TF IDF Vectors
    # accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    # print("NB, N-Gram Vectors: ", accuracy)
    # # Naive Bayes on Character Level TF IDF Vectors
    # accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    # print("NB, CharLevel Vectors: ", accuracy)

    print()
    # Linear Classifier on Count Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
    print("Linear Classifier (Logistic Regression), Count Vectors: ", accuracy)
    # # Linear Classifier on Word Level TF IDF Vectors
    # accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
    # print("LR, WordLevel TF-IDF: ", accuracy)
    #
    # # Linear Classifier on Ngram Level TF IDF Vectors
    # accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    # print("LR, N-Gram Vectors: ", accuracy)

    # # Linear Classifier on Character Level TF IDF Vectors
    # accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y,
    #                        xvalid_tfidf_ngram_chars)
    # print("LR, CharLevel Vectors: ", accuracy)

    print()
    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print("Support Vector Machine (SVM), Count Vectors:              ", accuracy)


    print()
    # Bagging Model RF on Count Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
    print("Random Forest Model, Count Vectors:                         ", accuracy)
    # # RF on Word Level TF IDF Vectors
    # accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
    # print("RF, WordLevel TF-IDF: ", accuracy)

    print()
    # Extereme Gradient Boosting on Count Vectors
    accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
    print("XgBoost, Count Vectors:                                      ", accuracy)

    # # Extereme Gradient Boosting on Word Level TF IDF Vectors
    # accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
    # print("Xgb, WordLevel TF-IDF: ", accuracy)
    #
    # # Extereme Gradient Boosting on Character Level TF IDF Vectors
    # accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y,
    #                        xvalid_tfidf_ngram_chars.tocsc())
    # print("Xgb, CharLevel Vectors: ", accuracy)

    # def create_model_architecture(input_size):
    #     # create input layer
    #     input_layer = layers.Input((input_size,), sparse=True)
    #
    #     # create hidden layer
    #     hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    #
    #     # create output layer
    #     output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)
    #
    #     classifier = models.Model(inputs=input_layer, outputs=output_layer)
    #     classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    #     return classifier
    #
    # classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
    # accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
    # print("Neuronal Networks, Ngram Level TF IDF Vectors", accuracy)


''''
Clasificacion por sentimiento
'''''
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

