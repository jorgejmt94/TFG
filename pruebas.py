#my files
import DB, Utils, Algorithms, DataFromInternet, Twitter

#text_to_classify = input('Enter the text to classify: ') #input mode

''' ################################################
    ###  PALABRAS REPETIDAS V1, WIKIPEDIA          #
    ################################################'''
# # text to classify
# text_to_classify = DB.GET_text_to_classify(Utils.get_data_name(12))
# # execute the algorithm
# Algorithms.palabras_repetidas_wikipedia(text_to_classify)
# # END PALABRAS REPETIDAS
###


''' ##################################################
    ###  LEER FICHEROS DE PALABRAS E INSERTARLOS   ###
    ##################################################'''
# for i in range(Utils.MIN_TYPES, Utils.MAX_TYPES):
#
#     file_name = Utils.get_data_name(0)
#     fichero = open ('./data/keyWords/'+ Utils.get_data_name(i) +'.txt',"r", encoding="utf-8")
#     key = sorted(fichero.read().lower().split('\n'))
#     #print (keyWords)
#     fichero = open ('./data/secondaryWords/'+ Utils.get_data_name(i) +'.txt',"r", encoding="utf-8")
#     secondary = sorted(fichero.read().lower().split('\n'))
#     #print (secondaryWords)
#     fichero = open('./data/excludingWords/' + Utils.get_data_name(i) + '.txt', "r", encoding="utf-8")
#     excluding = sorted(fichero.read().lower().split('\n'))
#
#     data_json = {
#         'type': Utils.get_data_name(i),
#         'keyWords': key,
#         'secondaryWords': secondary,
#         'excludingWords': excluding
#     }
#     print(data_json['type'])
#     print(data_json['keyWords'])
#     print(data_json['secondaryWords'])
#     print(data_json['excludingWords'],"\n---------------------------------------------")
#
#     DB.INSERT_json_toDB('train', 'dictionary', data_json)
###


''' ##################################
    ###   Insertar notica o text   ###
    ##################################'''
# text  = "La onda expansiva del ‘boom’ del boxeo vizcaíno sigue traspasando fronteras. Después de que Jon Fernández se estrenara en septiembre de 2016 en los rings americanos -ya ha combatido tres veces al otro lado del Atlántico- y que, posteriormente, Kerman Lejarraga siguiera su estela, otro paisano suyo vivirá la experiencia de pelear en la meca del deporte de las 16 cuerdas. Ibon Larrinaga vivirá su bautismo ‘USA’ y no lo hará en cualquier sitio. Será en el Madison Square Garden, el mítico pabellón neoyorquino que ha albergado en su larga historia tantos y tantos combates legendarios. Para la cita del 12 de mayo, el de Romo tendrá como contrincante al norirlandés Michael Conlan, natural de Belfast. Larrinaga afronta este importante paso en su carrera después de subir de peso. Abandonó el supergallo tras caer noqueado por Aritz ‘Chulito’ Pardal en el combate por el Mediterráneo WBC celebrado en el frontón Bizkaia el pasado mes de diciembre. Ya en el pluma, el getxotarra se impuso a los puntos el 23 de marzo al georgiano Levan Tsiklauri también en Miribilla.A sus 27 años, con un récord de 10 victorias y la derrota ante Pardal como único borrón, Larrinaga tendrá un buen examen para evaluar su adaptación al nuevo peso ante un rival peligroso. Conlan (26 años) no tiene una excesiva experiencia como profesional, puede debutó hace casi exactamente un año. Como amateur fue bronce en los Juegos Olímpicos de Londres 2012 y diploma en los de Río 2016.Ya en el boxeo de pago presenta un expediente inmaculado de seis victorias. Curiosamente, tres de esos seis combates los ha disputado en el teatro del Madison, un recinto anexo con capacidad para 5.000 espectadores, frente a los 21.000 de aforo del pabellón. De este modo, Larrinaga será uno de los pocos púgiles vascos en la historia en pelear en el Madison Square Garden. Antes lo hicieron los guipuzcoanos Isidoro Gaztañaga y el legendario Paulino Uzcudun. Pero lo cierto es que en aquella primera mitad del siglo XX, el Madison tenía otra ubicación. En ella, el errezildarra Uzcudun cerró su trayectoria profesional en 1935 perdiendo por KO ante Joe Louis"
#
# data_json = {
#     'type': 'Boxeo',
#     'content': text,
#     'url': None,
#     'from': 'AS'
# }
#
# DB.INSERT_json_toDB('train', 'news', data_json)
###


''' ####################################
    ###  clasificar por diccionario  ###
    ####################################'''
# while True:
#     print('Write a text: ')
#     input_value = input().lower()
#     input_value = Utils.delete_text_punctuation(input_value)
#     Algorithms.palabras_repetidas_dictionary_with_tree(input_value)
###


''' ####################################
    ###         Stemmer              ###
    ####################################'''
# import Stemmer
# print('Write a word: ')
# input_value = input().lower()
# stemmer = Stemmer.Stemmer(u'spanish')
# word = stemmer.stemWord(input_value.lower())
# print(input_value, word)
###


''' ####################################
    ###             S.A              ###
    ####################################'''
# tweets = Twitter.get_last_weets("as_tomasRoncero", 1)
#
# tweets[0].text = "Lionel Messi es el mejor jugador y el que mas me gusta. Creo que es muy buen jugador"
# #tweets[0].text = "Lionel Messi is the best player"
# #tweets[0].text = "Cristiano Ronaldo es una mierda de jugador, no me gusta mucho y marca goles porque tiene mucha suerte"
#
# tweets[0].text = Utils.translate(tweets[0].text, 'en')
#
#
# print(tweets[0].text)
# print('Vader:')
# print(Algorithms.analize_sentiment_with_vaderSentiment(tweets[0].text))
# print('TextLob:')
# print(Algorithms.analize_sentiment_with_textBlob(tweets[0].text))
#

#
# #TRADUCIR ARCHIVOS:
# # translated_lines = gs.translate(open('readme.txt'))
# # translation = '\n'.join(translated_lines)
# # print(translation)


''' ####################################
    ###         Naive Bayes          ###
    ####################################'''
# tweets = Twitter.get_last_weets("as_tomasRoncero", 10)
#
# for tweet in tweets:
#     print("-------------------Tweet---------------------")
#     print(tweet.text)
#     print("--------------------NB-----------------------")
#     Algorithms.text_classification_with_naive_bayes(tweet.text)
#     print()
#     print()


''' ##########################################################
    ###         Insertar palabras SA to mongoDB            ###
    ##########################################################'''
# fichero = open ('./data/SA/alegria.txt',"r", encoding="utf-8")
# words = sorted(fichero.read().lower().split('\n'))
# sentiments = []
# sentiment = Utils.Sentiment('alegria', words)
# sentiments.append(sentiment)
# fichero = open ('./data/SA/amor.txt',"r", encoding="utf-8")
# words = sorted(fichero.read().lower().split('\n'))
# sentiment = Utils.Sentiment('amor', words)
# sentiments.append(sentiment)
# fichero = open ('./data/SA/enfado.txt',"r", encoding="utf-8")
# words = sorted(fichero.read().lower().split('\n'))
# sentiment = Utils.Sentiment('enfado', words)
# sentiments.append(sentiment)
# fichero = open ('./data/SA/miedo.txt',"r", encoding="utf-8")
# words = sorted(fichero.read().lower().split('\n'))
# sentiment = Utils.Sentiment('miedo', words)
# sentiments.append(sentiment)
# fichero = open ('./data/SA/sorpresa.txt',"r", encoding="utf-8")
# words = sorted(fichero.read().lower().split('\n'))
# sentiment = Utils.Sentiment('sorpresa', words)
# sentiments.append(sentiment)
# fichero = open ('./data/SA/tristeza.txt',"r", encoding="utf-8")
# words = sorted(fichero.read().lower().split('\n'))
# sentiment = Utils.Sentiment('tristeza', words)
# sentiments.append(sentiment)
#
# for sentiment in sentiments:
#     data_json = {
#         'type': sentiment.sentiment,
#         'word_list': sentiment.words_list
#     }
#     DB.INSERT_json_toDB('train', 'sa', data_json)