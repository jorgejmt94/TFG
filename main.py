#import DataFromInternet, DB, Utils, Algorithms, DataFromInternet, Tree
import Twitter, Utils, Algorithms, DB
print()
# input_value = ''
# while input_value != 'exit':
#      print('Write a text: ')
#      input_value = input().lower()
#      input_value = Utils.delete_text_punctuation(input_value)
#      print(input_value)
#      Algorithms.text_classification_with_(input_value)


''''
TODO:
    - Una funcion que coja una frase y pruebe todos los algoritmos
    - Una funcion que coja 'X' frases y diga por cada algoritmo que 'Y' deportes se hablan
    - Probar algoritmos de clasificacion con stemer a ver si mejora
    - Un algoritmo más de clasificación de textos
    - Modificar como dice cada algoritmo de que es
    - Decision final es o no verdad :((((((((((
'''



''' ############################################
    ###   Clasificacion de texto tematica    ###
    ############################################'''

tweets = Twitter.get_last_weets("as_tomasRoncero", 4)
tweets[0].text = "Messi es un autentico crack, tiene un disparo envidiable, es un autentico killer, le amo"
tweets[1].text = "Ese jab le dejó besando la lona y a Myweather eufórico"
tweets[2].text = "Canastó de tres en el ultimo minuto, los aficionados de los Lakers no podian ocultar su furia"
tweets[3].text = "Ese flanker fué a por uvas"


''' HAY QUE QUITAR LOS QUE DEN 0 PUNTOS '''


for tweet in tweets:
    print()
    print("····························· Tweet ·····························")

    print()
    print(tweet.text)
    print()
    print(' ********************** Clasificación por temática ********************** ')
    print()
    print(" ----------------------- NaiveBayes -------------------------- ")
    Algorithms.text_classification_with_naive_bayes(tweet.text)
    print()
    print(" ---------------------- Dictionary --------------------------- ")
    Algorithms.palabras_repetidas_dictionary(tweet.text)
    print()
    print(" -------------------- Dictionary&Tree ----------------------- ")
    Algorithms.palabras_repetidas_dictionary_with_tree(tweet.text)



    #print()
    #print()
    print(' ********************** Sentiment Analysis ********************** ')
    #print()
    #print(" -------------------------    VADER   ----------------------------- ")
    #print(Algorithms.analize_sentiment_with_vaderSentiment(tweet.text))
    #print()
    #print()
    #print(" --------------------    VADER traducido   ----------------------- ")
    #print(Algorithms.(tweet.text))
    #print()
    #print(" --------------------------  TextBlob  ---------------------------- ")
    #print(Algorithms.analize_sentiment_with_textBlob(tweet.text))
    #print(" --------------------  TextBlob traducido  ----------------------- ")
    #print(Algorithms.(tweet.text))
    #print()
    print()
    print(" ------------------------- NaiveBayes ----------------------------- ")
    Algorithms.analize_sentiment_with_naive_bayes(tweet.text)
    print()
    print(" ------------------------- Dictionary ----------------------------- ")
    Algorithms.analize_sentiment_with_dictionary(tweet.text)



''' ############################################
    ###   Clasificacion de texto sentiment   ###
    ############################################'''



