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
    - Un algoritmo más de clasificación de textos
    - SVM
    - XGBOOST
    - Decision final es o no verdad :((((((((((
'''



tweets = Twitter.get_last_weets("as_tomasRoncero", 5)
tweets[0].text = "El portero la cogió fuera del área, debio ser falta y amarilla"
#tweets[0].text = "lionel messi es un autentico crack, tiene un disparo envidiable, es un autentico killer, le amo"
tweets[1].text = "Ese jab le dejó besando la lona y a Myweather eufórico"
tweets[2].text = "Canastó de tres en el ultimo minuto, los aficionados de los Lakers no podian ocultar su furia"
tweets[3].text = "Ese flanker fué a por uvas"
tweets[4].text = "Ese tio no vale nada"



''' HAY QUE QUITAR LOS QUE DEN 0 PUNTOS '''

print()

for tweet in tweets:
    print()
    print()
    print("····························· Tweet ·····························")

    print(tweet.text)
    print()
    #print(' ********************** Clasificación por temática ********************** ')

    #Algorithms.palabras_repetidas_dictionary(tweet.text)
    #Algorithms.palabras_repetidas_dictionary_with_tree(tweet.text)


    print()
    print()
    print(' ********************** Sentiment Analysis ********************** ')
    print(" -------------------------    VADER   ----------------------------- ")
    print(Algorithms.analize_sentiment_with_vaderSentiment(tweet.text))
    print()
    print()
    #print(" --------------------    VADER traducido   ----------------------- ")
    #print(Algorithms.(tweet.text))
    #print()
    print(" --------------------------  TextBlob  ---------------------------- ")
    print(Algorithms.analize_sentiment_with_textBlob(tweet.text))
    #print()
    #print()
    #print(" --------------------  TextBlob traducido  ----------------------- ")
    #print(Algorithms.(tweet.text))
    print()
    print(" ------------------------- Dictionary ----------------------------- ")
    Algorithms.analize_sentiment_with_dictionary(tweet.text)





