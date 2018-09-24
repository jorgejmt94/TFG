import DataFromInternet, DB, Utils, Algorithms, DataFromInternet, Tree, Twitter
print()

debug=0
input_value = tweet_to_analize = user_name = ''
# user_name = 'as_tomasRoncero'
# tweets = Twitter.get_last_weets('as_tomasRoncero', 5)

''' Ejemplo '''
# tweets[0].text = "lionel messi es un autentico crack, tiene un disparo envidiable, es un autentico killer, le amo"

''' Clasificación por temática '''
# tweets[0].text = "El portero la cogió fuera del área, debio ser falta y amarilla"
# tweets[1].text = "Canastó de tres en el ultimo minuto, los aficionados de los Lakers no podian ocultar su furia"
# tweets[2].text = "Ese jab le dejó besando la lona y a Myweather eufórico"
# tweets[3].text = "Ese flanker fué a por uvas"
# tweets[4].text = "Ese tio no vale nada"
# user_name = 'as_tomasRoncero'

''' Clasificación por SA'''
# tweets[0].text = "CR7 es un creído, me da asco"
# tweets[1].text = "Canastó de tres en el ultimo minuto, los aficionados de los Lakers no podian ocultar su furia"
# tweets[2].text = "Ese jab le dejó besando la lona y a Myweather eufórico"
# tweets[3].text = "Ese flanker fué a por uvas"
# tweets[4].text = "Nadal no pudo ocultar su asombro al perder contra Federer"
# user_name = 'as_tomasRoncero'

'''  SPAMMER TROLL  '''
# tweets[0].text = "Nadal no pudo ocultar su asombro al perder contra Federer"
# tweets[1].text = "Nadal asombrado al perder contra Federer"
# tweets[2].text = "Nadal no pudo ocultar su asombro al perder"
# tweets[3].text = "Federer ganó a Nadal"
# tweets[4].text = "Nadal pierde contra Federer"
# user_name = 'BangtanPromoARG'

''' OBSERVE TROLL'''
#user_name = 'Mandeep20016'
# tweets = Twitter.get_last_weets(user_name, 5)

''' HATER TROLL'''
# tweets[0].text = "CR7 es un creído, me da asco"
# tweets[1].text = "Canastó de tres en el ultimo minuto, los aficionados de los Lakers no podian ocultar su furia"
# tweets[2].text = "A Cristiano le gusta llegar a casa y hacer 150 abdominales y a mí prender fuego para un asado"
# tweets[3].text = "enfado"
# tweets[4].text = "hola"
# user_name = ''

'''  FAKE  TROLL'''
# tweets[0].text = "El barça cae eliminado de la champions, los de valverde no pudieron hacer nada"
# tweets[1].text = "Roger federer se proclama numero uno tras derrotar a nadal que acabó rompiendo la raqueta"
# tweets[2].text = "España vuelve gana el mundial de sevens de rugby"
# tweets[3].text = "Tiger Woods vuelve a nombrarse numero 1"
# tweets[4].text = "Mayweather acepta la revancha contra McGregor"
# user_name = 'as_tomasRoncero'

'''
Main menu
'''
print('\n### Welcome to Trolls Detector! ###\n')
while input_value != 'exit' and input_value != '6':
    print('\nOptions:')
    print('\t1 - Is a liar troll?')
    print('\t2 - Is a hater troll?')
    print('\t3 - Is a bot/observer/spammer troll?')
    print('\t4 - Text classification of a tweet (which is sport is)')
    print('\t5 - Sentiment Analysis of a tweet')
    print('\t6 - Exit')
    print('\n\tChoose an option: ')
    input_value = input()
    if input_value == '1':
        print('\n-> Option is fake?')
        print('\tWrite the tweet:')
        tweet_to_analize = input()
        print('\tWrite the user name:')
        user_name = input()
        tweets = Twitter.get_last_weets(user_name, 10)
        Algorithms.is_fake(tweet_to_analize, tweets, user_name, debug=debug)
    if input_value == '2':
        sentiment = []
        print('\n-> Option is an agresive user?')
        print('\tWrite the user name:')
        user_name = input()
        tweets = Twitter.get_last_weets(user_name, 7)
        if debug == 1:
            print('-> Last tweets:')
            for tweet in tweets:
                print(tweet.text)
        Algorithms.is_furious_user(tweets, user_name, debug=0)
    if input_value == '3':
        print('\n-> Option is a bot, spammer or observer troll?')
        print('\tWrite the user name:')
        user_name = input()
        tweets = Twitter.get_last_weets(user_name, 10)
        Algorithms.is_bot_spammer_observer(tweets, user_name, debug=debug)
    if input_value == '4':
        print('\n-> Option Text classification of a tweet.')
        print('\tWrite the tweet:')
        tweet_to_analize = input()
        Algorithms.palabras_repetidas_dictionary(tweet_to_analize,debug=debug)
    if input_value == '5':
        print('\n-> Option Sentiment Analysis of a tweet.')
        print('\tWrite the tweet:')
        tweet_to_analize = input()
        Algorithms.analize_sentiment_with_dictionary(tweet_to_analize, debug=1)

user_name = 'jorgejmt94'
tweets = Twitter.get_last_weets(user_name, 10)
tweet_to_analize = "Futbol: cr7 merecia el 'the best'"
Algorithms.is_fake(tweet_to_analize, tweets, user_name, debug=debug)


