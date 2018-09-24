__author__      = "Jorge Melguizo"
__copyright__   = "Copyright 2018, Trolls Detector"

import pymongo, json
from pymongo import MongoClient
import Utils, re, string

########################################################################################################################
#                  CONNECT                                                                                             #
########################################################################################################################
'''#####################################
# CONNECT TO DB                        #
# PARAM:                               #
#       name db                        #
#       name collection                #
# RETURN:                              #
#       db                             #
#       collection                     #
########################################'''
#84.88.154.252 easy

def connectDB(name_db, name_collection):
    client = MongoClient('localhost', 27017)
    return client[name_db], client[name_db][name_collection]


''' ###########################################
    # CONNECT TO BASE DE DATOS EL-NACIONAL    #
    ###########################################'''
#def connect_to_nacional_DB():
#    client = MongoClient("mongodb://easy:SMDagile2017@84.88.154.252:27017/el-nacional") #84.88.154.252 easy
#    return client['el-nacional'], client['el-nacional']['News']

########################################################################################################################
#                  GETS                                                                                                #
########################################################################################################################
''' #################################################
    # Get all data of wikipedia from MONGO DB       #
    #################################################'''
def GET_empty_words_from_DB():

    db, cl = connectDB('train', 'specialWords')


    cursor = cl.find({'type': 'emptyText'})
    data = []
    for document in cursor:
        data = document["emptyWords"]
    return data

def GET_empty_words_SA_from_DB():

    db, cl = connectDB('train', 'specialWords')
    cursor = cl.find({'type': 'emptySA'})
    data = []
    for document in cursor:
        data = document["emptyWordsSA"]
    return data

def GET_plus_words_from_DB():

    db, cl = connectDB('train', 'specialWords')
    cursor = cl.find({'type': 'plus'})
    data = []
    for document in cursor:
        data = document["words"]
    return data

def GET_less_words_from_DB():

    db, cl = connectDB('train', 'specialWords')
    cursor = cl.find({'type': 'less'})
    data = []
    for document in cursor:
        data = document["words"]
    return data

''' #################################################
    # Get all data of wikipedia from MONGO DB       #
    #################################################'''
def GET_wikipedia_data_from_DB():

    db, cl = connectDB('train', 'wikipedia')

    # print(cl.find_one({"type": "Futbol"})["top_words"])
    cursor = cl.find({})
    data = []
    for document in cursor:
        #print(document["top_words"])
        data.append(Utils.WikiData(
            document['text'],
            document['word_list'],
            document['top_words'],
            document['top_words_percentages'],
            document['type']
        ))
    return data


''' #################################
    # GET DICTIONARY FROM MONGODB   #
    #################################'''
def GET_dictionary_from_DB():

    db, cl = connectDB('train', 'dictionary')

    # print(cl.find_one({"type": "Futbol"})["top_words"])
    cursor = cl.find({})
    data = []
    for document in cursor:
        #print(document["keyWords"])
        # data.append(Utils.Dictionary(
        #     document['type'],
        #     document['keyWords'],
        #     document['secondaryWords'],
        #     document['excludingWords']
        # ))
        data.append(Utils.Dictionary(
            document['type'],
            document['keyWords'],
            document['secondaryWords'],
            document['excludingWords']
        ))
    return data


''' #################################
    # GET DICTIONARY FROM MONGODB   #
    #################################'''
def GET_SA_from_DB():

    db, cl = connectDB('train', 'sa')

    # print(cl.find_one({"type": "Futbol"})["top_words"])
    cursor = cl.find({})
    data = []
    for document in cursor:
        data.append(Utils.Sentiment(
            document['type'],
            document['word_list']
        ))
    return data


''' #################################
    # GET DICTIONARY FROM MONGODB   #
    #################################'''
def GET_text_to_classify(type):

    db, cl = connectDB('train', 'news')
    # print(cl.find_one({"type": "Futbol"})["top_words"])
    cursor = cl.find({})
    text = ""
    for document in cursor:
        #print(document["keyWords"])
        if(document['type'].lower() == type.lower()):
            text = document['content']
    return text

''' ############################################################
    # GET DATA FROM EL NACIONAL DB AN INSERT TO MY NACIONAL DB #
    ############################################################'''
def GET_all_sports__from_nacional_DB():

    db, cl = connectDB('nacional', 'noticia')
    cur = cl.find()

    categories = ['Tenis', 'Fútbol', 'Baloncesto', 'Beisbol', 'Motores', 'Deportes', 'Hipismo', 'Otros']

    for document in cur:
        try:
           newsCategory = re.sub('[%s,\d]' % re.escape(string.punctuation), '', document['newsCategory'])
        except KeyError:
           print('Not Found :(')
        else:
           for type in categories:
               if type == newsCategory:
                    INSERT_to_nacional_DB(document)


########################################################################################################################
#                  INSERTS                                                                                             #
########################################################################################################################
''' ##################################################
    # CONNECT TO DB                                  #
    # PARAM:                                         #
    #       name db                                  #
    #       name collection                          #
    #       data for create in a json format         #
    ##################################################'''
def INSERT_json_toDB(db_name, cl_name, data_json):

    db, cl = connectDB(db_name, cl_name)
    result = cl.insert_one(data_json)


''' ##########################################################################################
    # CONNECT TO DB                                                                          #
    # PARAM:                                                                                 #
    #       name db                                                                          #
    #       name collection                                                                  #
    #       wikipedia data for create a json                                                 #
    ##########################################################################################'''

def INSERT_toDB(db_name, cl_name,text, word_list, top_words, top_words_percentages, type):

    db, cl = connectDB(db_name, cl_name)
    data_json = {
        'text': text,
        'word_list': word_list,
        'top_words': top_words,
        'top_words_percentages': top_words_percentages,
        'type': type
    }
    result = cl.insert_one(data_json)


''' ########################################
    # INSER TO MY NACIONAL DB              #
    # PARAM: data to insert in json format #
    ########################################'''
def INSERT_to_nacional_DB(data):

    db, cl = connectDB('nacional', 'noticia')
    cl.insert_one(data)


