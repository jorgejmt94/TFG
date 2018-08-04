#myFiles
import Utils, DB


def get_text_from_wikipedia(type):
    import wikipedia
    # Obtener info de wikipedia
    wikipedia.set_lang("es")
    return wikipedia.page(Utils.get_data_name(type)).content.lower()


def get_top_words(word_list, percentage):
    import collections
    # WordCounter
    word_counter = collections.Counter(word_list)
    n_top_words = round(len(word_counter) * percentage)
    top_words=[]
    top_words_percentages = []
    for word, count in word_counter.most_common(n_top_words):
        top_words_percentages.append((count / len(word_list)) * 100)
        top_words.append(word)

    return top_words, top_words_percentages


def get_data_from_wikipedia(type, percentage):
    data = []
    if type in range(Utils.MIN_TYPES, Utils.MAX_TYPES):
        text                            = get_text_from_wikipedia(type).lower()
        text_without_punctuation        = Utils.delete_text_punctuation(text)
        word_list                       = text_without_punctuation.split()
        word_list_without_empty         = Utils.delete_empty_words(word_list)
        top_words, top_words_percentages= get_top_words(word_list_without_empty, percentage)
        new                           = Utils.WikiData(text,
                                                 word_list,
                                                 top_words,
                                                 top_words_percentages,
                                                 Utils.get_data_name(type))
        data.append(new)

        return data

    elif type.lower() == "all":
        print('<------GET_TEXTS_FROM_WIKIPEDIA------>')
        for type in range(Utils.MIN_TYPES, Utils.MAX_TYPES):
            #print(Utils.get_dataName(type))
            text = get_text_from_wikipedia(type)
            text = Utils.delete_text_punctuation(text)
            word_list = text.split()
            word_list = Utils.delete_empty_words(word_list)
            top_words, top_words_percentages = get_top_words(word_list, percentage)
            new = Utils.WikiData(text,
                               word_list,
                               top_words,
                               top_words_percentages,
                               Utils.get_data_name(type))
            #insert to mongodb
            print("Inserting the data to DB")
            DB.INSERT_toDB("train",
                           "wikipedia",
                           text,
                           word_list,
                           top_words,
                           top_words_percentages,
                           Utils.get_data_name(type)
                           )

            data.append(new)

        print('--------------------------------------')

        return data

