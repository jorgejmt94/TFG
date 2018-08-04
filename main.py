#import DataFromInternet, DB, Utils, Algorithms, DataFromInternet, Tree
import Twitter, Utils, Algorithms, DB
print()




sa = DB.GET_SA_from_DB()
for i in range(0,6):
    print(i,sa[i].sentiment)

frase = "CR7 es el que mas penaltis marca pero no es bueno en el fútbol"
frase_split = frase.split()
