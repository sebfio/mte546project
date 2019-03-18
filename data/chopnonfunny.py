import sys
import os
from random import random

textFolder      = sys.argv[1]
funnyFolder     = sys.argv[2]
funnyPrefix     = os.listdir(funnyFolder)[0].split("_")[0]

textFolder_new  = textFolder  + "_new"
funnyFolder_new = funnyFolder + "_new"

if not os.path.isdir(textFolder_new):
    os.mkdir(textFolder_new)
if not os.path.isdir(funnyFolder_new):
    os.mkdir(funnyFolder_new)

for f in os.listdir(textFolder):
    fileId = f.split("_")[1]

    funnyF      = funnyPrefix + "_" + str(fileId)

    textFile    = textFolder + "/" + f
    funnyFile   = funnyFolder + "/" + funnyF

    fdT = open(textFile, "rb")
    fdS = open(funnyFile, "r")

    dfTextBin       = fdT.readlines() 
    dfText          = [str(i) for i in dfTextBin]
    dfSentiment     = fdS.readlines() 

    fdTNew = open(textFolder_new    + "/" + f, "wb")
    fdSNew = open(funnyFolder_new   + "/" + funnyF, "w")

    assert(len(dfText) == len(dfSentiment))

    for i, strval in enumerate(dfSentiment):
        val = int(strval)
        
        if val == 0 and random() < float(sys.argv[3]):
            continue

        fdTNew.write(bytes(dfText[i], 'utf-8'))
        fdSNew.write(dfSentiment[i])
    
    fdT.close()
    fdS.close()
    fdTNew.close()
    fdSNew.close()

    fdTNew = open(textFolder_new    + "/" + f, "rb")
    fdSNew = open(funnyFolder_new   + "/" + funnyF, "r")

    dfTextBin       = fdTNew.readlines() 
    dfText          = [str(i) for i in dfTextBin]
    dfSentiment     = fdSNew.readlines() 
    print(len(dfText))
    print(len(dfSentiment))
    assert(len(dfText) == len(dfSentiment))

    fdTNew.close()
    fdSNew.close()
