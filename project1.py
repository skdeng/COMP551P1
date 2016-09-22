import csv
import numpy as np
import unicodedata
import string

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def cleanRaceName(s):
    s = s.strip()
    s = s.lower()
    s = strip_accents(s)
    puncList = [".",";",":","!","?","/","\\",",","#","@","$","&",")","(","\"","’","and","de","-"," ","'","‘","`","*"]
    for punc in puncList:
        s = s.replace(punc,'')
    s = s.replace("rocknroll","")
    return s

def cleanRaceTypes(s):
    s = s.lower()
    s = strip_accents(s)
    puncList = [" ","-","'","(",")","/"]
    for punc in puncList:
        s = s.replace(punc,'')
    s = s.replace("km","k")
    s = s.replace("course","")
    s = s.replace("race","")
    s = s.replace("demimarathon","halfmarathon")
    s = s.replace("run","")
    s = s.replace("walk","")
    s = s.replace("classique","")
    s = s.replace("route","")
    return s

def cleanRaceAges(s):
    s = s.upper()
    s = s.replace(" ","")
    s = s.replace("to","-")
    s = s.replace("'","")
    s = s.replace("HOMMES","MALE")
    s = s.replace("MALE","M")
    return s

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

list = []
raceNames = []
raceTypes = []
raceAges = []
transform = {}

def partition(data, k):
    size = len(data) / k
    #size now represents the size of each of our subsets

    #folding determing by array splicing via list comprehension, stepping through length of data by "size" to determine chunks
    subsets = [data[i:i+size] for i in range(0, len(data), size)]

    #handle case of leftover data after folding
    if len(data)%k != 0:
        print "Warning, %d is not a multiple of %d. Skipping %d elements." % (len(data), k, len(data)%k)
        return subsets[:-1]
    else:
        return subsets
    


def marathonYearAvg(data, year):
    return 0

"""
Reads csv source data and returns an array of the format 
"""
#['0', '2015-09-20', "Marathon Oasis Rock 'n' Roll de Montreal", 'Marathon', 14024, 'M50-54']
with open('Project1_data.csv', 'rt') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        for i in range(len(row)):
            if (i-1)%5 == 0:
                if row[i+3] != "-1":
                    row[i+3] = get_sec(row[i+3])
                row[i+1] = cleanRaceName(row[i+1])
                row[i+2] = cleanRaceTypes(row[i+2])
                row[i+4] = cleanRaceAges(row[i+4])
                if row[i+1] in transform:
                    row[i+1] = transform[row[i+1]]
                else:
                    transform[row[i+1]] = row[i+1]
                    for name in raceNames:
                        if levenshtein(row[i+1],name) <= (3):
                            transform[row[i+1]] = name                            
                            row[i+1] = name
                            break
                if row[i+1] not in raceNames:
                    raceNames.append(row[i+1])
                if row[i+2] not in raceTypes:
                    raceTypes.append(row[i+2])
                if row[i+4] not in raceAges:
                    raceAges.append(row[i+4])
        list.append(row)

x = []
y = []

for row in list:
    toAdd = []
    in2015MtlMarathon = 0

    #add participant ID first
    toAdd.append(row[0])

    races = [0] * len(raceNames)
    for j in range(len(row)):
        if(j-1)%5 == 0:
            #print(row[j])
            if (row[j][:4] == '2015') & (row[j+1] == 'marathonoasismontreal'):
                in2015MtlMarathon = 1
            else:
                races[raceNames.index(row[j+1])] += 1
        #if(j-3)%5 == 0:
            #print(row[j])

    toAdd.extend(races)
    x.append(toAdd)
    y.append(in2015MtlMarathon)


x = np.array(x, dtype=float)
y = np.array(y, dtype=float)
