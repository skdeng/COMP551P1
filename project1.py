import csv
import numpy
import unicodedata
import string
from collections import defaultdict

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
    puncList = [".",";",":","!","?","/","\\",",","#","@","$","&",")","(","\"","’","and","de","-"," ","'"]
    for punc in puncList:
        s = s.replace(punc,'')
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
list = []
raceNames = defaultdict(int)
raceTypes = defaultdict(int)
raceAges = defaultdict(int)
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
                raceTypes[row[i+2]] += 1
                raceNames[row[i+1]] += 1
                raceAges[row[i+4]] += 1
        list.append(row)

for i in list:
    print(i)

#print(sorted(raceNames))
#print(len(raceNames))
#print(sorted(raceTypes))
#print(len(raceTypes))
#print(sorted(raceAges))
#print(len(raceAges))

