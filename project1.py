import csv
import numpy as np
import unicodedata
import string
import log_reg
import lin_reg
import naive_bayes
import random


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


def linearRegression(X, Y):
    m = lin_reg.Model(2)
    m.solve(X,Y)
    return m.w

def logisticRegression(X, Y, testX):
    testX = np.array(testX, dtype=float)
    model = log_reg.Model(6, 0.1)
    for i in range(10):
        model.step(X,Y)
        #print('Iteration '+str(i)+' with error: '+str(model.error(X,Y)))
    Z = model.forward(testX)
    return Z

def naiveBayes(X,Y,testX):
    m = naive_bayes.Model(2)
    m.fit(X,Y,True)
    testX = np.array(testX, dtype=float)
    np.set_printoptions(threshold=np.nan)
    #print(m.forward(testX))
    return m.forward(testX)

def testLinearClassifier(w, testX, testY):
    output = []
    for row in testX:
        row = [1] + row
        xrow = np.array(row, dtype=float)
        wx = np.dot(xrow,w)
        correlation = 1 if (wx > 0.4442) else 0
        output.append(correlation)

    numCorrect = 0
    for i in range(len(output)):
        if(output[i] == testY[i]):
            numCorrect +=1
    #print("Linear: "+str(numCorrect)+"/"+str(len(output)))
    print("Linear: "+str(numCorrect/len(output)))

def testLogistic(output, testY):
    numCorrect = 0
    print("length output: "+str(len(output)))
    for i in range(len(output)):
        if(output[i][0] == testY[i]):
            numCorrect +=1
    print("Logistic: "+str(numCorrect/len(output)))

def testNaiveBayes(output, testY):
    numCorrect = 0
    print("length output: "+str(len(output)))
    for i in range(len(output)):
        if(output[i] == 1):
            output[i] = 0
        if(output[i] == 2):
            output[i] = 1
        #print(str(output[i])+" "+str(testY[i]))
        if(output[i] == testY[i]):
            numCorrect +=1
    print("Naive Bayes: "+str(numCorrect/len(output)))

def testLinearRegression(w, testX, testY):
    #output = []
    sumOfPercentDifference = 0
    numPercentDifference = 0
    for i, row in enumerate(testX):
        row = [1] + row
        xrow = np.array(row, dtype=float)
        wx = np.dot(xrow,w)
        #print(str(wx)+" "+str(testY[i]))
        #print(str(wx)+" "+str(testY[i]))
        percentDifference = abs(wx - testY[i])/testY[i]*100
        sumOfPercentDifference += percentDifference
        numPercentDifference += 1
        #output.append(percentDifference)
    print("Regression Percent Difference: "+str(sumOfPercentDifference/numPercentDifference))
    
dataFile = 'Project1_data.csv'
list = []
raceNames = []
raceTypes = []
raceAges = []
transform = {}
#['0', '2015-09-20', "Marathon Oasis Rock 'n' Roll de Montreal", 'Marathon', 14024, 'M50-54']
#Sanitize the data 

with open(dataFile, 'rt') as csvfile:
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

#print(raceNames)
#print(raceTypes)
#print(raceAges)

x=[]
y=[]
#Encode the data for participation
for row in list:
        toAdd = []
        in2015MtlMarathon = 0
        racesPerYear = {}
        racesPerYear['2012'] = 0
        racesPerYear['2013'] = 0
        racesPerYear['2014'] = 0
        racesPerYear['2015'] = 0
        racesPerYear['2016'] = 0
        numMontrealMarathons = 0
        for j in range(len(row)):
            if(j-1)%5 == 0:
                racesPerYear[row[j][:4]] += 1
                if(row[j+1] == 'marathonoasismontreal'):
                    numMontrealMarathons += 1
        #toAdd.append(1)
        toAdd.append(racesPerYear['2012'])
        toAdd.append(racesPerYear['2013'])
        toAdd.append(racesPerYear['2014'])
        toAdd.append(racesPerYear['2015'])
        toAdd.append(racesPerYear['2016'])
        toAdd.append(numMontrealMarathons)
        for j in range(len(row)):
            if(j-1)%5 == 0:
                if (row[j][:4] == '2015') & (row[j+1] == 'marathonoasismontreal'):
                    in2015MtlMarathon = 1
        x.append(toAdd)
        y.append(in2015MtlMarathon)

trainingX = []
testX = []
trainingY = []
testY = []

#Randomly take 4/5 data and put into trainingX, trainingY
#Take the remaining 1/5 data and put into testX, testY

for i in range(len(x)):
    if(random.randint(0,4) != 0):
        trainingX.append(x[i])
        trainingY.append(y[i])
    else:
        testX.append(x[i])
        testY.append(y[i])        

trainingX = np.array(trainingX, dtype=float)
trainingY = np.array(trainingY, dtype=float)

w = linearRegression(trainingX, trainingY)
outputLogistic = logisticRegression(trainingX, trainingY, testX)
testLinearClassifier(w, testX, testY)
testLogistic(outputLogistic, testY)

x=[]
y=[]
for row in list:
        toAdd = []
        in2015MtlMarathon = 0
        racesPerYear = {}
        racesPerYear['2012'] = 0
        racesPerYear['2013'] = 0
        racesPerYear['2014'] = 0
        racesPerYear['2015'] = 0
        racesPerYear['2016'] = 0
        mtlMarathonPerYear = {}
        mtlMarathonPerYear['2012'] = 0
        mtlMarathonPerYear['2013'] = 0
        mtlMarathonPerYear['2014'] = 0
        #mtlMarathonPerYear['2015'] = 0
        #mtlMarathonPerYear['2016'] = 0
        for j in range(len(row)):
            if(j-1)%5 == 0:
                racesPerYear[row[j][:4]] = 1
                if(row[j+1] == 'marathonoasismontreal'):
                    mtlMarathonPerYear[row[j][:4]] = 1
        toAdd.append(racesPerYear['2012'])
        toAdd.append(racesPerYear['2013'])
        toAdd.append(racesPerYear['2014'])
        toAdd.append(racesPerYear['2015'])
        toAdd.append(racesPerYear['2016'])
        toAdd.append(mtlMarathonPerYear['2012'])
        toAdd.append(mtlMarathonPerYear['2013'])
        toAdd.append(mtlMarathonPerYear['2014'])
        #toAdd.append(mtlMarathonPerYear['2015'])
        #toAdd.append(mtlMarathonPerYear['2016'])
        for j in range(len(row)):
            if(j-1)%5 == 0:
                if(row[j][:4] == '2015') & (row[j+1] == 'marathonoasismontreal'):
                    in2015MtlMarathon = 1
        x.append(toAdd)
        y.append(in2015MtlMarathon)

trainingX = []
testX = []
trainingY = []
testY = []

for i in range(len(x)):
    if(random.randint(0,4) != 0):
        trainingX.append(x[i])
        trainingY.append(y[i])
    else:
        testX.append(x[i])
        testY.append(y[i])        

trainingX = np.array(trainingX, dtype=float)
trainingY = np.array(trainingY, dtype=float)

outputNaive = naiveBayes(trainingX, trainingY, testX)
testNaiveBayes(outputNaive, testY)

x=[]
y=[]

#Encode the data for participation
for row in list:
    toAdd = []
    in2015MtlMarathon = 0
    racesPerYear = {}
    racesPerYear['2012'] = 0
    racesPerYear['2013'] = 0
    racesPerYear['2014'] = 0
    racesPerYear['2015'] = 0
    racesPerYear['2016'] = 0
    numMontrealMarathons = 0
    numTotalMarathons = 0
    totalTime = 0
    for j in range(len(row)):
        if(j-1)%5 == 0:
            if(row[j+2] == "marathon"):
                if(row[j+3] != "-1"):
                    numTotalMarathons += 1;
                    totalTime += row[j+3]

            if (row[j][:4] == '2015') & (row[j+1] == 'marathonoasismontreal'):
                MtlMarathon2015Time = row[j+3]
                in2015MtlMarathon = 1
            else:
                racesPerYear[row[j][:4]] += 1
                if(row[j+1] == 'marathonoasismontreal'):
                    numMontrealMarathons += 1

    toAdd.append(racesPerYear['2012'])
    toAdd.append(racesPerYear['2013'])
    toAdd.append(racesPerYear['2014'])
    toAdd.append(racesPerYear['2015'])
    toAdd.append(racesPerYear['2016'])
    toAdd.append(numTotalMarathons)

    if(numTotalMarathons != 0):
        averageMarathonTime = totalTime/numTotalMarathons
        toAdd.append(averageMarathonTime)

    toAdd.append(numMontrealMarathons)

    #only if in 2015 montreal marathon
    #print(type(MtlMarathon2015Time))
    if((in2015MtlMarathon == 1) & (numTotalMarathons != 0) & (str(MtlMarathon2015Time) != "-1")):
        x.append(toAdd)
        y.append(MtlMarathon2015Time)
trainingX = []
testX = []
trainingY = []
testY = []

for i in range(len(x)):
    if(random.randint(0,4) != 0):
        trainingX.append(x[i])
        trainingY.append(y[i])
    else:
        testX.append(x[i])
        testY.append(y[i]) 

trainingX = np.array(trainingX, dtype=float)
trainingY = np.array(trainingY, dtype=float)

w = linearRegression(trainingX, trainingY)
testLinearRegression(w, testX, testY)
