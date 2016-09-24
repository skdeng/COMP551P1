import csv
import numpy as np
import unicodedata
import string
import log_reg
import lin_reg
import naive_bayes
import random
import warnings

warnings.filterwarnings("ignore")

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h, m, s))

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

def encodeLogistic(list):
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
        numMontrealMarathons = 0
        for j in range(len(row)):
            if(j-1)%5 == 0:
                racesPerYear[row[j][:4]] += 1
                if(row[j+1] == 'marathonoasismontreal') & (row[j+2] == 'marathon'):
                    numMontrealMarathons += 1
                    if (row[j][:4] == '2015'):
                        in2015MtlMarathon = 1

        toAdd.append(racesPerYear['2012'])
        toAdd.append(racesPerYear['2013'])
        toAdd.append(racesPerYear['2014'])
        toAdd.append(racesPerYear['2015'])
        toAdd.append(racesPerYear['2016'])
        toAdd.append(numMontrealMarathons)
                    
        x.append(toAdd)
        y.append(in2015MtlMarathon)

    output = []
    output.append(x)
    output.append(y)
    return output

def encodeNaive(list):
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
        mtlMarathonPerYear[str(encodeYear-3)] = 0
        mtlMarathonPerYear[str(encodeYear-2)] = 0
        mtlMarathonPerYear[str(encodeYear-1)] = 0
        mtlMarathonPerYear[str(encodeYear)] = 0
        #mtlMarathonPerYear['2015'] = 0
        #mtlMarathonPerYear['2016'] = 0
        for j in range(len(row)):
            if(j-1)%5 == 0:
                racesPerYear[row[j][:4]] = 1
                if((row[j+1] == 'marathonoasismontreal') & (row[j+2] == 'marathon')):
                    mtlMarathonPerYear[row[j][:4]] = 1
        toAdd.append(racesPerYear['2012'])
        toAdd.append(racesPerYear['2013'])
        toAdd.append(racesPerYear['2014'])
        toAdd.append(racesPerYear['2015'])
        toAdd.append(racesPerYear['2016'])
        toAdd.append(mtlMarathonPerYear[str(encodeYear-3)])
        toAdd.append(mtlMarathonPerYear[str(encodeYear-2)])
        toAdd.append(mtlMarathonPerYear[str(encodeYear-1)])
        if(mtlMarathonPerYear['2015'] == 1):
            in2015MtlMarathon = 1
        x.append(toAdd)
        y.append(in2015MtlMarathon)
    output = []
    output.append(x)
    output.append(y)
    return output
def encodeRegression(list):
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
        numMontrealMarathons = 0
        numTotalMarathons = 0
        totalTime = 0
        MtlMarathon2015Time = "-1"
        for j in range(len(row)):
            if(j-1)%5 == 0:
                racesPerYear[row[j][:4]] += 1
                if(row[j+2] == "marathon"):
                    if(row[j+3] != "-1"):
                        numTotalMarathons += 1;
                        totalTime += row[j+3]
                if (row[j+2] == "marathon") & (row[j+1] == 'marathonoasismontreal'):
                    numMontrealMarathons += 1
                    if (row[j][:4] == '2015'):
                        MtlMarathon2015Time = row[j+3]
                        in2015MtlMarathon = 1
                        
        toAdd.append(racesPerYear['2012'])
        toAdd.append(racesPerYear['2013'])
        toAdd.append(racesPerYear['2014'])
        toAdd.append(racesPerYear['2015'])
        toAdd.append(racesPerYear['2016'])
        toAdd.append(numTotalMarathons)

        if(numTotalMarathons != 0):
            averageMarathonTime = totalTime/numTotalMarathons
            toAdd.append(averageMarathonTime)
        else:
            toAdd.append(-1)
        toAdd.append(numMontrealMarathons)
        if((in2015MtlMarathon == 1) & (numTotalMarathons != 0) & (str(MtlMarathon2015Time) != "-1")):
            x.append(toAdd)
            y.append(int(MtlMarathon2015Time))

    output = []
    output.append(x)
    output.append(y)
    return output    

def encodeOutputRegression(list):
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
        numMontrealMarathons = 0
        numTotalMarathons = 0
        totalTime = 0
        MtlMarathon2015Time = "-1"
        for j in range(len(row)):
            if(j-1)%5 == 0:
                racesPerYear[row[j][:4]] += 1
                if(row[j+2] == "marathon"):
                    if(row[j+3] != "-1"):
                        numTotalMarathons += 1;
                        totalTime += row[j+3]
                if (row[j+2] == "marathon") & (row[j+1] == 'marathonoasismontreal'):
                    numMontrealMarathons += 1
                    if (row[j][:4] == '2015'):
                        MtlMarathon2015Time = row[j+3]
                        in2015MtlMarathon = 1
                        
        toAdd.append(racesPerYear['2012'])
        toAdd.append(racesPerYear['2013'])
        toAdd.append(racesPerYear['2014'])
        toAdd.append(racesPerYear['2015'])
        toAdd.append(racesPerYear['2016'])
        toAdd.append(numTotalMarathons)

        if(numTotalMarathons == 0):
            if(row[5][:1] == "F"):
                totalTime = 17000
            else:
                totalTime = 15600
            numTotalMarathons = 1

        averageMarathonTime = totalTime/numTotalMarathons
        toAdd.append(int(averageMarathonTime))

        toAdd.append(numMontrealMarathons)
        #if((in2015MtlMarathon == 1) & (numTotalMarathons != 0) & (str(MtlMarathon2015Time) != "-1")):
        x.append(toAdd)
        y.append(int(MtlMarathon2015Time))

    output = []
    output.append(x)
    output.append(y)
    return output  

def logisticRegression(X, Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    model = log_reg.Model(6, 0.1)
    for i in range(10):
        model.step(X,Y)
        #print('Iteration '+str(i)+' with error: '+str(model.error(X,Y)))
    return model
    #Z = model.forward(testX)
    #return Z

def applyLogisticModel(model, testX):
    testX = np.array(testX, dtype=float)
    Z = model.forward(testX)
    return Z


def naiveBayes(X,Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    m = naive_bayes.Model(2)
    m.fit(X,Y,True)
    #print(m.forward(testX))
    return m

def applyNaiveBayes(model, testX):
    np.set_printoptions(threshold=np.nan)
    testX = np.array(testX, dtype=float)
    return model.forward(testX)

def linearRegression(X, Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    m = lin_reg.Model(2)
    m.solve(X,Y)
    return m

def applyLinearRegression(model, testX):
    output = []
    #return m.w
    for i, row in enumerate(testX):
        row = [1] + row
        xrow = np.array(row, dtype=float)
        finishTime = int(np.dot(xrow,model.w))
        output.append(finishTime)
    return output

def testLogistic(output, testY):
    numCorrect = 0
    print("length output: "+str(len(output)))
    for i in range(len(output)):
        if(output[i][0] == testY[i]):
            numCorrect +=1
    #print("Logistic: "+str(numCorrect/len(output)))
    return numCorrect/len(output)

def testNaiveBayes(output, testY):
    numCorrect = 0
    print("length output: "+str(len(output)))
    for i in range(len(output)):
        if(output[i] == testY[i]):
            numCorrect +=1
    #print("Naive Bayes: "+str(numCorrect/len(output)))
    return numCorrect/len(output)

def testLinearRegression(testX, testY):
    #output = []
    print("length output: "+str(len(testX)))
    sumOfPercentDifference = 0
    numPercentDifference = 0
    for i, row in enumerate(testX):
        #print(row)
        #print(str(wx)+" "+str(testY[i]))
        #print(str(type(testX[i]))+" "+str(type(testY[i])))
        #print(str(testX[i])+" "+str(testY[i]))
        percentDifference = abs(testX[i] - testY[i])/testY[i]*100
        sumOfPercentDifference += percentDifference
        numPercentDifference += 1
        #output.append(percentDifference)
    #print("Regression Percent Difference: "+str(sumOfPercentDifference/numPercentDifference))
    return sumOfPercentDifference/numPercentDifference
    
def kfoldValidation(x, y, k, train, applyModel, test, name):
    partitionSize = int(len(x)/k)
    errors = []
    for i in range(0, k):
        trainingX = []
        testX = []
        trainingY = []
        testY = []      

        start = i*partitionSize
        end = (i+1)*partitionSize
        #print(str(start)+" "+str(end))
        for j, row in enumerate(x):
            if (start <= j) & (j <end):
                testX.append(x[j])
                testY.append(y[j])
            else:
                trainingX.append(x[j])
                trainingY.append(y[j])
        trainingX = np.array(trainingX, dtype=float)
        trainingY = np.array(trainingY, dtype=float)
        model = train(trainingX, trainingY)
        output = applyModel(model, testX)
        error = test(output, testY)
        errors.append(error)

    meanError = 0
    for i in errors:
        meanError += i
    meanError = meanError / k
    print(name+" error:"+ str(meanError))


dataFile = 'Project1_data.csv'
list = []
raceNames = []
raceTypes = []
raceAges = []
transform = {}

#Sanitize the data 
#['0', '2015-09-20', "Marathon Oasis Rock 'n' Roll de Montreal", 'Marathon', 14024, 'M50-54']
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
#random.shuffle(list)
encodeYear = 2015
dataLogistic = encodeLogistic(list)
x = dataLogistic[0]
y = dataLogistic[1]
logisticModel = logisticRegression(x,y)

dataNaive = encodeNaive(list)
x = dataNaive[0]
y = dataNaive[1]
naiveBayesModel = naiveBayes(x,y)

dataRegression = encodeRegression(list)
x = dataRegression[0]
y = dataRegression[1]
regressionModel = linearRegression(x,y)

encodeYear = 2016
dataLogistic = encodeLogistic(list)
x = dataLogistic[0]
logisticOutput = applyLogisticModel(logisticModel,x)

dataNaive = encodeNaive(list)
x = dataNaive[0]
naiveOutput = applyNaiveBayes(naiveBayesModel,x)

dataRegression = encodeOutputRegression(list)
x = dataRegression[0]
regressionOutput = applyLinearRegression(regressionModel,x)

f = open('output.csv','w')
for i, row in enumerate(list):
    #print(str(i)+","+str(int(logisticOutput[i][0]))+","+str(naiveOutput[i])+","+get_hms(regressionOutput[i]))
    f.write(str(i)+","+str(int(logisticOutput[i][0]))+","+str(naiveOutput[i])+","+get_hms(regressionOutput[i])+"\n")
f.close()
#kfoldValidation(x, y, 5, logisticRegression, applyLogisticModel, testLogistic, "Logistic Regression")
#kfoldValidation(x, y, 5, naiveBayes, applyNaiveBayes, testNaiveBayes, "Naive Bayes") 
#kfoldValidation(x, y, 5, linearRegression, applyLinearRegression, testLinearRegression, "Linear Regression")

