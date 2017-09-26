'''
Created on 2017年9月24日

@author: chc
'''
from numpy import *

count = 0
statementList = ['x' for n in range(1102)]
for line in open("statementList.txt"): 
    statementList[count] = line.rstrip()
    count = count + 1

#statementList = [word.decode('UTF-8') for word in statementList]    
#statementList = [word.decode('UTF-8') for word in statementList]
print(statementList[0])
print(statementList.index("first prime fixture session entity not exist",))

syntaxVectors = loadtxt('syntaxVector.txt')
syntaxVectors = mat(syntaxVectors, dtype = float32)
s = array(syntaxVectors, dtype = float32)
print(s.shape)