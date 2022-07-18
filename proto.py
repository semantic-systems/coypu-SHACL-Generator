from logging import exception
import sys
from rdflib import Graph, URIRef, Literal, XSD
import numpy as np
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (auc, average_precision_score, 
                              roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.base import BaseEstimator

#get numerical value for objects
#objects have an 'objects.dataype' as well as 'type(object)'
#can be improved by more sophisticated conversion
def convert(object):
    #check for class Literal. Everything else is discarded.
    if type(object) == Literal:
        #check for numerics
        if((object.isdigit()) or (object.datatype == XSD.integer)):    
            #return numeric value
            return int(object)
        #for strings return lenght
        else:
            return len(object)
    #standard case is 0 (no string, no int)
    return 0

#loading the knowledge graph from file
def loadGraph():
    #load correct knowledge-graph from command line
    graph = Graph()
    graph.parse(sys.argv[1])
    #i.e. graph.parse('/Training74//mergedGraph257.nt')

    #graph syntax check
    for subj, pred, obj in graph:
        if(subj, pred, obj) not in graph:
            raise exception("It better be!")
    return graph

#takes an rdf-graph and returns a corresponding matrix
def preprocessing(graph):
    #saves the graph as a list (end product)
    graphArray = []

    #helpful variables for filling the matrix
    graphPredicates = [] #saves a set of all properties in a graph
    graphPredicatesCount = [] #saves the max cardinality for each property in a graph

    #saves a set of all subjects in a graph (useful only for iteration)
    graphSubjects = []

    #iterates through all subjects of a graph to transform into a numerical version
    for s in graph.subjects():
        if s not in graphSubjects:
            #updates iteration over subjects
            graphSubjects.append(s)

            #creates a Concise Bounded Description (CBD) for a given ressource
            cbd = Graph.cbd(graph, s)
            #TEST
            #print("New CBD\n")

            #appends [subject] and empty predicates to array
            graphArray.append([s])
            for element in graphPredicates:
                #adds all existing predicates with empty values to the end of 's'
                graphArray[-1].append(element)
                graphArray[-1].append([0,0])

            #helpful variable
            cbdPredicates = []

            for pred in cbd.predicates(s,None):
                if pred not in cbdPredicates:
                    cbdPredicates.append(pred)
                    predGraph = Graph()
                    predGraph += cbd.triples((s,pred,None))
                    
                    #get count for pred
                    count = len(predGraph)
                    #set init rangeCount for pred
                    rangeCount = 0

                    #get rangeCount
                    cbdObjects = []
                    rangeCountTypes = []
                    for obj in predGraph.objects():
                        if obj not in cbdObjects:
                            cbdObjects.append(obj)
                            a = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                            type = graph.value(obj, a)
                            if type not in rangeCountTypes:
                                rangeCountTypes.append(type)
                                rangeCount += 1
                            #print(obj, type)
                            
                            #adds element to graphArray
                            #check for existing or new predicate
                            if pred in graphPredicates:
                                #convert into proper datatype
                                addObj = convert(obj)
                                #add obj to graphArray
                                ind = ((graphPredicates.index(pred)*2) + 2)
                                graphArray[-1][ind].append(addObj)
                            else:
                                #update graphPredicates with new predicate
                                graphPredicates.append(pred)
                                #update graphPredicatesCount with new Count
                                graphPredicatesCount.append(count)

                                #append pred for all PREVIOUS entries
                                for newPredInd in range(0,len(graphArray)-1):
                                    graphArray[newPredInd].append(pred)
                                    graphArray[newPredInd].append([0,0])
                                #append pred for current entry
                                graphArray[-1].append(pred)

                                #convert into proper datatype
                                addObj = convert(obj)
                                #add correct obj value into array
                                graphArray[-1].append([0,0,addObj])

                    #adds count and rangeCount into array/list
                    ind = ((graphPredicates.index(pred)*2) + 2)   
                    graphArray[-1][ind][0] = count
                    graphArray[-1][ind][1] = rangeCount
                    #update graphPredicatesCount
                    if(count > graphPredicatesCount[graphPredicates.index(pred)]):
                        graphPredicatesCount[graphPredicates.index(pred)] = count

    #add zeros to fill up
    #iterate through all subject-vectors         
    for resS in range (0, len(graphArray)):
        #iterate through all predicates starting at index 1 bc index 0 is the subject name
        for resP in range (1, len(graphArray[resS]),2):
            #get the current predicate
            fillPred = graphArray[resS][resP]
            #get the index of the current predicate
            ind = graphPredicates.index(fillPred)
            #see how many entries that predicate should have
            maxCount = graphPredicatesCount[ind]
            #TEST
            #print("Das ist:",len(graphArray[resS][resP+1]), "Das müsste:", maxCount+2, "\n")
            #add the necessary empty values(0)
            #+2 to take into account the range and rangeCount attributes
            while len(graphArray[resS][resP+1]) < (maxCount + 2):
                graphArray[resS][resP+1].append(0)

    #TEST
    #print("End-Array:")
    #for f in graph:
    #    print("Next Subject:",f,"\n")
    #print("End-Array Done\n")
    return graphArray, graphSubjects, graphPredicates, graphPredicatesCount

#cuts all the subjects and predicates and leaves only the objects/numbers
def listToArray(listArray):
    #reduce list to only the object dimension
    #resulting list is two dimensional with vectors of equal lenght for every subject of a graph
    tempList = []
    for s in range(0,len(listArray)):
        tempList.append([])
        for p in range (2, len(listArray[s]), 2):
            for o in range (0, len(listArray[s][p])):
                tempList[s].append(listArray[s][p][o])
    
    #convert to numpy for better calculation
    npArray = np.array(tempList)

    #normalize vectors of matrix
    npArray = normalize(npArray, norm='l1')
    return npArray  

#taken from 'https://donernesto.github.io/blog/outlier-detection-with-dbscan/'
def labels_from_DBclusters(db):
    """
    Returns labels for each point for "outlierness", based on DBSCAN results.
    The higher the score, the more likely the point is an outlier, based on its cluster membership
    
    - dbscan label -1 (outliers): highest score of 1
    - largest cluster gets score 0  
    - points belonging to clusters get a score that is higher when the cluster size is smaller
    
    db: a fitted DBscan instance
    Returns: labels (similar to "y_predicted", but the values merely reflect a ranking)
    """
    labels = np.zeros(len(db.labels_))
    
    # make a list of tuples: (i, num points in i) for i in db.labels_
    label_counts = [(i, np.sum(db.labels_==i)) for i in set(db.labels_) - set([-1])]
    label_counts.sort(key=lambda x : -x[1]) # sort by counts per class, descending
    
    # assign the labels. Those points with label =-1 get highest label (equal to number of classes -1) 
    labels[db.labels_== -1] = len(set(db.labels_)) - 1
    for i, (label, label_count) in enumerate(label_counts):
        labels[db.labels_==label] = i
        
    # Scale the values between 0 and 1
    labels = (labels - min(labels)) / (max(labels) - min(labels))
    return(labels)   

#TODO
#does machine learning based clustering on matrix of vectors
def clustering(graph):
    db = DBSCAN(eps=0.5, min_samples=5)
    db.fit(graph)
    return labels_from_DBclusters(db)

#TODO review
#does some postprocessing and visualization
def postprocessing(labels, subjects):
    result = []
    for e in range(0,len(labels)):
        result.append([subjects[e],labels[e]])
    
    with open('result.txt', 'w') as resfile:
        for e in result:
            resfile.write(str(e)+"\n")
    return result    

#main contains all function calls
def main():
    graph = loadGraph()
    print("Graph Loading Done!")
    graph, subjects, predicates, pCount = preprocessing(graph)
    print("Preprocessing Done!")
    graph = listToArray(graph)
    print("Numeric extraction Done!")
    labels = clustering(graph)
    print("Clustering Done!")
    result = postprocessing(labels, subjects)
    print("Done!")

    #TODO
    #Shacl Creation

main()