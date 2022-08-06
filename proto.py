from logging import exception
import sys, time, os

from rdflib import Graph, URIRef, Literal, XSD
import numpy as np
import sparse
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

#loading the knowledge graph from file
def loadGraph(source):
    #load correct knowledge-graph from command line
    graph = Graph()
    graph.parse(source)

    #graph syntax check
    for subj, pred, obj in graph:
        if(subj, pred, obj) not in graph:
            raise exception("It better be!")
    return graph

#takes an rdf-graph and returns a matrix
def preprocessing_tensor(graph):

    graphPredicates = [] #saves a set of all properties in a graph
    graphSubjects = [] #saves a set of all subjects in a graph
    graphObjects = [] #saves a set of all objects in a graph

    #INIT MATRIX

    subjectSet = set()
    for s in graph.subjects(unique=True):
        subjectSet.add(s)
    graphSubjects = list(subjectSet)

    predicateSet = set()
    for p in graph.predicates(unique=True):
        predicateSet.add(p)
    graphPredicates = list(predicateSet)

    objectSet = set()
    for o in graph.objects(unique=True):
        if(type(o)== URIRef):
            objectSet.add(o)
    graphObjects = list(objectSet)

    graphMatrix = sparse.DOK(shape=(len(graphPredicates),len(graphSubjects),len(graphObjects)), dtype=int)
    
    #Fills matrix with data
    #iterates through all subjects of a graph and adds corresponding entry into the matrix
    for subIndex, s in enumerate(graphSubjects):
        #creates a Concise Bounded Description (CBD) for a given ressource
        cbd = Graph.cbd(graph, s)

        for predIndex, pred in enumerate(graphPredicates):
            predGraph = Graph()
            predGraph += cbd.triples((s,pred,None))

            for obj in predGraph.objects(unique=True):

                #adds element to Matrix after checking for URI Type
                if type(obj) == URIRef:
                    #test if obj exists
                    try:
                        objIndex = graphObjects.index(obj)
                        #Test           
                        #print(obj, "<- wants to be added with possible new Index:", predIndex, subIndex, objIndex)
                        graphMatrix[predIndex,subIndex,objIndex] = 1
                    except ValueError:
                        #TEST
                        #print(obj,"not in objectlist!")
                        continue
    
    #TEST
    #print("Graphsubjects:",graphSubjects)
    #print("GraphPredicates:",graphPredicates)
    #print("GraphObjects:",graphObjects)
    #print("Graphmatrix:\n",graphMatrix.todense())

    #transform 3D matrix into correctly sliced 2D csr matrix
    #TEST
    startC = time.time()
    graphMatrix = sparse.COO(graphMatrix)

    for n in range(0,len(graphSubjects)):
        slice = graphMatrix[:,n:(n+1),:].flatten() 
        if n == 0:
            oneDMatrix = sparse.COO(slice)
        else:
            oneDMatrix = sparse.concatenate((oneDMatrix, slice), axis=0)
        #TEST
        #print("Slice of subject",n,":\n", slice.todense())

    twoDMatrix = oneDMatrix.reshape(shape=(len(graphSubjects),(len(graphObjects)*len(graphPredicates))))
    result = twoDMatrix.tocsr()
    result = normalize(result, norm='l1')
    
    #TEST
    #print("Endresult:\n",twoDMatrix.todense())
    endC = time.time()
    print("Conversion done in: {:.2f}".format(endC-startC))
    return result, graphSubjects, graphPredicates, graphObjects


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

    return labels   

#does clustering on matrix of vectors
def clustering(graph):
    resultTable = []
    for epsilon in range(30,60,5):
        db = DBSCAN(eps=(epsilon/100), min_samples=100)
        db.fit(graph)
        resultTable.append([(epsilon/100), db.labels_, labels_from_DBclusters(db)])

    return resultTable

#TODO expand
#does some postprocessing and visualization
def postprocessing(erlTable, graph, subjects, predicates, objects):
    X_2D = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=400, init='random').fit_transform(graph) # collapse in 2-D space for plotting

    #get manually inserted errors
    manErr1 = loadGraph("outlier%20detection//Training74//Set2_errors.nt")
    manErr2 = loadGraph("outlier%20detection//Training74//Set5_errors.nt")
    manErr3 = loadGraph("outlier%20detection//Training74//Set7_errors.nt")
    manErrSum = (manErr1 + manErr2) + manErr3
    errSub = list(manErrSum.subjects(unique=True))

    #get indices of errorSubjects
    errFound = np.empty(shape=(0), dtype= int)
    for enumS, sub in enumerate(subjects):
        for err in errSub:
            if sub == err:
                errFound = np.append(errFound, enumS)
    #print(subjects)
    #print("len(errSub)):",len(errSub))
    #print("len(errFound):",len(errFound))
    #print("errFound:",errFound)

    #saving result .txt and .png for each Epsilon 
    for index, labelTable in enumerate(erlTable):
        
        #SAVING DATA TO DISK
           
        total = len(labelTable[1])
        errLabels = labelTable[1][errFound] #works to find all values in the trueLabel table for the erorrs. Returns a list.
        ptotal = sum(labelTable[1] == -1)
        tp = sum(errLabels == -1)
        fp = ptotal - tp
        ntotal = sum(labelTable[1] != -1)
        fn = len(errLabels) - tp
        tn = ntotal - fn

        result = []
        for e in range(0,len(labelTable[1])):
            #version with copied label method
            #result.append([subjects[e],labelTable[1][e],labelTable[2][e]])
            #version with exclusively raw labels
            result.append([subjects[e],labelTable[1][e]])

        filename = "results/resultWithEpsilon{}.txt".format(labelTable[0])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as resfile:
            resfile.write("Epsilon:"+str(labelTable[0])+"\n")
            resfile.write("True Positives: {}  False Positives: {}  Positives Total: {}\nTrue Negatives: {}  False Negatives: {}  Total Negatives: {} \nTotal Amount of Subjects: {}\n\n"
                                        .format(tp, fp, ptotal, tn, fn, ntotal, total))
            for e in result:
                resfile.write(str(e)+"\n")         

        #VISUALISATION

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        for i in set(labelTable[1]):
            if i == -1: 
                ax.scatter(X_2D[errFound, 0], X_2D[errFound, 1], c='r', marker='x', s=80, label='True Positive DBSCAN Outlier')
            else:
                ax.scatter(X_2D[errFound, 0], X_2D[errFound, 1], c='b', marker='x', s=80, label='False Negative')

        for i in set(labelTable[1]):
            if i == -1: 
                #outlier according to dbscan
                ax.scatter(X_2D[labelTable[1]==i, 0], X_2D[labelTable[1]==i, 1], c='r', s=4, label='DBSCAN Outlier')
            elif i == 0: 
                #base class according to dbscan
                ax.scatter(X_2D[labelTable[1]==i, 0], X_2D[labelTable[1]==i, 1], c='b', s=4, label='DBSCAN within range')
            else:
                ax.scatter(X_2D[labelTable[1]==i, 0], X_2D[labelTable[1]==i, 1], c='b', s=4)

        plt.axis('off')
        plt.legend(loc = 5, fontsize = 8)
        plt.savefig('results/resultWithEpsilon{}.png'.format(labelTable[0]))

    return result

#main contains all function calls
def main():
    start = time.time()
    graph = loadGraph(sys.argv[1])
    end = time.time()
    print("Loading Graph done in: {:.2f}".format(end-start))

    start = time.time()
    graph, subjects, predicates, objects = preprocessing_tensor(graph)
    end = time.time()
    print("Preprocessing done in: {:.2f}".format(end-start))

    start = time.time()
    erlTable = clustering(graph)
    end = time.time()
    print("Clustering done in: {:.2f}".format(end-start))

    start = time.time()
    result = postprocessing(erlTable, graph, subjects, predicates, objects)
    end = time.time()
    print("Postprocessing done in: {:.2f}".format(end-start))

    #TODO
    #Shacl Creation
    
    print("Done!")

main()