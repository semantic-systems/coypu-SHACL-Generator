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
    #startC = time.time()
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
    #endC = time.time()
    #print("Conversion done in: {:.2f}".format(endC-startC))
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
        minSamples = 100
        db = DBSCAN(eps=(epsilon/100), min_samples=minSamples)
        db.fit(graph)
        resultTable.append([(epsilon/100), db.labels_, labels_from_DBclusters(db), minSamples])

    return resultTable

#TODO expand
#does some postprocessing and visualization
def postprocessing(erlmTable, graph, subjects, predicates, objects):
    X_2D = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=400, init='random').fit_transform(graph) # collapse in 2-D space for plotting

    #get manually inserted errors
    manErr1 = loadGraph("outlier%20detection//Training74//Set2_errors.nt")
    manErr2 = loadGraph("outlier%20detection//Training74//Set5_errors.nt")
    manErr3 = loadGraph("outlier%20detection//Training74//Set7_errors.nt")
    manErrSum = (manErr1 + manErr2) + manErr3
    errSub = list(manErrSum.subjects(unique=True))

    #gets True values at i where subjects[i] == element in errSub
    #gets False values everywhere else
    errFound = np.isin(subjects,errSub)
    #return only the indices where errFound == true
    errInd = np.where(errFound)

    #TEST
    #print("errSum non unique:",len(list(manErrSum.subjects())))
    #print(subjects)
    #print("len(errSub)):",len(errSub))
    #print("len(errFound):",len(errFound))
    #print("errFound:",errFound)

    #saving result .txt and .png for each Epsilon 
    for index, labelTable in enumerate(erlmTable):
        
        #WRITING DATA TO DISK
           
        total = len(labelTable[1])          #number of subjects
        errLabels = labelTable[1][errInd]   #works to find all values in the trueLabel table for the errors. Returns a list.
        ptotal = sum(labelTable[1] == -1)   #number of all outliers found
        tp = sum(errLabels == -1)           #number of correct outliers found
        fp = ptotal - tp                    #number of normal data determined as outliers
        ntotal = sum(labelTable[1] != -1)   #number of normal data found
        fn = len(errLabels) - tp            #number of outlier determined as normal
        tn = ntotal - fn                    #number of normal data determined as normal
        prec =  tp / (tp + fp) if (tp + fp) > 0 else 0                              #Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0                             #Recall
        f1 = 2 * ((prec * recall) / (prec + recall)) if (prec + recall) > 0 else 0  #F1 Score

        result = []
        for e in range(0,len(labelTable[1])):
            #version with copied label method
            #result.append([subjects[e],labelTable[1][e],labelTable[2][e]])
            #version with exclusively raw labels
            result.append([subjects[e],labelTable[1][e]])

        filename = "results/resultWithEpsilon{}.txt".format(labelTable[0])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as resfile:
            resfile.write("Epsilon: {} min_samples: {}\n".format(labelTable[0],labelTable[3]))
            resfile.write("Total Amount of Subjects: {}\n".format(total))
            resfile.write("True Positives: {}  False Positives: {}  Positives Total: {}\n".format(tp, fp, ptotal))
            resfile.write("True Negatives: {}  False Negatives: {}  Total Negatives: {}\n".format(tn, fn, ntotal))
            resfile.write("Precision: {:.2%} Recall: {:.2%} F1-Score: {:.2f}\n\n".format(prec, recall, f1))
            for e in result:
                resfile.write("{}\n".format(e))     

        #VISUALISATION

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        pos = errInd[0][labelTable[1][errInd[0]] == -1]
        neg = errInd[0][labelTable[1][errInd[0]] != -1] 
        
        #TEST
        #print("len(errFound):",len(errFound))
        #print("errInd[0]:", errInd[0])
        #print("len(errInd[0]):", len(errInd[0]))
        #print("pos:", pos)
        #print("len(pos):", len(pos))
        #print("len(pos) == tp:",len(pos) == tp)
        #print("neg:", neg)
        #print("len(neg):", len(neg))
        #print("len(neg) == fn:",len(neg) == fn)

        ax.scatter(X_2D[pos, 0], X_2D[pos, 1], c='r', marker='x', s=80, label='Correctly Identified Outlier')
        ax.scatter(X_2D[neg, 0], X_2D[neg, 1], c='b', marker='x', s=80, label='Falsely Overlooked Outlier')

        #outlier according to dbscan
        ax.scatter(X_2D[labelTable[1]==-1, 0], X_2D[labelTable[1]==-1, 1], c='r', s=4, label='DBSCAN Outlier')
        #base class according to dbscan
        ax.scatter(X_2D[labelTable[1]!=-1, 0], X_2D[labelTable[1]!=-1, 1], c='b', s=4, label='DBSCAN within range')

        plt.axis('off')
        plt.legend(fontsize = 8)
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

if __name__ == "__main__":
    main()