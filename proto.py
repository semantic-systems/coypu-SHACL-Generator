from logging import exception
import sys, time
from rdflib import Graph, URIRef, Literal, XSD
import numpy as np
import sparse
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

#loading the knowledge graph from file
def loadGraph():
    #load correct knowledge-graph from command line
    graph = Graph()
    graph.parse(sys.argv[1])

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

#TODO refine
#does clustering on matrix of vectors
def clustering(graph):
    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(graph)
    return db.labels_, labels_from_DBclusters(db)

#TODO expand
#does some postprocessing and visualization
def postprocessing(rawLabels, labels, graph, subjects, predicates, objects):
    result = []
    for e in range(0,len(labels)):
        result.append([subjects[e],labels[e],rawLabels[e]])
    
    with open('result.txt', 'w') as resfile:
        for e in result:
            resfile.write(str(e)+"\n")

    X_2D = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=400, init='random').fit_transform(graph) # collapse in 2-D space for plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i in set(rawLabels):
        if i == -1: 
            #outlier according to dbscan
            ax.scatter(X_2D[rawLabels==i, 0], X_2D[rawLabels==i, 1], c='r', s=4, label='DBSCAN Outlier')
        elif i == 0: 
            #base class according to dbscan
            ax.scatter(X_2D[rawLabels==i, 0], X_2D[rawLabels==i, 1], c='b', s=4, label='DBSCAN within range')
        else:
            ax.scatter(X_2D[rawLabels==i, 0], X_2D[rawLabels==i, 1], c='b', s=4)

    plt.axis('off')
    plt.legend(loc = 5, fontsize = 8)
    plt.savefig('result.png')         

    return result

#main contains all function calls
def main():
    graph = loadGraph()
    print("Graph Loading Done!")
    start = time.time()
    graph, subjects, predicates, objects = preprocessing_tensor(graph)
    end = time.time()
    print("Preprocessing done in: {:.2f}".format(end-start))
    rawLabels, labels = clustering(graph)
    print("Clustering Done!")
    start = time.time()
    result = postprocessing(rawLabels, labels, graph, subjects, predicates, objects)
    end = time.time()
    print("Postprocessing done in: {:.2f}".format(end-start))

    #TODO
    #Shacl Creation
    
    print("Done!")

main()