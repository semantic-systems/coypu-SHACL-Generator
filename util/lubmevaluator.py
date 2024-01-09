"""
Evaluation utils for the LUMB dataset (here called Training74)
"""
from typing import List

from pykeen.datasets import Dataset
from rdflib import URIRef
import numpy as np
from sklearn.cluster import DBSCAN

ERRONEOUS_RESOURCES = {
    # Set2_errors.nt --> implausible age
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent144'),  # <http://example.org/#age> "17" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent410'),  # <http://example.org/#age> "30" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent514'),  # <http://example.org/#age> "17" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent151'),  # <http://example.org/#age> "23" .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent155'),  # <http://example.org/#age> "17" .
    URIRef('http://www.Department3.University0.edu/UndergraduateStudent96'),  # <http://example.org/#age> "28" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent419'),  # <http://example.org/#age> "21" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent306'),  # <http://example.org/#age> "20" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent40'),  # <http://example.org/#age> "27" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent126'),  # <http://example.org/#age> "30" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent138'),  # <http://example.org/#age> "20" .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent126'),  # <http://example.org/#age> "28" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent115'),  # <http://example.org/#age> "21" .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent130'),  # <http://example.org/#age> "25" .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent288'),  # <http://example.org/#age> "19" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent23'),  # <http://example.org/#age> "17" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent70'),  # <http://example.org/#age> "23" .
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent272'),  # <http://example.org/#age> "27" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent452'),  # <http://example.org/#age> "25" .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent177'),  # <http://example.org/#age> "30" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent38'),  # <http://example.org/#age> "27" .
    URIRef('http://www.Department3.University0.edu/UndergraduateStudent136'),  # <http://example.org/#age> "18" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent413'),  # <http://example.org/#age> "21" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent226'),  # <http://example.org/#age> "17" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent99'),  # <http://example.org/#age> "22" .
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent46'),  # <http://example.org/#age> "24" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent390'),  # <http://example.org/#age> "22" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent259'),  # <http://example.org/#age> "23" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent101'),  # <http://example.org/#age> "17" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent362'),  # <http://example.org/#age> "29" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent300'),  # <http://example.org/#age> "20" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent309'),  # <http://example.org/#age> "18" .
    URIRef('http://www.Department2.University0.edu/UndergraduateStudent263'),  # <http://example.org/#age> "29" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent85'),  # <http://example.org/#age> "30" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent61'),  # <http://example.org/#age> "18" .
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent178'),  # <http://example.org/#age> "27" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent301'),  # <http://example.org/#age> "22" .
    URIRef('http://www.Department14.University0.edu/UndergraduateStudent134'),  # <http://example.org/#age> "24" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent219'),  # <http://example.org/#age> "23" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent94'),  # <http://example.org/#age> "23" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent310'),  # <http://example.org/#age> "23" .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent240'),  # <http://example.org/#age> "26" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent356'),  # <http://example.org/#age> "27" .
    # Set5_errors.nt -> implausible age
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent211'),  # <http://example.org/#age> "73"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent234'),  # <http://example.org/#age> "60"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent15'),  # <http://example.org/#age> "71"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department3.University0.edu/UndergraduateStudent189'),  # <http://example.org/#age> "62"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent196'),  # <http://example.org/#age> "59"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent321'),  # <http://example.org/#age> "71"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department14.University0.edu/UndergraduateStudent73'),  # <http://example.org/#age> "65"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent11'),  # <http://example.org/#age> "62"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department14.University0.edu/UndergraduateStudent75'),  # <http://example.org/#age> "58"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent299'),  # <http://example.org/#age> "62"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent248'),  # <http://example.org/#age> "67"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department3.University0.edu/UndergraduateStudent305'),  # <http://example.org/#age> "64"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department2.University0.edu/UndergraduateStudent132'),  # <http://example.org/#age> "71"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent190'),  # <http://example.org/#age> "75"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department3.University0.edu/UndergraduateStudent127'),  # <http://example.org/#age> "72"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department3.University0.edu/UndergraduateStudent248'),  # <http://example.org/#age> "69"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent177'),  # <http://example.org/#age> "66"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent34'),  # <http://example.org/#age> "60"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent348'),  # <http://example.org/#age> "75"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent303'),  # <http://example.org/#age> "57"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent314'),  # <http://example.org/#age> "74"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent118'),  # <http://example.org/#age> "65"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent246'),  # <http://example.org/#age> "59"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent10'),  # <http://example.org/#age> "58"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent42'),  # <http://example.org/#age> "55"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent69'),  # <http://example.org/#age> "61"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department3.University0.edu/UndergraduateStudent172'),  # <http://example.org/#age> "64"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent224'),  # <http://example.org/#age> "64"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent166'),  # <http://example.org/#age> "64"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent214'),  # <http://example.org/#age> "74"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent359'),  # <http://example.org/#age> "69"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent379'),  # <http://example.org/#age> "68"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department14.University0.edu/UndergraduateStudent99'),  # <http://example.org/#age> "69"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent124'),  # <http://example.org/#age> "74"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent252'),  # <http://example.org/#age> "74"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent122'),  # <http://example.org/#age> "57"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent124'),  # <http://example.org/#age> "72"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent32'),  # <http://example.org/#age> "59"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent25'),  # <http://example.org/#age> "59"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent427'),  # <http://example.org/#age> "62"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent222'),  # <http://example.org/#age> "69"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent211'),  # <http://example.org/#age> "55"^^<http://www.w3.org/2001/XMLSchema#integer> .
    URIRef('http://www.Department2.University0.edu/UndergraduateStudent101'),  # <http://example.org/#age> "74"^^<http://www.w3.org/2001/XMLSchema#integer> .
    # Set7_errors.nt -> broken phone number
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent87'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent11'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent357'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent364'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent350'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent24'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent210'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent375'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent325'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent93'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent212'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent97'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department14.University0.edu/UndergraduateStudent142'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent398'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department14.University0.edu/UndergraduateStudent45'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent246'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department9.University0.edu/UndergraduateStudent271'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent287'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent347'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent171'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent333'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent295'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent182'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent254'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent58'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent148'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department13.University0.edu/UndergraduateStudent438'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent263'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent213'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent223'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department10.University0.edu/UndergraduateStudent322'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department2.University0.edu/UndergraduateStudent203'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent238'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department12.University0.edu/UndergraduateStudent154'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent267'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent77'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department14.University0.edu/UndergraduateStudent213'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department8.University0.edu/UndergraduateStudent71'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department11.University0.edu/UndergraduateStudent258'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent373'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department1.University0.edu/UndergraduateStudent238'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent321'),  # <http://example.org/#telephone> "xxx-xx-x" .
    URIRef('http://www.Department0.University0.edu/UndergraduateStudent195'),  # <http://example.org/#telephone> "xxx-xx-x" .
}

NUM_RESOURCES = 12868  # 12855 <- without classes


class LUMBEvaluator:
    """
    The task to evaluate here is to correctly find all outliers. The overall
    dataset (data/Training/mergedGraph257.nt) has 12868 resources (in subject
    or object position). Properties are not considered.
    The clusterer should find certain clusters and all outliers, ideally, won't
    belong to any of those clusters, and thus be in the '-1' dummy cluster
    holding all nodes not being a member of any cluster.

    So, if all outliers were found, the '-1' cluster will contain all resources
    held in ERRONEOUS_RESOURCES. Ideally, these would be the only nodes in the
    '-1' cluster.

    As the proportion of outliers and non-outliers is highly skewed (12868:130)
    just looking at the accuracy won't help much, as, e.g., with very high eps
    value in DBSCAN, we could end up with one big cluster (say cluster '1')
    containing all resources. In terms of true positives (tp),
    true negatives (tn), false positives (fp), and false negatives (fn),
    this would mean

      tp = 0  # no outlier was in the expected cluster '-1'
      tn = 12868 - 130  # all non-outliers (= all nodes minus the outliers) were
                        # correctly put into a custer other than '-1'
      fp = 0  # none the non-outliers were put into the '-1' cluster
      fn = 130  # all outliers were part of a cluster other than '-1'

    This would mean we missed all outliers entirely, however get an
    accuracy of
                                 .-- none of the actual outliers
                                 v       v- all non-outliers (all nodes minus outliers)
                tp + tn          0 + (12868-130)
    acc = ------------------- = ----------------- = 0.9898
           tp + tn + fp + fn         12868

    Hence, we will also look at the F1-Score here, which better reflects the
    circumstance:
                     2*tp               0
    f1-score = ---------------- = ------------- = 0
                2*tp + fp + fn     0 + 0 + 130
    """
    def __init__(self):
        # As we will iterate over the nodes of the '-1' cluster, we will...
        # ...count up tp if we found an actual outlier -- tp would be 130 then in
        #    the ideal case
        self.tp = 0
        # ...count down tn if we found a non-outlier in the '-1' cluster -- in
        #    the worst case of all non-outliers being in '-1', tn would be 0
        self.tn = NUM_RESOURCES - len(ERRONEOUS_RESOURCES)
        # ...count down fn in case we found an actual outlier -- in the ideal
        #    case, fn would be 0
        self.fn = len(ERRONEOUS_RESOURCES)
        # ...count up fp if we found a non-outlier in the '-1' cluster -- in the
        #    worst case of all outliers being in '-1', fn would be
        #    NUM_RESOURCES - len(ERRONEOUS_RESOURCES)
        self.fp = 0

    def evaluate_clusters(self, dataset: Dataset, clusterer: DBSCAN):
        # get all indexes for those nodes not belonging to any cluster
        indexes = np.where(clusterer.labels_ == -1)

        # reverse the entity-to-index mapping to get an index-to-entity mapping
        id_to_entity = {v: k for k, v in dataset.entity_to_id.items()}

        # build URIs for the nodes found in the '-1' cluster, i.e., the
        # outlier nodes
        outlier_nodes: List[URIRef] = \
            [
                URIRef(id_to_entity[idx])
                for idx in indexes[0]
                if id_to_entity[idx].startswith('http://')
            ]

        for outlier_node in outlier_nodes:
            if outlier_node in ERRONEOUS_RESOURCES:
                # if everything is correct, then
                self.tp += 1  # tp == len(ERRONEOUS_RESOURCES)
                self.fn -= 1  # fn == 0
            else:
                # if everything is wrong, then
                self.fp += 1  # fp == NUM_RESOURCES - len(ERRONEOUS_RESOURCES)
                self.tn -= 1  # tn == 0

    def get_acc(self):
        return (self.tp + self.tn) / NUM_RESOURCES

    def get_f1_score(self):
        return (2 * self.tp) / ((2 * self.tp) + self.fn + self.fp)
