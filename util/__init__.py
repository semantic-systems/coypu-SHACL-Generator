from enum import Enum
from rdflib import Graph, URIRef
from typing import List


class BackEnd(Enum):
    SHEXER = 'sheXer'
    SHACLGEN = 'Shaclgen'

def get_all_classes(
        graph_file_input: str, 
        type_property: str) -> List[str]:

    g = Graph().parse(graph_file_input)

    classes = set()
    for s, p, o in g.triples((None, URIRef(type_property), None)):
        classes.add(str(o))

    return list(classes)