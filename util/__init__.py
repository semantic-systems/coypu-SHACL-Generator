from enum import Enum
from rdflib import Graph, URIRef
from typing import List

from rdflib import Graph


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

def nt_to_tsv(nt_input_file_path: str, tsv_output_file_path: str):
    g = Graph()
    g.parse(nt_input_file_path, format='ntriples')

    with open(tsv_output_file_path, 'w') as out_file:
        for s, p, o in g:
            out_file.write(f'{str(s)}\t{str(p)}\t{str(o)}\n')
