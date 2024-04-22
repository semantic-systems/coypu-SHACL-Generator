import logging
from enum import Enum

from rdflib import Graph

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('util')


class BackEnd(Enum):
    SHEXER = 'sheXer'
    SHACLGEN = 'Shaclgen'


def nt_to_tsv(nt_input_file_path: str, tsv_output_file_path: str):
    g = Graph()
    g.parse(nt_input_file_path, format='ntriples')

    with open(tsv_output_file_path, 'w') as out_file:
        for s, p, o in g:
            out_file.write(f'{str(s)}\t{str(p)}\t{str(o)}\n')
