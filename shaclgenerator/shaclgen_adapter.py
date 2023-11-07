import io
from contextlib import redirect_stdout
from io import StringIO

from rdflib import Graph
from shaclgen.shaclgen import data_graph

from shaclgenerator import SHACLGenerator


class ShaclgenAdapter(SHACLGenerator):
    def __init__(self, input_file_path: str):
        # with open(input_file_path, 'rb') as input_file:
        #     input = input_file.read()
        g = Graph()
        g.parse(input_file_path)
        self._shaclgen = data_graph(g)

    def generate_shacl(self) -> Graph:
        return self._shaclgen.gen_graph()
