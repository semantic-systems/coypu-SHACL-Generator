from io import StringIO

from rdflib import Graph, RDF
from shexer.consts import NT, SHACL_TURTLE
from shexer.shaper import Shaper

from shaclgenerator import SHACLGenerator


class ShexerAdapter(SHACLGenerator):
    def __init__(self, input_file_path_or_sparql_endpoint: str):
        if input_file_path_or_sparql_endpoint.startswith('http'):
            self.shaper = Shaper(
                all_classes_mode=True,
                url_endpoint=input_file_path_or_sparql_endpoint,
            )
        else:
            self.shaper = Shaper(
                all_classes_mode=True,
                graph_file_input=input_file_path_or_sparql_endpoint,
                input_format=NT,
            )
        self.acceptance_threshold = 0.0
        self.type_property = str(RDF.type)
            
    def generate_shacl(self) -> Graph:
        g = Graph()
        g.parse(
            StringIO(self.shaper.shex_graph(
                output_format=SHACL_TURTLE,
                string_output=True,
                acceptance_threshold=self.acceptance_threshold,
            )),
            format='ttl')

        return g
