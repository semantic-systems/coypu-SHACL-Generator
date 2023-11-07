from io import StringIO
from typing import List

from rdflib import Graph
from shexer.shaper import Shaper
from shexer.consts import NT, SHACL_TURTLE

from shaclgenerator import SHACLGenerator


class ShexerAdapter(SHACLGenerator):
    def __init__(self, input_file_path: str, target_classes: List[str]):
        self.shaper = Shaper(
            target_classes=target_classes,
            graph_file_input=input_file_path,
            input_format=NT,
        )
            
    def generate_shacl(self, min_support=0.5) -> Graph:
        g = Graph()
        g.parse(
            StringIO(self.shaper.shex_graph(
                output_format=SHACL_TURTLE,
                string_output=True,
                acceptance_threshold=min_support,
            )),
            format='ttl')

        return g