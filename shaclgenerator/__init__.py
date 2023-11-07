from abc import ABC, abstractmethod

from rdflib import Graph


class SHACLGenerator(ABC):
    @abstractmethod
    def generate_shacl(self) -> Graph:
        pass
