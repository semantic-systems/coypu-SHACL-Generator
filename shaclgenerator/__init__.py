from abc import ABC, abstractmethod


class SHACLGenerator(ABC):
    @abstractmethod
    def generate_shacl(self):
        pass
