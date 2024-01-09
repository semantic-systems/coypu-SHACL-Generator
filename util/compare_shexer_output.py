import rdflib
from rdflib.compare import graph_diff, to_canonical_graph
from rdflib import RDF,SH
from pprint import pprint

g1_path = "out_0/shexer_result.ttl"
g2_path = "out/pykeen_TuckER_1_81_shexer_result.ttl"

g1 = rdflib.Graph().parse(g1_path)
g2 = rdflib.Graph().parse(g2_path)

def normalize(g:rdflib.Graph, uri:rdflib.URIRef):
    return g.namespace_manager.normalizeUri(uri.toPython())

def get_shexer_constraints(g:rdflib.Graph):
    res = {}
    for node_shape in g.subjects(RDF.type, SH.NodeShape):
        target_class = normalize(g,list(g.objects(node_shape, SH.targetClass))[0])

        props = {}
        for property in g.objects(node_shape, SH.property):
            prop_path = normalize(g,list(g.objects(property, SH.path))[0])
            constaints = []
            for p,o in g.predicate_objects(property):
                if p == SH["in"]:
                    in_members = [normalize(g,m) for _,m in g.predicate_objects(object)]
                    constaints.append((normalize(g,p), tuple(in_members)))
                elif p != SH.path:
                    constaints.append((normalize(g,p), normalize(g,o)))
            props[prop_path] = tuple(constaints)

        res[target_class] = props

    return res

p1 = get_shexer_constraints(g1)
p2 = get_shexer_constraints(g2)

def get_differences(data1, data2):
    for target, properties1 in data1.items():
        properties2 = data2[target]
        for prop_path,constraints1 in properties1.items():
            constaints2 = properties2[prop_path] if prop_path in properties2 else []
            for constraint in constraints1:
                if  target not in data2 or \
                    prop_path not in properties2 or \
                    constraint not in constaints2:
                    yield (target, prop_path, constraint)


print(f"Constraints from {g1_path} not part of {g2_path}:")
num = 0
g1_vs_g2 = set()
for target, prop_path, constraint in get_differences(p1,p2):
    print(target, prop_path, constraint)
    g1_vs_g2.add((target, prop_path, constraint))
    num += 1
print(f"Total: {num}")

num = 0
g2_vs_g1 = set()
print(f"Constraints from {g2_path} not part of {g1_path}:")
for target, prop_path, constraint in get_differences(p2,p1):
    print(target, prop_path, constraint)
    g2_vs_g1.add((target, prop_path, constraint))
    num += 1
print(f"Total: {num}")

assert len(g1_vs_g2.intersection(g2_vs_g1)) == 0


