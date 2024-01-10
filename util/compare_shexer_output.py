import rdflib
from rdflib.compare import graph_diff, to_canonical_graph
from rdflib import RDF,SH
from pprint import pprint
from argparse import ArgumentParser

if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('shexer_shacl_path_1')
    argument_parser.add_argument('shexer_shacl_path_2')
    args = argument_parser.parse_args()

    g1_path = args.shexer_shacl_path_1
    g2_path = args.shexer_shacl_path_2

    g1 = rdflib.Graph().parse(g1_path)
    g2 = rdflib.Graph().parse(g2_path)

    def normalize(g:rdflib.Graph, uri:rdflib.URIRef):
        return g.namespace_manager.normalizeUri(uri.toPython())

    def get_shexer_constraints(g:rdflib.Graph):
        res = {}
        for node_shape in g.subjects(RDF.type, SH.NodeShape):
            target_class = normalize(g,list(g.objects(node_shape, SH.targetClass))[0])

            constraints = {}
            for constraint in g.objects(node_shape, SH.property):
                prop_path = normalize(g,list(g.objects(constraint, SH.path))[0])
                components = []
                for p,o in g.predicate_objects(constraint):
                    if p == SH["in"]:
                        in_members = [normalize(g,m) for _,m in g.predicate_objects(object)]
                        components.append((normalize(g,p), tuple(in_members)))
                    elif p != SH.path:
                        components.append((normalize(g,p), normalize(g,o)))

                if prop_path not in constraints:
                    constraints[prop_path] = []
                constraints[prop_path].append(tuple(components))

            res[target_class] = constraints

        return res

    p1 = get_shexer_constraints(g1)
    p2 = get_shexer_constraints(g2)

    def print_constraints(data):
        for target in sorted(data.keys()):
            for path in sorted(data[target].keys()):
                print(target, path)
                for constraint in data[target][path]:
                    pprint(constraint)

    # print("\nConstraints of", g1_path)
    # print_constraints(p1)

    # print("\nConstraints of", g2_path)
    # print_constraints(p2)

    def get_differences(data1, data2):
        for target in sorted(data1.keys()):
            properties1 = data1[target]
            properties2 = data2[target]
            for prop_path in sorted(properties1.keys()):
                constraints1 = properties1[prop_path]
                constraints2 = properties2[prop_path] if prop_path in properties2 else []
                for constraint in constraints1:
                    if  target not in data2 or \
                        prop_path not in properties2 or \
                        constraint not in constraints2:
                        yield (target, prop_path, constraint)


    print(f"Constraints from {g1_path} not part of {g2_path}:")
    num = 0
    for target, prop_path, constraints in get_differences(p1,p2):
        print(target, prop_path)
        pprint(constraints)
        num += 1
    print(f"Total: {num}")

    num = 0
    print(f"Constraints from {g2_path} not part of {g1_path}:")
    for target, prop_path, constraints in get_differences(p2,p1):
        print(target, prop_path)
        pprint(constraints)
        num += 1
    print(f"Total: {num}")



