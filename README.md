# The CoyPu SHACL Generator

## Install

To install the CoyPu SHACL Generator the source code has to be fetched from GitHub first:

```
$ git clone https://github.com/semantic-systems/coypu-SHACL-Generator.git
$ cd  coypu-SHACL-Generator/
```

To install the CoyPu SHACL Generator and all required dependencies, run

```
$ pip install .
```

After the installation the two executables `generate_shacl` and `evaluate_shacl` should be available:
```
$ generate_shacl -h
usage: generate_shacl [-h] [--shexer] [--shaclgen] [--shacl_output_directory SHACL_OUTPUT_DIRECTORY] [--shexer_acceptance_threshold SHEXER_ACCEPTANCE_THRESHOLD] [--shexer_type_property SHEXER_TYPE_PROPERTY] input_file

positional arguments:
  input_file

options:
  -h, --help            show this help message and exit
  --shexer
  --shaclgen
  --shacl_output_directory SHACL_OUTPUT_DIRECTORY
  --shexer_acceptance_threshold SHEXER_ACCEPTANCE_THRESHOLD
                        Sets shexers threshold for including shacl constraints.
  --shexer_type_property SHEXER_TYPE_PROPERTY
                        Property which indicated class membership in the input graph.
$ evaluate_shacl -h
usage: evaluate_shacl [-h] [--input_rdf_store_url INPUT_RDF_STORE_URL] [--input_rdf_file INPUT_RDF_FILE] [--auth_name AUTH_NAME] [--auth_pw AUTH_PW] input_shacl_file

positional arguments:
  input_shacl_file

options:
  -h, --help            show this help message and exit
  --input_rdf_store_url INPUT_RDF_STORE_URL
  --input_rdf_file INPUT_RDF_FILE
  --auth_name AUTH_NAME
  --auth_pw AUTH_PW
```

## SHACL Generation

For the SHACL generation part there are currently two 'backends', i.e. existing solutions, that can be used, namely SheXer and SHACLGen.
An example run for a dataset called `cities_wikidata.nt` with the SheXer backend would look like this:

```
$ generate_shacl --shexer cities_wikidata.nt
INFO:generate_shacl:generate_shacl called with /tmp/coypu-SHACL-Generator/cities_wikidata.nt and backend(s) sheXer
```

The result SHACL file will usually be inside a directory called `out/` unless configured otherwise:

```
$ ls out/
shexer_result.ttl
$ head shexer_result
@prefix : <http://weso.es/shapes/> .
@prefix geo: <http://www.opengis.net/ont/geosparql#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:City a sh:NodeShape ;
    sh:property [ a sh:PropertyShape ;
$
```


## SHACL Validation

To validate or evaluate a generated SHACL file, the `evaluate_shacl` executable is ued.
To continue the example above, the `out/shexer_result.ttl` file would be used as follows:

```
$ evaluate_shacl --input_rdf_file ~/hitec/projects/coypu/shacl_induction/datasets/cities_wikidata.nt out/shexer_result.ttl 
INFO:evaluate_shacl:data graph: [a rdfg:Graph;rdflib:storage [a rdflib:Store;rdfs:label 'Memory']].
INFO:evaluate_shacl:shacl file: out/shexer_result.ttl
```

The evaluation result will again be in the `out/` directory unless configured otherwise:

```
$ ls out/
eval_shacl.ttl
shexer_result.ttl
$ head out/eval_shacl.ttl 
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

[] a sh:ValidationReport ;
    sh:conforms false ;
    sh:result [ a sh:ValidationResult ;
            sh:focusNode <http://www.wikidata.org/entity/Q113513748> ;
            sh:resultMessage "Value <https://schema.coypu.org/global#City> not in list ['<http://www.wikidata.org/entity/Q515>']" ;
```
