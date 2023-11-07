from setuptools import setup

setup(
    name='coypu-SHACL-Generator',
    version='',
    packages=[
        'shaclgenerator',
        'test',
        'util',
    ],
    url='',
    license='Apache License 2.0',
    author='Lars Michaelis, Patrick Westphal',
    author_email='',
    description='',
    scripts=[
        'bin/generate_shacl',
        'bin/evaluate_shacl',
    ],
    install_requires=[
        'pyshacl',
        'rdflib',
        # 'shaclgen==0.2.5.2',  # outdated
        'shaclgen @ git+https://github.com/patrickwestphal/shaclgen@0.2.5.3-patrickwestphal#egg=shaclgen',
        'shexer==2.2.1',
    ],
)
