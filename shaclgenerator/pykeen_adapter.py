import os
import tempfile
from enum import Enum
from typing import List

import numpy as np
from pykeen.datasets import Dataset
from pykeen.triples import TriplesFactory
from rdflib import Graph, URIRef
from pykeen.pipeline import pipeline
from shaclgen.shaclgen import data_graph
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import util.cache as cache
from lubmevaluator import LUMBEvaluator
from util.cache import CacheMiss
from shaclgenerator import SHACLGenerator
from util import nt_to_tsv


class EmbeddingMethod(Enum):
    # Nope, 0 clusters found on Training74 for all eps values (0 <= ... <= 1)
    AutoSF = 'AutoSF'

    # Training74 results:
    # current eps: 0.50 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.51 -> 2 clusters found; f1-score: 0.0197
    # current eps: 0.52 -> 2 clusters found; f1-score: 0.0197
    # current eps: 0.53 -> 2 clusters found; f1-score: 0.0198
    # current eps: 0.54 -> 3 clusters found; f1-score: 0.0199
    # current eps: 0.55 -> 4 clusters found; f1-score: 0.0200
    # current eps: 0.56 -> 5 clusters found; f1-score: 0.0202
    # current eps: 0.57 -> 6 clusters found; f1-score: 0.0204
    # current eps: 0.58 -> 7 clusters found; f1-score: 0.0207
    # current eps: 0.59 -> 6 clusters found; f1-score: 0.0209
    # current eps: 0.60 -> 5 clusters found; f1-score: 0.0211
    # current eps: 0.61 -> 5 clusters found; f1-score: 0.0209
    # current eps: 0.62 -> 5 clusters found; f1-score: 0.0212
    # current eps: 0.63 -> 6 clusters found; f1-score: 0.0219
    # current eps: 0.64 -> 9 clusters found; f1-score: 0.0222
    # current eps: 0.65 -> 8 clusters found; f1-score: 0.0229
    # current eps: 0.66 -> 8 clusters found; f1-score: 0.0236
    # current eps: 0.67 -> 7 clusters found; f1-score: 0.0245
    # current eps: 0.68 -> 6 clusters found; f1-score: 0.0246
    # current eps: 0.69 -> 6 clusters found; f1-score: 0.0243
    # current eps: 0.70 -> 7 clusters found; f1-score: 0.0252
    # current eps: 0.71 -> 7 clusters found; f1-score: 0.0259
    # current eps: 0.72 -> 6 clusters found; f1-score: 0.0267
    # current eps: 0.73 -> 7 clusters found; f1-score: 0.0269
    # current eps: 0.74 -> 7 clusters found; f1-score: 0.0272
    # current eps: 0.75 -> 7 clusters found; f1-score: 0.0260
    # current eps: 0.76 -> 6 clusters found; f1-score: 0.0272
    # current eps: 0.77 -> 5 clusters found; f1-score: 0.0277
    # current eps: 0.78 -> 5 clusters found; f1-score: 0.0290
    # current eps: 0.79 -> 5 clusters found; f1-score: 0.0282
    # current eps: 0.80 -> 6 clusters found; f1-score: 0.0295
    # current eps: 0.81 -> 6 clusters found; f1-score: 0.0290
    # current eps: 0.82 -> 6 clusters found; f1-score: 0.0318  <- best
    # current eps: 0.83 -> 6 clusters found; f1-score: 0.0272
    # current eps: 0.84 -> 7 clusters found; f1-score: 0.0221
    # current eps: 0.85 -> 7 clusters found; f1-score: 0.0202
    # current eps: 0.86 -> 6 clusters found; f1-score: 0.0217
    # current eps: 0.87 -> 6 clusters found; f1-score: 0.0226
    # current eps: 0.88 -> 6 clusters found; f1-score: 0.0230
    # current eps: 0.89 -> 3 clusters found; f1-score: 0.0269
    # current eps: 0.90 -> 3 clusters found; f1-score: 0.0229
    # current eps: 0.91 -> 3 clusters found; f1-score: 0.0211
    # current eps: 0.92 -> 3 clusters found; f1-score: 0.0185
    # current eps: 0.93 -> 2 clusters found; f1-score: 0.0211
    # current eps: 0.94 -> 2 clusters found; f1-score: 0.0243
    # current eps: 0.95 -> 2 clusters found; f1-score: 0.0090
    # current eps: 0.96 -> 2 clusters found; f1-score: 0.0099
    # current eps: 0.97 -> 2 clusters found; f1-score: 0.0109
    # current eps: 0.98 -> 2 clusters found; f1-score: 0.0116
    # current eps: 0.99 -> 2 clusters found; f1-score: 0.0125
    # current eps: 1.00 -> 2 clusters found; f1-score: 0.0130
    # current eps: 1.01 -> 2 clusters found; f1-score: 0.0134
    BoxE = 'BoxE'

    CompGCN = 'CompGCN'  # Causes assertion error on training
    ComplEx = 'ComplEx'  # Complex data not supported in DBSCAN
    ConvE = 'ConvE'  # Not enough RAM + RAM leakage
    ConvKB = 'ConvKB'  # Not enough RAM
    CP = 'CP'  # 3 dimensional, skipped

    # Training74 results:
    # current eps: 1.21 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.22 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.23 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.24 -> 2 clusters found; f1-score: 0.0197
    # current eps: 1.20 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.25 -> 6 clusters found; f1-score: 0.0197 <- best
    # current eps: 1.26 -> 12 clusters found; f1-score: 0.0194
    # current eps: 1.27 -> 19 clusters found; f1-score: 0.0195
    # current eps: 1.28 -> 45 clusters found; f1-score: 0.0194
    # current eps: 1.29 -> 62 clusters found; f1-score: 0.0189
    # current eps: 1.30 -> 93 clusters found; f1-score: 0.0187
    # current eps: 1.31 -> 84 clusters found; f1-score: 0.0187
    # current eps: 1.32 -> 84 clusters found; f1-score: 0.0188
    # current eps: 1.33 -> 83 clusters found; f1-score: 0.0186
    # current eps: 1.34 -> 65 clusters found; f1-score: 0.0183
    # current eps: 1.35 -> 59 clusters found; f1-score: 0.0181
    # current eps: 1.36 -> 45 clusters found; f1-score: 0.0169
    # current eps: 1.37 -> 44 clusters found; f1-score: 0.0176
    # current eps: 1.38 -> 26 clusters found; f1-score: 0.0149
    # current eps: 1.39 -> 13 clusters found; f1-score: 0.0152
    # current eps: 1.40 -> 8 clusters found; f1-score: 0.0142
    # current eps: 1.41 -> 3 clusters found; f1-score: 0.0115
    # current eps: 1.42 -> 1 clusters found; f1-score: 0.0118
    # current eps: 1.43 -> 1 clusters found; f1-score: 0.0088
    # current eps: 1.44 -> 1 clusters found; f1-score: 0.0082
    # current eps: 1.45 -> 1 clusters found; f1-score: 0.0080
    # current eps: 1.46 -> 1 clusters found; f1-score: 0.0045
    # current eps: 1.47 -> 1 clusters found; f1-score: 0.0058
    # current eps: 1.48 -> 1 clusters found; f1-score: 0.0051
    # current eps: 1.49 -> 1 clusters found; f1-score: 0.0065
    # current eps: 1.50 -> 1 clusters found; f1-score: 0.0083
    CrossE = 'CrossE'

    DistMA = 'DistMA'  # No clusters found on Training74

    # Training74 results:
    # current eps: 0.94 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.95 -> 18 clusters found; f1-score: 0.0197
    # current eps: 0.96 -> 53 clusters found; f1-score: 0.0196
    # current eps: 0.97 -> 179 clusters found; f1-score: 0.0195
    # current eps: 0.98 -> 457 clusters found; f1-score: 0.0193
    # current eps: 0.99 -> 790 clusters found; f1-score: 0.0206  <- best
    # current eps: 1.00 -> 593 clusters found; f1-score: 0.0183
    # current eps: 1.01 -> 135 clusters found; f1-score: 0.0120
    # current eps: 1.02 -> 15 clusters found; f1-score: 0.0067
    DistMult = 'DistMult'

    ERMLP = 'ERMLP'  # No clusters found on Training74
    ERMLPE = 'ERMLPE'  # No clusters found on Training74

    # Training74 results:
    # current eps: 1.09 -> 3 clusters found; f1-score: 0.0195
    # current eps: 1.10 -> 15 clusters found; f1-score: 0.0195  <- best
    # current eps: 1.11 -> 23 clusters found; f1-score: 0.0193
    # current eps: 1.12 -> 18 clusters found; f1-score: 0.0188
    # current eps: 1.13 -> 13 clusters found; f1-score: 0.0175
    # current eps: 1.14 -> 14 clusters found; f1-score: 0.0159
    # current eps: 1.15 -> 6 clusters found; f1-score: 0.0135
    # current eps: 1.16 -> 13 clusters found; f1-score: 0.0133
    # current eps: 1.17 -> 18 clusters found; f1-score: 0.0100
    # current eps: 1.18 -> 28 clusters found; f1-score: 0.0082
    # current eps: 1.19 -> 25 clusters found; f1-score: 0.0057
    # current eps: 1.20 -> 29 clusters found; f1-score: 0.0014
    HolE = 'HolE'

    # Training74 results:
    # current eps: 1.09 -> 3 clusters found; f1-score: 0.0195
    # current eps: 1.10 -> 15 clusters found; f1-score: 0.0195  <- best
    # current eps: 1.11 -> 23 clusters found; f1-score: 0.0193
    # current eps: 1.12 -> 18 clusters found; f1-score: 0.0188
    # current eps: 1.13 -> 13 clusters found; f1-score: 0.0175
    # current eps: 1.14 -> 14 clusters found; f1-score: 0.0159
    # current eps: 1.15 -> 6 clusters found; f1-score: 0.0135
    # current eps: 1.16 -> 13 clusters found; f1-score: 0.0133
    # current eps: 1.17 -> 18 clusters found; f1-score: 0.0100
    # current eps: 1.18 -> 28 clusters found; f1-score: 0.0082
    # current eps: 1.19 -> 25 clusters found; f1-score: 0.0057
    # current eps: 1.20 -> 29 clusters found; f1-score: 0.0014
    KG2E = 'KG2E'

    # Training74 results:
    # current eps: 0.13 -> 1 clusters found; f1-score: 0.0196
    # current eps: 0.14 -> 3 clusters found; f1-score: 0.0196
    # current eps: 0.15 -> 5 clusters found; f1-score: 0.0196
    # current eps: 0.16 -> 10 clusters found; f1-score: 0.0197
    # current eps: 0.17 -> 19 clusters found; f1-score: 0.0199
    # current eps: 0.18 -> 13 clusters found; f1-score: 0.0201
    # current eps: 0.19 -> 12 clusters found; f1-score: 0.0206
    # current eps: 0.20 -> 12 clusters found; f1-score: 0.0214
    # current eps: 0.21 -> 16 clusters found; f1-score: 0.0227
    # current eps: 0.22 -> 19 clusters found; f1-score: 0.0243
    # current eps: 0.23 -> 11 clusters found; f1-score: 0.0257
    # current eps: 0.24 -> 15 clusters found; f1-score: 0.0265
    # current eps: 0.25 -> 13 clusters found; f1-score: 0.0265  <- best
    # current eps: 0.26 -> 13 clusters found; f1-score: 0.0239
    # current eps: 0.27 -> 10 clusters found; f1-score: 0.0245
    # current eps: 0.28 -> 6 clusters found; f1-score: 0.0240
    # current eps: 0.29 -> 3 clusters found; f1-score: 0.0260
    # current eps: 0.30 -> 4 clusters found; f1-score: 0.0248
    # current eps: 0.31 -> 2 clusters found; f1-score: 0.0242
    # current eps: 0.32 -> 3 clusters found; f1-score: 0.0156
    # current eps: 0.33 -> 3 clusters found; f1-score: 0.0171
    # current eps: 0.34 -> 2 clusters found; f1-score: 0.0147
    # current eps: 0.35 -> 1 clusters found; f1-score: 0.0156
    # current eps: 0.36 -> 1 clusters found; f1-score: 0.0152
    # current eps: 0.37 -> 1 clusters found; f1-score: 0.0194
    # current eps: 0.38 -> 1 clusters found; f1-score: 0.0237
    # current eps: 0.39 -> 1 clusters found; f1-score: 0.0174
    # current eps: 0.40 -> 1 clusters found; f1-score: 0.0100
    # current eps: 0.41 -> 1 clusters found; f1-score: 0.0110
    # current eps: 0.42 -> 1 clusters found; f1-score: 0.0117
    # current eps: 0.43 -> 1 clusters found; f1-score: 0.0123
    MuRE = 'MuRE'

    NodePiece = 'NodePiece'  # Needs reverse triples and thus throws exception
    NTN = 'NTN'  # No clusters found on Training74

    # Training74 results:
    # current eps: 1.13 -> 3 clusters found; f1-score: 0.0197
    # current eps: 1.14 -> 14 clusters found; f1-score: 0.0197
    # current eps: 1.15 -> 60 clusters found; f1-score: 0.0197
    # current eps: 1.16 -> 175 clusters found; f1-score: 0.0199
    # current eps: 1.17 -> 268 clusters found; f1-score: 0.0208
    # current eps: 1.18 -> 289 clusters found; f1-score: 0.0235
    # current eps: 1.19 -> 191 clusters found; f1-score: 0.0263
    # current eps: 1.20 -> 45 clusters found; f1-score: 0.0270  <- best
    # current eps: 1.21 -> 3 clusters found; f1-score: 0.0171
    # current eps: 1.22 -> 1 clusters found; f1-score: 0.0217
    PairRE = 'PairRE'

    # Training74 results:
    # current eps: 1.25 -> 5 clusters found; f1-score: 0.0197
    # current eps: 1.26 -> 8 clusters found; f1-score: 0.0197
    # current eps: 1.27 -> 17 clusters found; f1-score: 0.0197
    # current eps: 1.28 -> 21 clusters found; f1-score: 0.0198
    # current eps: 1.29 -> 44 clusters found; f1-score: 0.0199
    # current eps: 1.30 -> 59 clusters found; f1-score: 0.0197
    # current eps: 1.31 -> 80 clusters found; f1-score: 0.0199
    # current eps: 1.32 -> 104 clusters found; f1-score: 0.0196
    # current eps: 1.33 -> 113 clusters found; f1-score: 0.0191
    # current eps: 1.34 -> 103 clusters found; f1-score: 0.0196
    # current eps: 1.35 -> 96 clusters found; f1-score: 0.0197
    # current eps: 1.36 -> 54 clusters found; f1-score: 0.0207
    # current eps: 1.37 -> 43 clusters found; f1-score: 0.0206
    # current eps: 1.38 -> 40 clusters found; f1-score: 0.0188
    # current eps: 1.39 -> 24 clusters found; f1-score: 0.0169
    # current eps: 1.40 -> 10 clusters found; f1-score: 0.0179
    # current eps: 1.41 -> 4 clusters found; f1-score: 0.0184
    # current eps: 1.42 -> 4 clusters found; f1-score: 0.0182
    # current eps: 1.43 -> 2 clusters found; f1-score: 0.0199
    # current eps: 1.44 -> 2 clusters found; f1-score: 0.0202
    # current eps: 1.45 -> 1 clusters found; f1-score: 0.0187
    # current eps: 1.46 -> 1 clusters found; f1-score: 0.0191
    # current eps: 1.47 -> 2 clusters found; f1-score: 0.0203
    # current eps: 1.48 -> 1 clusters found; f1-score: 0.0187
    # current eps: 1.49 -> 1 clusters found; f1-score: 0.0165
    # current eps: 1.50 -> 1 clusters found; f1-score: 0.0210
    # current eps: 1.51 -> 1 clusters found; f1-score: 0.0188
    # current eps: 1.52 -> 1 clusters found; f1-score: 0.0141
    # current eps: 1.53 -> 1 clusters found; f1-score: 0.0174
    # current eps: 1.54 -> 1 clusters found; f1-score: 0.0206
    # current eps: 1.55 -> 1 clusters found; f1-score: 0.0242  <- best
    # current eps: 1.56 -> 1 clusters found; f1-score: 0.0186
    # current eps: 1.57 -> 1 clusters found; f1-score: 0.0211
    ProjE = 'ProjE'

    QuatE = 'QuatE'  # Found array with dim 3. DBSCAN expected <= 2.

    # Training74 results:
    # current eps: 1.58 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.59 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.60 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.61 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.62 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.63 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.64 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.65 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.66 -> 2 clusters found; f1-score: 0.0197
    # current eps: 1.67 -> 4 clusters found; f1-score: 0.0197
    # current eps: 1.68 -> 5 clusters found; f1-score: 0.0197
    # current eps: 1.69 -> 8 clusters found; f1-score: 0.0196
    # current eps: 1.70 -> 9 clusters found; f1-score: 0.0196
    # current eps: 1.71 -> 13 clusters found; f1-score: 0.0196
    # current eps: 1.72 -> 15 clusters found; f1-score: 0.0196
    # current eps: 1.73 -> 17 clusters found; f1-score: 0.0197
    # current eps: 1.74 -> 20 clusters found; f1-score: 0.0194
    # current eps: 1.75 -> 32 clusters found; f1-score: 0.0193
    # current eps: 1.76 -> 40 clusters found; f1-score: 0.0194
    # current eps: 1.77 -> 51 clusters found; f1-score: 0.0195
    # current eps: 1.78 -> 58 clusters found; f1-score: 0.0195
    # current eps: 1.79 -> 69 clusters found; f1-score: 0.0197
    # current eps: 1.80 -> 73 clusters found; f1-score: 0.0198
    # current eps: 1.81 -> 83 clusters found; f1-score: 0.0193
    # current eps: 1.82 -> 92 clusters found; f1-score: 0.0194
    # current eps: 1.83 -> 84 clusters found; f1-score: 0.0184
    # current eps: 1.84 -> 86 clusters found; f1-score: 0.0180
    # current eps: 1.85 -> 77 clusters found; f1-score: 0.0182
    # current eps: 1.86 -> 62 clusters found; f1-score: 0.0179
    # current eps: 1.87 -> 51 clusters found; f1-score: 0.0190
    # current eps: 1.88 -> 38 clusters found; f1-score: 0.0189
    # current eps: 1.89 -> 29 clusters found; f1-score: 0.0192
    # current eps: 1.90 -> 23 clusters found; f1-score: 0.0195
    # current eps: 1.91 -> 16 clusters found; f1-score: 0.0180
    # current eps: 1.92 -> 19 clusters found; f1-score: 0.0180
    # current eps: 1.93 -> 10 clusters found; f1-score: 0.0189
    # current eps: 1.94 -> 5 clusters found; f1-score: 0.0178
    # current eps: 1.95 -> 2 clusters found; f1-score: 0.0188
    # current eps: 1.96 -> 1 clusters found; f1-score: 0.0202
    # current eps: 1.97 -> 1 clusters found; f1-score: 0.0203  <- best
    # current eps: 1.98 -> 1 clusters found; f1-score: 0.0196
    # current eps: 1.99 -> 1 clusters found; f1-score: 0.0182
    RESCAL = 'RESCAL'

    # Training74 results:
    # current eps: 0.80 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.81 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.82 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.83 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.84 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.85 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.86 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.87 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.88 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.89 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.90 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.91 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.92 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.93 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.94 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.95 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.96 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.97 -> 2 clusters found; f1-score: 0.0197
    # current eps: 0.98 -> 3 clusters found; f1-score: 0.0197
    # current eps: 0.99 -> 3 clusters found; f1-score: 0.0197
    # current eps: 1.00 -> 3 clusters found; f1-score: 0.0197
    # current eps: 1.01 -> 3 clusters found; f1-score: 0.0197
    # current eps: 1.02 -> 4 clusters found; f1-score: 0.0197
    # current eps: 1.03 -> 5 clusters found; f1-score: 0.0197
    # current eps: 1.04 -> 7 clusters found; f1-score: 0.0197
    # current eps: 1.05 -> 6 clusters found; f1-score: 0.0197
    # current eps: 1.06 -> 5 clusters found; f1-score: 0.0197
    # current eps: 1.07 -> 5 clusters found; f1-score: 0.0197
    # current eps: 1.08 -> 5 clusters found; f1-score: 0.0197
    # current eps: 1.09 -> 6 clusters found; f1-score: 0.0198
    # current eps: 1.10 -> 6 clusters found; f1-score: 0.0198
    # current eps: 1.11 -> 6 clusters found; f1-score: 0.0198
    # current eps: 1.12 -> 7 clusters found; f1-score: 0.0198
    # current eps: 1.13 -> 8 clusters found; f1-score: 0.0198
    # current eps: 1.14 -> 7 clusters found; f1-score: 0.0198
    # current eps: 1.15 -> 7 clusters found; f1-score: 0.0198
    # current eps: 1.16 -> 7 clusters found; f1-score: 0.0199
    # current eps: 1.17 -> 8 clusters found; f1-score: 0.0199
    # current eps: 1.18 -> 9 clusters found; f1-score: 0.0199
    # current eps: 1.19 -> 8 clusters found; f1-score: 0.0199
    # current eps: 1.20 -> 7 clusters found; f1-score: 0.0199
    # current eps: 1.21 -> 7 clusters found; f1-score: 0.0199
    # current eps: 1.22 -> 7 clusters found; f1-score: 0.0200
    # current eps: 1.23 -> 6 clusters found; f1-score: 0.0200
    # current eps: 1.24 -> 6 clusters found; f1-score: 0.0200
    # current eps: 1.25 -> 4 clusters found; f1-score: 0.0200
    # current eps: 1.26 -> 4 clusters found; f1-score: 0.0200
    # current eps: 1.27 -> 6 clusters found; f1-score: 0.0200
    # current eps: 1.28 -> 6 clusters found; f1-score: 0.0200
    # current eps: 1.29 -> 6 clusters found; f1-score: 0.0200
    # current eps: 1.30 -> 7 clusters found; f1-score: 0.0201
    # current eps: 1.31 -> 8 clusters found; f1-score: 0.0201
    # current eps: 1.32 -> 10 clusters found; f1-score: 0.0201
    # current eps: 1.33 -> 12 clusters found; f1-score: 0.0201
    # current eps: 1.34 -> 12 clusters found; f1-score: 0.0201
    # current eps: 1.35 -> 12 clusters found; f1-score: 0.0202
    # current eps: 1.36 -> 11 clusters found; f1-score: 0.0202
    # current eps: 1.37 -> 14 clusters found; f1-score: 0.0202
    # current eps: 1.38 -> 15 clusters found; f1-score: 0.0203
    # current eps: 1.39 -> 16 clusters found; f1-score: 0.0203
    # current eps: 1.40 -> 17 clusters found; f1-score: 0.0204
    # current eps: 1.41 -> 17 clusters found; f1-score: 0.0204
    # current eps: 1.42 -> 17 clusters found; f1-score: 0.0204
    # current eps: 1.43 -> 16 clusters found; f1-score: 0.0204
    # current eps: 1.44 -> 14 clusters found; f1-score: 0.0205
    # current eps: 1.45 -> 14 clusters found; f1-score: 0.0205
    # current eps: 1.46 -> 16 clusters found; f1-score: 0.0206
    # current eps: 1.47 -> 15 clusters found; f1-score: 0.0206
    # current eps: 1.48 -> 15 clusters found; f1-score: 0.0207
    # current eps: 1.49 -> 15 clusters found; f1-score: 0.0207
    # current eps: 1.50 -> 14 clusters found; f1-score: 0.0208
    # current eps: 1.51 -> 13 clusters found; f1-score: 0.0209
    # current eps: 1.52 -> 14 clusters found; f1-score: 0.0209
    # current eps: 1.53 -> 14 clusters found; f1-score: 0.0210
    # current eps: 1.54 -> 15 clusters found; f1-score: 0.0210
    # current eps: 1.55 -> 14 clusters found; f1-score: 0.0211
    # current eps: 1.56 -> 13 clusters found; f1-score: 0.0212
    # current eps: 1.57 -> 13 clusters found; f1-score: 0.0212
    # current eps: 1.58 -> 14 clusters found; f1-score: 0.0213
    # current eps: 1.59 -> 13 clusters found; f1-score: 0.0214
    # current eps: 1.60 -> 13 clusters found; f1-score: 0.0215
    # current eps: 1.61 -> 16 clusters found; f1-score: 0.0216
    # current eps: 1.62 -> 16 clusters found; f1-score: 0.0216
    # current eps: 1.63 -> 15 clusters found; f1-score: 0.0217
    # current eps: 1.64 -> 15 clusters found; f1-score: 0.0218
    # current eps: 1.65 -> 15 clusters found; f1-score: 0.0219
    # current eps: 1.66 -> 17 clusters found; f1-score: 0.0220
    # current eps: 1.67 -> 15 clusters found; f1-score: 0.0221
    # current eps: 1.68 -> 14 clusters found; f1-score: 0.0222
    # current eps: 1.69 -> 17 clusters found; f1-score: 0.0224
    # current eps: 1.70 -> 15 clusters found; f1-score: 0.0225
    # current eps: 1.71 -> 14 clusters found; f1-score: 0.0226
    # current eps: 1.72 -> 14 clusters found; f1-score: 0.0227
    # current eps: 1.73 -> 13 clusters found; f1-score: 0.0228
    # current eps: 1.74 -> 12 clusters found; f1-score: 0.0229
    # current eps: 1.75 -> 10 clusters found; f1-score: 0.0230
    # current eps: 1.76 -> 12 clusters found; f1-score: 0.0231
    # current eps: 1.77 -> 12 clusters found; f1-score: 0.0232
    # current eps: 1.78 -> 12 clusters found; f1-score: 0.0234
    # current eps: 1.79 -> 13 clusters found; f1-score: 0.0235
    # current eps: 1.80 -> 14 clusters found; f1-score: 0.0236
    # current eps: 1.81 -> 16 clusters found; f1-score: 0.0238
    # current eps: 1.82 -> 20 clusters found; f1-score: 0.0239
    # current eps: 1.83 -> 18 clusters found; f1-score: 0.0240
    # current eps: 1.84 -> 19 clusters found; f1-score: 0.0242
    # current eps: 1.85 -> 20 clusters found; f1-score: 0.0243
    # current eps: 1.86 -> 20 clusters found; f1-score: 0.0244
    # current eps: 1.87 -> 19 clusters found; f1-score: 0.0245
    # current eps: 1.88 -> 20 clusters found; f1-score: 0.0247
    # current eps: 1.89 -> 21 clusters found; f1-score: 0.0248
    # current eps: 1.90 -> 20 clusters found; f1-score: 0.0249
    # current eps: 1.91 -> 20 clusters found; f1-score: 0.0250
    # current eps: 1.92 -> 22 clusters found; f1-score: 0.0251
    # current eps: 1.93 -> 22 clusters found; f1-score: 0.0252
    # current eps: 1.94 -> 21 clusters found; f1-score: 0.0253
    # current eps: 1.95 -> 24 clusters found; f1-score: 0.0254
    # current eps: 1.96 -> 24 clusters found; f1-score: 0.0255
    # current eps: 1.97 -> 20 clusters found; f1-score: 0.0256
    # current eps: 1.98 -> 23 clusters found; f1-score: 0.0258
    # current eps: 1.99 -> 24 clusters found; f1-score: 0.0259
    # current eps: 2.01 -> 27 clusters found; f1-score: 0.0261
    # current eps: 2.02 -> 27 clusters found; f1-score: 0.0262
    # current eps: 2.03 -> 26 clusters found; f1-score: 0.0264
    # current eps: 2.04 -> 21 clusters found; f1-score: 0.0265
    # current eps: 2.05 -> 22 clusters found; f1-score: 0.0266
    # current eps: 2.06 -> 25 clusters found; f1-score: 0.0268
    # current eps: 2.07 -> 25 clusters found; f1-score: 0.0269
    # current eps: 2.08 -> 24 clusters found; f1-score: 0.0269
    # current eps: 2.09 -> 25 clusters found; f1-score: 0.0270
    # current eps: 2.10 -> 24 clusters found; f1-score: 0.0272
    # current eps: 2.11 -> 22 clusters found; f1-score: 0.0273
    # current eps: 2.12 -> 22 clusters found; f1-score: 0.0274
    # current eps: 2.13 -> 21 clusters found; f1-score: 0.0275
    # current eps: 2.14 -> 23 clusters found; f1-score: 0.0277
    # current eps: 2.15 -> 24 clusters found; f1-score: 0.0277
    # current eps: 2.16 -> 25 clusters found; f1-score: 0.0278
    # current eps: 2.17 -> 25 clusters found; f1-score: 0.0279
    # current eps: 2.18 -> 24 clusters found; f1-score: 0.0280
    # current eps: 2.19 -> 25 clusters found; f1-score: 0.0281
    # current eps: 2.20 -> 22 clusters found; f1-score: 0.0282
    # current eps: 2.21 -> 22 clusters found; f1-score: 0.0283
    # current eps: 2.22 -> 24 clusters found; f1-score: 0.0285
    # current eps: 2.23 -> 25 clusters found; f1-score: 0.0286
    # current eps: 2.24 -> 27 clusters found; f1-score: 0.0286
    # current eps: 2.25 -> 27 clusters found; f1-score: 0.0288
    # current eps: 2.26 -> 30 clusters found; f1-score: 0.0286
    # current eps: 2.27 -> 30 clusters found; f1-score: 0.0287
    # current eps: 2.28 -> 30 clusters found; f1-score: 0.0288
    # current eps: 2.29 -> 30 clusters found; f1-score: 0.0289
    # current eps: 2.30 -> 31 clusters found; f1-score: 0.0289
    # current eps: 2.31 -> 29 clusters found; f1-score: 0.0290
    # current eps: 2.32 -> 27 clusters found; f1-score: 0.0291
    # current eps: 2.33 -> 28 clusters found; f1-score: 0.0290
    # current eps: 2.34 -> 27 clusters found; f1-score: 0.0291
    # current eps: 2.35 -> 30 clusters found; f1-score: 0.0290
    # current eps: 2.36 -> 29 clusters found; f1-score: 0.0291
    # current eps: 2.37 -> 31 clusters found; f1-score: 0.0292
    # current eps: 2.38 -> 31 clusters found; f1-score: 0.0293
    # current eps: 2.39 -> 33 clusters found; f1-score: 0.0294
    # current eps: 2.40 -> 35 clusters found; f1-score: 0.0291
    # current eps: 2.41 -> 36 clusters found; f1-score: 0.0290
    # current eps: 2.42 -> 34 clusters found; f1-score: 0.0290
    # current eps: 2.43 -> 35 clusters found; f1-score: 0.0292
    # current eps: 2.44 -> 34 clusters found; f1-score: 0.0290
    # current eps: 2.45 -> 37 clusters found; f1-score: 0.0291
    # current eps: 2.46 -> 39 clusters found; f1-score: 0.0292
    # current eps: 2.47 -> 43 clusters found; f1-score: 0.0291
    # current eps: 2.48 -> 49 clusters found; f1-score: 0.0288
    # current eps: 2.49 -> 43 clusters found; f1-score: 0.0289
    # current eps: 2.50 -> 38 clusters found; f1-score: 0.0290
    # current eps: 2.51 -> 35 clusters found; f1-score: 0.0291
    # current eps: 2.52 -> 36 clusters found; f1-score: 0.0292
    # current eps: 2.53 -> 38 clusters found; f1-score: 0.0294
    # current eps: 2.54 -> 36 clusters found; f1-score: 0.0295
    # current eps: 2.55 -> 34 clusters found; f1-score: 0.0296
    # current eps: 2.56 -> 36 clusters found; f1-score: 0.0298
    # current eps: 2.57 -> 34 clusters found; f1-score: 0.0301
    # current eps: 2.58 -> 31 clusters found; f1-score: 0.0303
    # current eps: 2.59 -> 33 clusters found; f1-score: 0.0300
    # current eps: 2.60 -> 30 clusters found; f1-score: 0.0297
    # current eps: 2.61 -> 30 clusters found; f1-score: 0.0296
    # current eps: 2.62 -> 29 clusters found; f1-score: 0.0298
    # current eps: 2.63 -> 30 clusters found; f1-score: 0.0300
    # current eps: 2.64 -> 27 clusters found; f1-score: 0.0299
    # current eps: 2.65 -> 28 clusters found; f1-score: 0.0300
    # current eps: 2.66 -> 28 clusters found; f1-score: 0.0302
    # current eps: 2.67 -> 29 clusters found; f1-score: 0.0300
    # current eps: 2.68 -> 29 clusters found; f1-score: 0.0303
    # current eps: 2.69 -> 33 clusters found; f1-score: 0.0302
    # current eps: 2.70 -> 31 clusters found; f1-score: 0.0304
    # current eps: 2.71 -> 28 clusters found; f1-score: 0.0304
    # current eps: 2.72 -> 29 clusters found; f1-score: 0.0303
    # current eps: 2.73 -> 31 clusters found; f1-score: 0.0306
    # current eps: 2.74 -> 33 clusters found; f1-score: 0.0300
    # current eps: 2.75 -> 34 clusters found; f1-score: 0.0302
    # current eps: 2.76 -> 35 clusters found; f1-score: 0.0304
    # current eps: 2.77 -> 35 clusters found; f1-score: 0.0307
    # current eps: 2.78 -> 40 clusters found; f1-score: 0.0307
    # current eps: 2.79 -> 41 clusters found; f1-score: 0.0308
    # current eps: 2.80 -> 40 clusters found; f1-score: 0.0309
    # current eps: 2.81 -> 39 clusters found; f1-score: 0.0311
    # current eps: 2.82 -> 39 clusters found; f1-score: 0.0314
    # current eps: 2.83 -> 37 clusters found; f1-score: 0.0316
    # current eps: 2.84 -> 36 clusters found; f1-score: 0.0319
    # current eps: 2.85 -> 35 clusters found; f1-score: 0.0321
    # current eps: 2.86 -> 34 clusters found; f1-score: 0.0314
    # current eps: 2.87 -> 32 clusters found; f1-score: 0.0317
    # current eps: 2.88 -> 31 clusters found; f1-score: 0.0320
    # current eps: 2.89 -> 31 clusters found; f1-score: 0.0322
    # current eps: 2.90 -> 30 clusters found; f1-score: 0.0322
    # current eps: 2.91 -> 28 clusters found; f1-score: 0.0324
    # current eps: 2.92 -> 27 clusters found; f1-score: 0.0324
    # current eps: 2.93 -> 28 clusters found; f1-score: 0.0327
    # current eps: 2.94 -> 29 clusters found; f1-score: 0.0330
    # current eps: 2.95 -> 29 clusters found; f1-score: 0.0330
    # current eps: 2.96 -> 28 clusters found; f1-score: 0.0325
    # current eps: 2.97 -> 26 clusters found; f1-score: 0.0328
    # current eps: 2.98 -> 26 clusters found; f1-score: 0.0331
    # current eps: 2.99 -> 23 clusters found; f1-score: 0.0334
    # current eps: 3.00 -> 24 clusters found; f1-score: 0.0337
    # current eps: 3.01 -> 24 clusters found; f1-score: 0.0340
    # current eps: 3.02 -> 25 clusters found; f1-score: 0.0343
    # current eps: 3.03 -> 23 clusters found; f1-score: 0.0338
    # current eps: 3.04 -> 25 clusters found; f1-score: 0.0342
    # current eps: 3.05 -> 26 clusters found; f1-score: 0.0345
    # current eps: 3.06 -> 27 clusters found; f1-score: 0.0347
    # current eps: 3.07 -> 29 clusters found; f1-score: 0.0340
    # current eps: 3.08 -> 32 clusters found; f1-score: 0.0344
    # current eps: 3.09 -> 30 clusters found; f1-score: 0.0343
    # current eps: 3.10 -> 30 clusters found; f1-score: 0.0346
    # current eps: 3.11 -> 30 clusters found; f1-score: 0.0344
    # current eps: 3.12 -> 28 clusters found; f1-score: 0.0344
    # current eps: 3.13 -> 27 clusters found; f1-score: 0.0343
    # current eps: 3.14 -> 28 clusters found; f1-score: 0.0346
    # current eps: 3.15 -> 28 clusters found; f1-score: 0.0348
    # current eps: 3.16 -> 29 clusters found; f1-score: 0.0351
    # current eps: 3.17 -> 27 clusters found; f1-score: 0.0350
    # current eps: 3.18 -> 27 clusters found; f1-score: 0.0348
    # current eps: 3.19 -> 25 clusters found; f1-score: 0.0350
    # current eps: 3.20 -> 25 clusters found; f1-score: 0.0349
    # current eps: 3.21 -> 25 clusters found; f1-score: 0.0344
    # current eps: 3.22 -> 24 clusters found; f1-score: 0.0342
    # current eps: 3.23 -> 24 clusters found; f1-score: 0.0345
    # current eps: 3.24 -> 24 clusters found; f1-score: 0.0349
    # current eps: 3.25 -> 25 clusters found; f1-score: 0.0349
    # current eps: 3.26 -> 30 clusters found; f1-score: 0.0349
    # current eps: 3.27 -> 28 clusters found; f1-score: 0.0352
    # current eps: 3.28 -> 27 clusters found; f1-score: 0.0355
    # current eps: 3.29 -> 26 clusters found; f1-score: 0.0353
    # current eps: 3.30 -> 28 clusters found; f1-score: 0.0352
    # current eps: 3.31 -> 26 clusters found; f1-score: 0.0346
    # current eps: 3.32 -> 25 clusters found; f1-score: 0.0349
    # current eps: 3.33 -> 25 clusters found; f1-score: 0.0351
    # current eps: 3.34 -> 25 clusters found; f1-score: 0.0349
    # current eps: 3.35 -> 26 clusters found; f1-score: 0.0347
    # current eps: 3.36 -> 24 clusters found; f1-score: 0.0350
    # current eps: 3.37 -> 24 clusters found; f1-score: 0.0343
    # current eps: 3.38 -> 23 clusters found; f1-score: 0.0346
    # current eps: 3.39 -> 22 clusters found; f1-score: 0.0349
    # current eps: 3.40 -> 19 clusters found; f1-score: 0.0351
    # current eps: 3.41 -> 18 clusters found; f1-score: 0.0353
    # current eps: 3.42 -> 17 clusters found; f1-score: 0.0355
    # current eps: 3.43 -> 18 clusters found; f1-score: 0.0352
    # current eps: 3.44 -> 18 clusters found; f1-score: 0.0356
    # current eps: 3.45 -> 20 clusters found; f1-score: 0.0358  <- best
    # current eps: 3.46 -> 18 clusters found; f1-score: 0.0356
    # current eps: 3.47 -> 18 clusters found; f1-score: 0.0348
    # current eps: 3.48 -> 18 clusters found; f1-score: 0.0351
    # current eps: 3.49 -> 17 clusters found; f1-score: 0.0348
    # current eps: 3.50 -> 19 clusters found; f1-score: 0.0351
    # current eps: 3.51 -> 19 clusters found; f1-score: 0.0348
    # current eps: 3.52 -> 22 clusters found; f1-score: 0.0352
    # current eps: 3.53 -> 23 clusters found; f1-score: 0.0349
    # current eps: 3.54 -> 22 clusters found; f1-score: 0.0347
    # current eps: 3.55 -> 23 clusters found; f1-score: 0.0350
    # current eps: 3.56 -> 21 clusters found; f1-score: 0.0348
    # current eps: 3.57 -> 21 clusters found; f1-score: 0.0351
    # current eps: 3.58 -> 21 clusters found; f1-score: 0.0353
    # current eps: 3.59 -> 21 clusters found; f1-score: 0.0356
    # current eps: 3.60 -> 23 clusters found; f1-score: 0.0353
    # current eps: 3.61 -> 23 clusters found; f1-score: 0.0338
    # current eps: 3.62 -> 22 clusters found; f1-score: 0.0342
    # current eps: 3.63 -> 19 clusters found; f1-score: 0.0338
    # current eps: 3.64 -> 17 clusters found; f1-score: 0.0334
    # current eps: 3.65 -> 17 clusters found; f1-score: 0.0319
    # current eps: 3.66 -> 17 clusters found; f1-score: 0.0322
    # current eps: 3.67 -> 17 clusters found; f1-score: 0.0319
    # current eps: 3.68 -> 18 clusters found; f1-score: 0.0321
    # current eps: 3.69 -> 21 clusters found; f1-score: 0.0324
    # current eps: 3.70 -> 20 clusters found; f1-score: 0.0327
    # current eps: 3.71 -> 17 clusters found; f1-score: 0.0330
    # current eps: 3.72 -> 17 clusters found; f1-score: 0.0325
    # current eps: 3.73 -> 19 clusters found; f1-score: 0.0322
    # current eps: 3.74 -> 19 clusters found; f1-score: 0.0325
    # current eps: 3.75 -> 18 clusters found; f1-score: 0.0314
    # current eps: 3.76 -> 19 clusters found; f1-score: 0.0316
    # current eps: 3.77 -> 19 clusters found; f1-score: 0.0306
    # current eps: 3.78 -> 20 clusters found; f1-score: 0.0307
    # current eps: 3.79 -> 22 clusters found; f1-score: 0.0311
    # current eps: 3.80 -> 22 clusters found; f1-score: 0.0314
    # current eps: 3.81 -> 22 clusters found; f1-score: 0.0309
    # current eps: 3.82 -> 22 clusters found; f1-score: 0.0313
    # current eps: 3.83 -> 22 clusters found; f1-score: 0.0308
    # current eps: 3.84 -> 22 clusters found; f1-score: 0.0312
    # current eps: 3.85 -> 18 clusters found; f1-score: 0.0314
    # current eps: 3.86 -> 18 clusters found; f1-score: 0.0302
    # current eps: 3.87 -> 19 clusters found; f1-score: 0.0304
    # current eps: 3.88 -> 19 clusters found; f1-score: 0.0300
    # current eps: 3.89 -> 19 clusters found; f1-score: 0.0294
    # current eps: 3.90 -> 18 clusters found; f1-score: 0.0297
    # current eps: 3.91 -> 18 clusters found; f1-score: 0.0293
    # current eps: 3.92 -> 18 clusters found; f1-score: 0.0289
    # current eps: 3.93 -> 18 clusters found; f1-score: 0.0283
    # current eps: 3.94 -> 17 clusters found; f1-score: 0.0287
    # current eps: 3.95 -> 15 clusters found; f1-score: 0.0289
    # current eps: 3.96 -> 16 clusters found; f1-score: 0.0292
    # current eps: 3.97 -> 15 clusters found; f1-score: 0.0294
    # current eps: 3.98 -> 14 clusters found; f1-score: 0.0297
    # current eps: 3.99 -> 13 clusters found; f1-score: 0.0291
    # current eps: 4.00 -> 12 clusters found; f1-score: 0.0286
    # current eps: 4.01 -> 11 clusters found; f1-score: 0.0279
    # current eps: 4.02 -> 11 clusters found; f1-score: 0.0275
    # current eps: 4.03 -> 11 clusters found; f1-score: 0.0277
    # current eps: 4.04 -> 12 clusters found; f1-score: 0.0279
    # current eps: 4.05 -> 12 clusters found; f1-score: 0.0273
    # current eps: 4.06 -> 14 clusters found; f1-score: 0.0277
    # current eps: 4.07 -> 14 clusters found; f1-score: 0.0279
    # current eps: 4.08 -> 15 clusters found; f1-score: 0.0272
    # current eps: 4.09 -> 15 clusters found; f1-score: 0.0266
    # current eps: 4.10 -> 15 clusters found; f1-score: 0.0269
    # current eps: 4.11 -> 15 clusters found; f1-score: 0.0271
    # current eps: 4.12 -> 15 clusters found; f1-score: 0.0274
    # current eps: 4.13 -> 15 clusters found; f1-score: 0.0267
    # current eps: 4.14 -> 15 clusters found; f1-score: 0.0258
    # current eps: 4.15 -> 16 clusters found; f1-score: 0.0261
    # current eps: 4.16 -> 16 clusters found; f1-score: 0.0263
    # current eps: 4.17 -> 16 clusters found; f1-score: 0.0267
    # current eps: 4.18 -> 15 clusters found; f1-score: 0.0269
    # current eps: 4.19 -> 11 clusters found; f1-score: 0.0261
    # current eps: 4.20 -> 12 clusters found; f1-score: 0.0264
    # current eps: 4.21 -> 12 clusters found; f1-score: 0.0266
    # current eps: 4.22 -> 11 clusters found; f1-score: 0.0269
    # current eps: 4.23 -> 11 clusters found; f1-score: 0.0260
    # current eps: 4.24 -> 11 clusters found; f1-score: 0.0263
    # current eps: 4.25 -> 11 clusters found; f1-score: 0.0266
    # current eps: 4.26 -> 11 clusters found; f1-score: 0.0257
    # current eps: 4.27 -> 13 clusters found; f1-score: 0.0260
    # current eps: 4.28 -> 13 clusters found; f1-score: 0.0262
    # current eps: 4.29 -> 13 clusters found; f1-score: 0.0264
    # current eps: 4.30 -> 13 clusters found; f1-score: 0.0243
    # current eps: 4.31 -> 13 clusters found; f1-score: 0.0246
    # current eps: 4.32 -> 13 clusters found; f1-score: 0.0237
    # current eps: 4.33 -> 12 clusters found; f1-score: 0.0240
    # current eps: 4.34 -> 11 clusters found; f1-score: 0.0243
    # current eps: 4.35 -> 11 clusters found; f1-score: 0.0233
    # current eps: 4.36 -> 12 clusters found; f1-score: 0.0224
    # current eps: 4.37 -> 12 clusters found; f1-score: 0.0227
    # current eps: 4.38 -> 11 clusters found; f1-score: 0.0229
    # current eps: 4.39 -> 12 clusters found; f1-score: 0.0231
    # current eps: 4.40 -> 12 clusters found; f1-score: 0.0233
    # current eps: 4.41 -> 12 clusters found; f1-score: 0.0222
    # current eps: 4.42 -> 11 clusters found; f1-score: 0.0224
    # current eps: 4.43 -> 11 clusters found; f1-score: 0.0225
    # current eps: 4.44 -> 11 clusters found; f1-score: 0.0227
    # current eps: 4.45 -> 11 clusters found; f1-score: 0.0216
    # current eps: 4.46 -> 9 clusters found; f1-score: 0.0220
    # current eps: 4.47 -> 9 clusters found; f1-score: 0.0222
    # current eps: 4.48 -> 9 clusters found; f1-score: 0.0223
    # current eps: 4.49 -> 9 clusters found; f1-score: 0.0226
    # current eps: 4.50 -> 9 clusters found; f1-score: 0.0228
    # current eps: 4.51 -> 8 clusters found; f1-score: 0.0230
    # current eps: 4.52 -> 9 clusters found; f1-score: 0.0219
    # current eps: 4.53 -> 8 clusters found; f1-score: 0.0207
    # current eps: 4.54 -> 8 clusters found; f1-score: 0.0210
    # current eps: 4.55 -> 8 clusters found; f1-score: 0.0212
    # current eps: 4.56 -> 9 clusters found; f1-score: 0.0214
    # current eps: 4.57 -> 9 clusters found; f1-score: 0.0216
    # current eps: 4.58 -> 8 clusters found; f1-score: 0.0219
    # current eps: 4.59 -> 7 clusters found; f1-score: 0.0222
    # current eps: 4.60 -> 7 clusters found; f1-score: 0.0209
    # current eps: 4.61 -> 7 clusters found; f1-score: 0.0211
    # current eps: 4.62 -> 7 clusters found; f1-score: 0.0198
    # current eps: 4.63 -> 7 clusters found; f1-score: 0.0199
    # current eps: 4.64 -> 7 clusters found; f1-score: 0.0201
    # current eps: 4.65 -> 7 clusters found; f1-score: 0.0203
    # current eps: 4.66 -> 6 clusters found; f1-score: 0.0205
    # current eps: 4.67 -> 6 clusters found; f1-score: 0.0207
    # current eps: 4.68 -> 6 clusters found; f1-score: 0.0208
    # current eps: 4.69 -> 6 clusters found; f1-score: 0.0211
    # current eps: 4.70 -> 5 clusters found; f1-score: 0.0212
    # current eps: 4.71 -> 5 clusters found; f1-score: 0.0198
    # current eps: 4.72 -> 5 clusters found; f1-score: 0.0199
    # current eps: 4.73 -> 5 clusters found; f1-score: 0.0202
    # current eps: 4.74 -> 5 clusters found; f1-score: 0.0203
    # current eps: 4.75 -> 5 clusters found; f1-score: 0.0204
    # current eps: 4.76 -> 6 clusters found; f1-score: 0.0206
    # current eps: 4.77 -> 6 clusters found; f1-score: 0.0208
    # current eps: 4.78 -> 6 clusters found; f1-score: 0.0193
    # current eps: 4.79 -> 6 clusters found; f1-score: 0.0194
    # current eps: 4.80 -> 6 clusters found; f1-score: 0.0195
    # current eps: 4.81 -> 8 clusters found; f1-score: 0.0199
    # current eps: 4.82 -> 7 clusters found; f1-score: 0.0199
    # current eps: 4.83 -> 6 clusters found; f1-score: 0.0183
    # current eps: 4.84 -> 6 clusters found; f1-score: 0.0184
    # current eps: 4.85 -> 6 clusters found; f1-score: 0.0186
    # current eps: 4.86 -> 6 clusters found; f1-score: 0.0187
    # current eps: 4.87 -> 6 clusters found; f1-score: 0.0189
    # current eps: 4.88 -> 6 clusters found; f1-score: 0.0191
    # current eps: 4.89 -> 6 clusters found; f1-score: 0.0193
    # current eps: 4.90 -> 5 clusters found; f1-score: 0.0194
    # current eps: 4.91 -> 5 clusters found; f1-score: 0.0195
    # current eps: 4.92 -> 5 clusters found; f1-score: 0.0196
    # current eps: 4.93 -> 5 clusters found; f1-score: 0.0197
    # current eps: 4.94 -> 5 clusters found; f1-score: 0.0200
    # current eps: 4.95 -> 5 clusters found; f1-score: 0.0202
    # current eps: 4.96 -> 5 clusters found; f1-score: 0.0203
    # current eps: 4.97 -> 5 clusters found; f1-score: 0.0204
    # current eps: 4.98 -> 5 clusters found; f1-score: 0.0205
    # current eps: 4.99 -> 6 clusters found; f1-score: 0.0208
    # current eps: 5.00 -> 6 clusters found; f1-score: 0.0210
    # current eps: 5.01 -> 6 clusters found; f1-score: 0.0212
    # current eps: 5.02 -> 6 clusters found; f1-score: 0.0213
    # current eps: 5.03 -> 6 clusters found; f1-score: 0.0214
    # current eps: 5.04 -> 6 clusters found; f1-score: 0.0216
    # current eps: 5.05 -> 6 clusters found; f1-score: 0.0218
    # current eps: 5.06 -> 6 clusters found; f1-score: 0.0198
    # current eps: 5.07 -> 5 clusters found; f1-score: 0.0199
    # current eps: 5.08 -> 5 clusters found; f1-score: 0.0177
    # current eps: 5.09 -> 5 clusters found; f1-score: 0.0178
    # current eps: 5.10 -> 6 clusters found; f1-score: 0.0158
    # current eps: 5.11 -> 5 clusters found; f1-score: 0.0159
    # current eps: 5.12 -> 5 clusters found; f1-score: 0.0160
    # current eps: 5.13 -> 5 clusters found; f1-score: 0.0138
    # current eps: 5.14 -> 5 clusters found; f1-score: 0.0138
    # current eps: 5.15 -> 5 clusters found; f1-score: 0.0139
    # current eps: 5.16 -> 5 clusters found; f1-score: 0.0117
    # current eps: 5.17 -> 5 clusters found; f1-score: 0.0117
    # current eps: 5.18 -> 5 clusters found; f1-score: 0.0118
    # current eps: 5.19 -> 5 clusters found; f1-score: 0.0118
    # current eps: 5.20 -> 5 clusters found; f1-score: 0.0119
    # current eps: 5.21 -> 6 clusters found; f1-score: 0.0120
    # current eps: 5.22 -> 5 clusters found; f1-score: 0.0121
    # current eps: 5.23 -> 5 clusters found; f1-score: 0.0121
    # current eps: 5.24 -> 5 clusters found; f1-score: 0.0122
    # current eps: 5.25 -> 5 clusters found; f1-score: 0.0123
    # current eps: 5.26 -> 5 clusters found; f1-score: 0.0124
    # current eps: 5.27 -> 6 clusters found; f1-score: 0.0125
    # current eps: 5.28 -> 6 clusters found; f1-score: 0.0101
    # current eps: 5.29 -> 6 clusters found; f1-score: 0.0102
    # current eps: 5.30 -> 5 clusters found; f1-score: 0.0102
    # current eps: 5.31 -> 5 clusters found; f1-score: 0.0103
    # current eps: 5.32 -> 6 clusters found; f1-score: 0.0104
    # current eps: 5.33 -> 6 clusters found; f1-score: 0.0104
    # current eps: 5.34 -> 6 clusters found; f1-score: 0.0105
    # current eps: 5.35 -> 6 clusters found; f1-score: 0.0106
    # current eps: 5.36 -> 6 clusters found; f1-score: 0.0107
    # current eps: 5.37 -> 5 clusters found; f1-score: 0.0107
    # current eps: 5.38 -> 6 clusters found; f1-score: 0.0107
    # current eps: 5.39 -> 6 clusters found; f1-score: 0.0108
    # current eps: 5.40 -> 6 clusters found; f1-score: 0.0109
    # current eps: 5.41 -> 6 clusters found; f1-score: 0.0109
    # current eps: 5.42 -> 6 clusters found; f1-score: 0.0109
    # current eps: 5.43 -> 6 clusters found; f1-score: 0.0110
    # current eps: 5.44 -> 6 clusters found; f1-score: 0.0110
    # current eps: 5.45 -> 6 clusters found; f1-score: 0.0110
    # current eps: 5.46 -> 6 clusters found; f1-score: 0.0083
    # current eps: 5.47 -> 6 clusters found; f1-score: 0.0084
    # current eps: 5.48 -> 6 clusters found; f1-score: 0.0084
    # current eps: 5.49 -> 6 clusters found; f1-score: 0.0084
    # current eps: 5.50 -> 6 clusters found; f1-score: 0.0084
    # current eps: 5.51 -> 6 clusters found; f1-score: 0.0084
    # current eps: 5.52 -> 6 clusters found; f1-score: 0.0085
    # current eps: 5.53 -> 6 clusters found; f1-score: 0.0085
    # current eps: 5.54 -> 6 clusters found; f1-score: 0.0085
    # current eps: 5.55 -> 6 clusters found; f1-score: 0.0085
    # current eps: 5.56 -> 6 clusters found; f1-score: 0.0086
    # current eps: 5.57 -> 6 clusters found; f1-score: 0.0086
    # current eps: 5.58 -> 6 clusters found; f1-score: 0.0086
    # current eps: 5.59 -> 7 clusters found; f1-score: 0.0087
    # current eps: 5.60 -> 6 clusters found; f1-score: 0.0088
    # current eps: 5.61 -> 6 clusters found; f1-score: 0.0088
    # current eps: 5.62 -> 6 clusters found; f1-score: 0.0088
    # current eps: 5.63 -> 6 clusters found; f1-score: 0.0089
    # current eps: 5.64 -> 6 clusters found; f1-score: 0.0089
    # current eps: 5.65 -> 6 clusters found; f1-score: 0.0089
    # current eps: 5.66 -> 7 clusters found; f1-score: 0.0090
    # current eps: 5.67 -> 7 clusters found; f1-score: 0.0091
    # current eps: 5.68 -> 7 clusters found; f1-score: 0.0091
    # current eps: 5.69 -> 7 clusters found; f1-score: 0.0091
    # current eps: 5.70 -> 7 clusters found; f1-score: 0.0091
    # current eps: 5.71 -> 7 clusters found; f1-score: 0.0061
    # current eps: 5.72 -> 7 clusters found; f1-score: 0.0061
    # current eps: 5.73 -> 7 clusters found; f1-score: 0.0062
    # current eps: 5.74 -> 7 clusters found; f1-score: 0.0062
    # current eps: 5.75 -> 7 clusters found; f1-score: 0.0062
    # current eps: 5.76 -> 7 clusters found; f1-score: 0.0062
    # current eps: 5.77 -> 7 clusters found; f1-score: 0.0062
    # current eps: 5.78 -> 8 clusters found; f1-score: 0.0031
    # current eps: 5.79 -> 8 clusters found; f1-score: 0.0031
    # current eps: 5.80 -> 8 clusters found; f1-score: 0.0031
    # current eps: 5.81 -> 8 clusters found; f1-score: 0.0031
    # current eps: 5.82 -> 8 clusters found; f1-score: 0.0032
    # current eps: 5.83 -> 8 clusters found; f1-score: 0.0032
    # current eps: 5.84 -> 8 clusters found; f1-score: 0.0032
    # current eps: 5.85 -> 8 clusters found; f1-score: 0.0032
    # current eps: 5.86 -> 8 clusters found; f1-score: 0.0032
    # current eps: 5.87 -> 8 clusters found; f1-score: 0.0032
    # current eps: 5.88 -> 8 clusters found; f1-score: 0.0032
    # current eps: 5.89 -> 8 clusters found; f1-score: 0.0033
    # current eps: 5.90 -> 8 clusters found; f1-score: 0.0033
    # current eps: 5.91 -> 8 clusters found; f1-score: 0.0033
    # current eps: 5.92 -> 8 clusters found; f1-score: 0.0033
    # current eps: 5.93 -> 8 clusters found; f1-score: 0.0033
    # current eps: 5.94 -> 8 clusters found; f1-score: 0.0033
    # current eps: 5.95 -> 9 clusters found; f1-score: 0.0033
    # current eps: 5.96 -> 9 clusters found; f1-score: 0.0033
    # current eps: 5.97 -> 9 clusters found; f1-score: 0.0033
    # current eps: 5.98 -> 8 clusters found; f1-score: 0.0034
    # current eps: 5.99 -> 8 clusters found; f1-score: 0.0034
    # current eps: 6.00 -> 9 clusters found; f1-score: 0.0034
    # current eps: 6.01 -> 10 clusters found; f1-score: 0.0034
    # current eps: 6.02 -> 10 clusters found; f1-score: 0.0035
    # current eps: 6.03 -> 10 clusters found; f1-score: 0.0035
    # current eps: 6.04 -> 10 clusters found; f1-score: 0.0035
    # current eps: 6.05 -> 9 clusters found; f1-score: 0.0035
    # current eps: 6.06 -> 9 clusters found; f1-score: 0.0035
    # current eps: 6.07 -> 9 clusters found; f1-score: 0.0035
    # current eps: 6.08 -> 9 clusters found; f1-score: 0.0035
    # current eps: 6.09 -> 9 clusters found; f1-score: 0.0035
    # current eps: 6.10 -> 9 clusters found; f1-score: 0.0035
    # current eps: 6.11 -> 8 clusters found; f1-score: 0.0035
    # current eps: 6.12 -> 9 clusters found; f1-score: 0.0035
    # current eps: 6.13 -> 9 clusters found; f1-score: 0.0036
    # current eps: 6.14 -> 9 clusters found; f1-score: 0.0036
    # current eps: 6.15 -> 9 clusters found; f1-score: 0.0036
    # current eps: 6.16 -> 9 clusters found; f1-score: 0.0037
    # current eps: 6.17 -> 9 clusters found; f1-score: 0.0037
    # current eps: 6.18 -> 9 clusters found; f1-score: 0.0037
    # current eps: 6.19 -> 9 clusters found; f1-score: 0.0037
    # current eps: 6.20 -> 9 clusters found; f1-score: 0.0038
    # current eps: 6.21 -> 9 clusters found; f1-score: 0.0038
    # current eps: 6.22 -> 9 clusters found; f1-score: 0.0038
    # current eps: 6.23 -> 9 clusters found; f1-score: 0.0038
    # current eps: 6.24 -> 8 clusters found; f1-score: 0.0038
    # current eps: 6.25 -> 8 clusters found; f1-score: 0.0038
    # current eps: 6.26 -> 8 clusters found; f1-score: 0.0038
    # current eps: 6.27 -> 8 clusters found; f1-score: 0.0038
    # current eps: 6.28 -> 8 clusters found; f1-score: 0.0039
    # current eps: 6.29 -> 8 clusters found; f1-score: 0.0039
    # current eps: 6.30 -> 8 clusters found; f1-score: 0.0040
    # current eps: 6.31 -> 8 clusters found; f1-score: 0.0040
    # current eps: 6.32 -> 8 clusters found; f1-score: 0.0040
    # current eps: 6.33 -> 7 clusters found; f1-score: 0.0040
    # current eps: 6.34 -> 7 clusters found; f1-score: 0.0041
    # current eps: 6.35 -> 7 clusters found; f1-score: 0.0041
    # current eps: 6.36 -> 8 clusters found; f1-score: 0.0041
    # current eps: 6.37 -> 8 clusters found; f1-score: 0.0041
    # current eps: 6.38 -> 7 clusters found; f1-score: 0.0042
    # current eps: 6.39 -> 7 clusters found; f1-score: 0.0042
    # current eps: 6.40 -> 7 clusters found; f1-score: 0.0042
    # current eps: 6.41 -> 7 clusters found; f1-score: 0.0042
    # current eps: 6.42 -> 7 clusters found; f1-score: 0.0043
    # current eps: 6.43 -> 7 clusters found; f1-score: 0.0043
    # current eps: 6.44 -> 7 clusters found; f1-score: 0.0043
    # current eps: 6.45 -> 7 clusters found; f1-score: 0.0043
    # current eps: 6.46 -> 7 clusters found; f1-score: 0.0044
    # current eps: 6.47 -> 7 clusters found; f1-score: 0.0044
    # current eps: 6.48 -> 7 clusters found; f1-score: 0.0044
    # current eps: 6.49 -> 7 clusters found; f1-score: 0.0044
    # current eps: 6.50 -> 6 clusters found; f1-score: 0.0045
    # current eps: 6.51 -> 6 clusters found; f1-score: 0.0045
    # current eps: 6.52 -> 6 clusters found; f1-score: 0.0045
    # current eps: 6.53 -> 6 clusters found; f1-score: 0.0045
    # current eps: 6.54 -> 6 clusters found; f1-score: 0.0046
    # current eps: 6.55 -> 6 clusters found; f1-score: 0.0046
    # current eps: 6.56 -> 6 clusters found; f1-score: 0.0046
    # current eps: 6.57 -> 6 clusters found; f1-score: 0.0046
    # current eps: 6.58 -> 6 clusters found; f1-score: 0.0046
    # current eps: 6.59 -> 5 clusters found; f1-score: 0.0046
    # current eps: 6.60 -> 5 clusters found; f1-score: 0.0047
    # current eps: 6.61 -> 4 clusters found; f1-score: 0.0047
    # current eps: 6.62 -> 4 clusters found; f1-score: 0.0047
    # current eps: 6.63 -> 4 clusters found; f1-score: 0.0047
    # current eps: 6.64 -> 4 clusters found; f1-score: 0.0047
    # current eps: 6.65 -> 4 clusters found; f1-score: 0.0048
    # current eps: 6.66 -> 4 clusters found; f1-score: 0.0048
    # current eps: 6.67 -> 4 clusters found; f1-score: 0.0048
    # current eps: 6.68 -> 4 clusters found; f1-score: 0.0048
    # current eps: 6.69 -> 4 clusters found; f1-score: 0.0048
    # current eps: 6.70 -> 4 clusters found; f1-score: 0.0048
    # current eps: 6.71 -> 4 clusters found; f1-score: 0.0049
    # current eps: 6.72 -> 5 clusters found; f1-score: 0.0049
    # current eps: 6.73 -> 5 clusters found; f1-score: 0.0049
    # current eps: 6.74 -> 5 clusters found; f1-score: 0.0050
    # current eps: 6.75 -> 5 clusters found; f1-score: 0.0050
    # current eps: 6.76 -> 5 clusters found; f1-score: 0.0050
    # current eps: 6.77 -> 4 clusters found; f1-score: 0.0051
    # current eps: 6.78 -> 4 clusters found; f1-score: 0.0051
    # current eps: 6.79 -> 4 clusters found; f1-score: 0.0051
    # current eps: 6.80 -> 4 clusters found; f1-score: 0.0051
    # current eps: 6.81 -> 3 clusters found; f1-score: 0.0052
    # current eps: 6.82 -> 3 clusters found; f1-score: 0.0052
    # current eps: 6.83 -> 3 clusters found; f1-score: 0.0052
    # current eps: 6.84 -> 3 clusters found; f1-score: 0.0052
    # current eps: 6.85 -> 3 clusters found; f1-score: 0.0052
    # current eps: 6.86 -> 3 clusters found; f1-score: 0.0053
    # current eps: 6.87 -> 3 clusters found; f1-score: 0.0053
    # current eps: 6.88 -> 3 clusters found; f1-score: 0.0053
    # current eps: 6.89 -> 2 clusters found; f1-score: 0.0053
    # current eps: 6.90 -> 2 clusters found; f1-score: 0.0054
    # current eps: 6.91 -> 2 clusters found; f1-score: 0.0054
    # current eps: 6.92 -> 2 clusters found; f1-score: 0.0054
    # current eps: 6.93 -> 2 clusters found; f1-score: 0.0054
    # current eps: 6.94 -> 2 clusters found; f1-score: 0.0055
    # current eps: 6.95 -> 2 clusters found; f1-score: 0.0056
    # current eps: 6.96 -> 2 clusters found; f1-score: 0.0056
    # current eps: 6.97 -> 2 clusters found; f1-score: 0.0056
    # current eps: 6.98 -> 2 clusters found; f1-score: 0.0056
    # current eps: 6.99 -> 2 clusters found; f1-score: 0.0057
    # current eps: 7.00 -> 2 clusters found; f1-score: 0.0057
    # current eps: 7.01 -> 2 clusters found; f1-score: 0.0058
    # current eps: 7.02 -> 2 clusters found; f1-score: 0.0058
    # current eps: 7.03 -> 2 clusters found; f1-score: 0.0058
    # current eps: 7.04 -> 2 clusters found; f1-score: 0.0058
    # current eps: 7.05 -> 2 clusters found; f1-score: 0.0059
    # current eps: 7.06 -> 2 clusters found; f1-score: 0.0059
    # current eps: 7.07 -> 2 clusters found; f1-score: 0.0059
    # current eps: 7.08 -> 2 clusters found; f1-score: 0.0059
    # current eps: 7.09 -> 2 clusters found; f1-score: 0.0059
    # current eps: 7.10 -> 2 clusters found; f1-score: 0.0059
    # current eps: 7.11 -> 2 clusters found; f1-score: 0.0060
    # current eps: 7.12 -> 2 clusters found; f1-score: 0.0060
    # current eps: 7.13 -> 2 clusters found; f1-score: 0.0061
    # current eps: 7.14 -> 2 clusters found; f1-score: 0.0061
    # current eps: 7.15 -> 2 clusters found; f1-score: 0.0061
    # current eps: 7.16 -> 2 clusters found; f1-score: 0.0061
    # current eps: 7.17 -> 2 clusters found; f1-score: 0.0061
    # current eps: 7.18 -> 2 clusters found; f1-score: 0.0062
    # current eps: 7.19 -> 2 clusters found; f1-score: 0.0062
    # current eps: 7.20 -> 2 clusters found; f1-score: 0.0063
    # current eps: 7.21 -> 2 clusters found; f1-score: 0.0063
    # current eps: 7.22 -> 2 clusters found; f1-score: 0.0063
    # current eps: 7.23 -> 2 clusters found; f1-score: 0.0063
    # current eps: 7.24 -> 2 clusters found; f1-score: 0.0063
    RGCN = 'RGCN'

    RotatE = 'RotatE'  # ValueError: Complex data not supported
    SimplE = 'SimplE'  # No clusters found on Training74

    # Training74 results:
    # current eps: 0.94 -> 4 clusters found; f1-score: 0.0197
    # current eps: 0.95 -> 9 clusters found; f1-score: 0.0197
    # current eps: 0.96 -> 45 clusters found; f1-score: 0.0196
    # current eps: 0.97 -> 172 clusters found; f1-score: 0.0199
    # current eps: 0.98 -> 425 clusters found; f1-score: 0.0198
    # current eps: 0.99 -> 812 clusters found; f1-score: 0.0209
    # current eps: 1.00 -> 626 clusters found; f1-score: 0.0227  <- best
    # current eps: 1.01 -> 138 clusters found; f1-score: 0.0199
    # current eps: 1.02 -> 10 clusters found; f1-score: 0.0216
    # current eps: 1.03 -> 1 clusters found; f1-score: 0.0199
    SE = 'SE'

    TorusE = 'TorusE'  # No clusters found on Training74

    # Training74 results:
    # current eps: 0.62 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.63 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.64 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.65 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.66 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.67 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.68 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.69 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.70 -> 2 clusters found; f1-score: 0.0197
    # current eps: 0.71 -> 3 clusters found; f1-score: 0.0197
    # current eps: 0.72 -> 3 clusters found; f1-score: 0.0197
    # current eps: 0.73 -> 5 clusters found; f1-score: 0.0197
    # current eps: 0.74 -> 8 clusters found; f1-score: 0.0197
    # current eps: 0.75 -> 9 clusters found; f1-score: 0.0198
    # current eps: 0.76 -> 10 clusters found; f1-score: 0.0197
    # current eps: 0.77 -> 7 clusters found; f1-score: 0.0198
    # current eps: 0.78 -> 6 clusters found; f1-score: 0.0199  <- best
    # current eps: 0.79 -> 3 clusters found; f1-score: 0.0198
    # current eps: 0.80 -> 7 clusters found; f1-score: 0.0198
    # current eps: 0.81 -> 6 clusters found; f1-score: 0.0196
    # current eps: 0.82 -> 4 clusters found; f1-score: 0.0198
    # current eps: 0.83 -> 2 clusters found; f1-score: 0.0195
    # current eps: 0.84 -> 1 clusters found; f1-score: 0.0183
    # current eps: 0.85 -> 2 clusters found; f1-score: 0.0174
    # current eps: 0.86 -> 1 clusters found; f1-score: 0.0173
    # current eps: 0.87 -> 2 clusters found; f1-score: 0.0164
    # current eps: 0.88 -> 2 clusters found; f1-score: 0.0159
    # current eps: 0.89 -> 1 clusters found; f1-score: 0.0165
    # current eps: 0.90 -> 1 clusters found; f1-score: 0.0146
    # current eps: 0.91 -> 2 clusters found; f1-score: 0.0138
    # current eps: 0.92 -> 3 clusters found; f1-score: 0.0132
    # current eps: 0.93 -> 3 clusters found; f1-score: 0.0122
    # current eps: 0.94 -> 3 clusters found; f1-score: 0.0085
    # current eps: 0.95 -> 4 clusters found; f1-score: 0.0068
    # current eps: 0.96 -> 3 clusters found; f1-score: 0.0077
    # current eps: 0.97 -> 2 clusters found; f1-score: 0.0107
    # current eps: 0.98 -> 1 clusters found; f1-score: 0.0039
    TransD = 'TransD'

    # Training74 results:
    # current eps: 0.93 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.94 -> 8 clusters found; f1-score: 0.0197
    # current eps: 0.95 -> 21 clusters found; f1-score: 0.0198
    # current eps: 0.96 -> 54 clusters found; f1-score: 0.0195
    # current eps: 0.97 -> 154 clusters found; f1-score: 0.0197
    # current eps: 0.98 -> 421 clusters found; f1-score: 0.0185
    # current eps: 0.99 -> 818 clusters found; f1-score: 0.0173
    # current eps: 1.00 -> 654 clusters found; f1-score: 0.0159
    # current eps: 1.01 -> 165 clusters found; f1-score: 0.0209
    # current eps: 1.02 -> 11 clusters found; f1-score: 0.0285  <- best
    # current eps: 1.03 -> 1 clusters found; f1-score: 0.0096
    TransE = 'TransE'

    TransF = 'TransF'  # No clusters found on Training74

    # Training74 results:
    # current eps: 0.03 -> 10 clusters found; f1-score: 0.0196  <- best
    # current eps: 0.04 -> 11 clusters found; f1-score: 0.0189
    # current eps: 0.05 -> 11 clusters found; f1-score: 0.0186
    # current eps: 0.06 -> 9 clusters found; f1-score: 0.0169
    # current eps: 0.07 -> 13 clusters found; f1-score: 0.0163
    # current eps: 0.08 -> 16 clusters found; f1-score: 0.0132
    # current eps: 0.09 -> 16 clusters found; f1-score: 0.0126
    # current eps: 0.10 -> 14 clusters found; f1-score: 0.0094
    # current eps: 0.11 -> 16 clusters found; f1-score: 0.0064
    # current eps: 0.12 -> 18 clusters found; f1-score: 0.0056
    # current eps: 0.13 -> 16 clusters found; f1-score: 0.0030
    # current eps: 0.14 -> 14 clusters found; f1-score: 0.0024
    TransH = 'TransH'

    # Training74 results:
    # current eps: 0.81 -> 1 clusters found; f1-score: 0.0196
    # current eps: 0.82 -> 1 clusters found; f1-score: 0.0196
    # current eps: 0.83 -> 1 clusters found; f1-score: 0.0196
    # current eps: 0.84 -> 1 clusters found; f1-score: 0.0196
    # current eps: 0.85 -> 1 clusters found; f1-score: 0.0196
    # current eps: 0.86 -> 1 clusters found; f1-score: 0.0196
    # current eps: 0.87 -> 4 clusters found; f1-score: 0.0197
    # current eps: 0.88 -> 3 clusters found; f1-score: 0.0197
    # current eps: 0.89 -> 6 clusters found; f1-score: 0.0197
    # current eps: 0.90 -> 9 clusters found; f1-score: 0.0197
    # current eps: 0.91 -> 15 clusters found; f1-score: 0.0196
    # current eps: 0.92 -> 30 clusters found; f1-score: 0.0197
    # current eps: 0.93 -> 68 clusters found; f1-score: 0.0201
    # current eps: 0.94 -> 139 clusters found; f1-score: 0.0200
    # current eps: 0.95 -> 278 clusters found; f1-score: 0.0202  <- best
    # current eps: 0.96 -> 419 clusters found; f1-score: 0.0197
    # current eps: 0.97 -> 426 clusters found; f1-score: 0.0193
    # current eps: 0.98 -> 232 clusters found; f1-score: 0.0187
    # current eps: 0.99 -> 65 clusters found; f1-score: 0.0145
    # current eps: 1.00 -> 14 clusters found; f1-score: 0.0149
    # current eps: 1.01 -> 1 clusters found; f1-score: 0.0082
    TransR = 'TransR'

    # Training74 results:
    # current eps: 1.06 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.07 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.08 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.09 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.10 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.11 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.12 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.13 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.14 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.15 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.16 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.17 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.18 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.19 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.20 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.21 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.22 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.23 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.24 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.25 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.26 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.27 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.28 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.29 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.30 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.31 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.32 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.33 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.34 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.35 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.36 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.37 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.38 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.39 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.40 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.41 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.42 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.43 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.44 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.45 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.46 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.47 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.48 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.49 -> 1 clusters found; f1-score: 0.0196
    # current eps: 1.50 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.51 -> 1 clusters found; f1-score: 0.0194
    # current eps: 1.52 -> 1 clusters found; f1-score: 0.0195
    # current eps: 1.53 -> 1 clusters found; f1-score: 0.0196
    # current eps: 1.54 -> 1 clusters found; f1-score: 0.0196
    # current eps: 1.55 -> 1 clusters found; f1-score: 0.0195
    # current eps: 1.56 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.57 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.58 -> 1 clusters found; f1-score: 0.0200
    # current eps: 1.59 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.60 -> 1 clusters found; f1-score: 0.0204
    # current eps: 1.61 -> 1 clusters found; f1-score: 0.0204
    # current eps: 1.62 -> 2 clusters found; f1-score: 0.0207
    # current eps: 1.63 -> 2 clusters found; f1-score: 0.0213
    # current eps: 1.64 -> 1 clusters found; f1-score: 0.0210
    # current eps: 1.65 -> 1 clusters found; f1-score: 0.0223
    # current eps: 1.66 -> 1 clusters found; f1-score: 0.0227
    # current eps: 1.67 -> 1 clusters found; f1-score: 0.0234
    # current eps: 1.68 -> 1 clusters found; f1-score: 0.0217
    # current eps: 1.69 -> 1 clusters found; f1-score: 0.0228
    # current eps: 1.70 -> 1 clusters found; f1-score: 0.0247
    # current eps: 1.71 -> 1 clusters found; f1-score: 0.0239
    # current eps: 1.72 -> 1 clusters found; f1-score: 0.0206
    # current eps: 1.73 -> 1 clusters found; f1-score: 0.0204
    # current eps: 1.74 -> 1 clusters found; f1-score: 0.0205
    # current eps: 1.75 -> 1 clusters found; f1-score: 0.0220
    # current eps: 1.76 -> 1 clusters found; f1-score: 0.0238
    # current eps: 1.77 -> 1 clusters found; f1-score: 0.0252
    # current eps: 1.78 -> 1 clusters found; f1-score: 0.0256
    # current eps: 1.79 -> 1 clusters found; f1-score: 0.0235
    # current eps: 1.80 -> 1 clusters found; f1-score: 0.0274
    # current eps: 1.81 -> 1 clusters found; f1-score: 0.0283  <- best
    # current eps: 1.82 -> 1 clusters found; f1-score: 0.0153
    # current eps: 1.83 -> 1 clusters found; f1-score: 0.0067
    # current eps: 1.84 -> 1 clusters found; f1-score: 0.0083
    TuckER = 'TuckER'

    # Training74 results:
    # current eps: 0.59 -> 0 clusters found; f1-score: 0.0196
    # current eps: 0.60 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.61 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.62 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.63 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.64 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.65 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.66 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.67 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.68 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.69 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.70 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.71 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.72 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.73 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.74 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.75 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.76 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.77 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.78 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.79 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.80 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.81 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.82 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.83 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.84 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.85 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.86 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.87 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.88 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.89 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.90 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.91 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.92 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.93 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.94 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.95 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.96 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.97 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.98 -> 1 clusters found; f1-score: 0.0197
    # current eps: 0.99 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.00 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.01 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.02 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.03 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.04 -> 1 clusters found; f1-score: 0.0199
    # current eps: 1.05 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.06 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.07 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.08 -> 1 clusters found; f1-score: 0.0199
    # current eps: 1.09 -> 1 clusters found; f1-score: 0.0200  <- best
    # current eps: 1.10 -> 1 clusters found; f1-score: 0.0199
    # current eps: 1.11 -> 1 clusters found; f1-score: 0.0201
    # current eps: 1.12 -> 1 clusters found; f1-score: 0.0199
    # current eps: 1.13 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.14 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.15 -> 1 clusters found; f1-score: 0.0197
    # current eps: 1.16 -> 1 clusters found; f1-score: 0.0199
    # current eps: 1.17 -> 1 clusters found; f1-score: 0.0196
    # current eps: 1.18 -> 1 clusters found; f1-score: 0.0195
    # current eps: 1.19 -> 1 clusters found; f1-score: 0.0193
    # current eps: 1.20 -> 1 clusters found; f1-score: 0.0191
    # current eps: 1.21 -> 1 clusters found; f1-score: 0.0194
    # current eps: 1.22 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.23 -> 1 clusters found; f1-score: 0.0198
    # current eps: 1.24 -> 1 clusters found; f1-score: 0.0189
    # current eps: 1.25 -> 1 clusters found; f1-score: 0.0183
    # current eps: 1.26 -> 1 clusters found; f1-score: 0.0174
    # current eps: 1.27 -> 1 clusters found; f1-score: 0.0161
    # current eps: 1.28 -> 1 clusters found; f1-score: 0.0145
    # current eps: 1.29 -> 1 clusters found; f1-score: 0.0120
    # current eps: 1.30 -> 1 clusters found; f1-score: 0.0120
    # current eps: 1.31 -> 1 clusters found; f1-score: 0.0117
    # current eps: 1.32 -> 1 clusters found; f1-score: 0.0113
    # current eps: 1.33 -> 1 clusters found; f1-score: 0.0105
    # current eps: 1.34 -> 1 clusters found; f1-score: 0.0089
    # current eps: 1.35 -> 1 clusters found; f1-score: 0.0093
    # current eps: 1.36 -> 1 clusters found; f1-score: 0.0099
    # current eps: 1.37 -> 1 clusters found; f1-score: 0.0105
    # current eps: 1.38 -> 1 clusters found; f1-score: 0.0078
    # current eps: 1.39 -> 1 clusters found; f1-score: 0.0080
    # current eps: 1.40 -> 1 clusters found; f1-score: 0.0064
    # current eps: 1.41 -> 1 clusters found; f1-score: 0.0052
    # current eps: 1.42 -> 1 clusters found; f1-score: 0.0036
    # current eps: 1.43 -> 1 clusters found; f1-score: 0.0014
    # current eps: 1.44 -> 1 clusters found; f1-score: 0.0016
    UM = 'UM'


class OutlierDetector:
    def __init__(
            self,
            input_file_path: str,
            dataset_name: str,
            rdf_graph: Graph,
            embedding_method: str,
            eps: float = 0.5
    ):

        self.input_file_path = input_file_path
        self.rdf_graph = rdf_graph
        self.triples_factory = TriplesFactory.from_path(input_file_path)
        self.dataset = Dataset.from_path(input_file_path)

        # reverse the entity-to-index mapping to get an index-to-entity mapping
        self.id_to_entity = {v: k for k, v in self.dataset.entity_to_id.items()}

        self.eps = eps

        try:
            self.model = cache.load(dataset_name, embedding_method)

        except CacheMiss:
            embedding_results = pipeline(
                dataset=self.dataset,
                model=embedding_method
            )

            cache.store(dataset_name, embedding_method, embedding_results)
            self.model = embedding_results.model

    def remove_outliers(self) -> Graph:
        outlier_entities = []
        if hasattr(self.model, 'entity_representations'):
            # -> https://pykeen.readthedocs.io/en/latest/tutorial/first_steps.html#using-learned-embeddings :
            #
            # ``Knowledge graph embedding models can potentially have multiple
            #   entity representations and multiple relation representations, so
            #   they are respectively stored as sequences in the
            #   entity_representations and relation_representations attributes
            #   of each model. While the exact contents of these sequences are
            #   model-dependent, the first element of each is usually the
            #   "primary" representation for either the entities or relations.
            #
            #   Typically, the values in these sequences are instances of the
            #   `pykeen.nn.representation.Embedding`. This implements a similar,
            #   but more powerful, interface to the built-in
            #   `torch.nn.Embedding` class. However, the values in these
            #   sequences can more generally be instances of any subclasses of
            #   `pykeen.nn.representation.Representation`. This allows for more
            #   powerful encoders those in GNNs such as `pykeen.models.RGCN` to
            #   be implemented and used.''
            main_embedding_index = 0
            entity_embedding_tensor = \
                self.model.entity_representations[main_embedding_index]()

            numpy_embeddings = entity_embedding_tensor.detach().numpy()

            dbscan = DBSCAN(eps=self.eps)
            dbscan.fit(numpy_embeddings)
            # evaluator = LUMBEvaluator()
            # evaluator.evaluate_clusters(self.dataset, dbscan)
            # print(f'current eps: {eps} -> {num_clusters} clusters found; f1-score: {evaluator.get_f1_score()}')

            # get all indexes for those nodes not belonging to any cluster
            indexes = np.where(dbscan.labels_ == -1)

            # build URIs for the nodes found in the '-1' cluster, i.e., the
            # outlier nodes
            outliers: List[URIRef] = \
                [
                    URIRef(self.id_to_entity[idx])
                    for idx in indexes[0]
                    if self.id_to_entity[idx].startswith('http://')
                ]

            cleaned_g = Graph()

            for s, p, o in self.rdf_graph:
                if s in outliers or o in outliers:
                    continue
                cleaned_g.add((s, p, o))

            return cleaned_g

        else:
            return self.rdf_graph


class PyKEENAdapter(SHACLGenerator):
    def __init__(
            self,
            input_file_path: str,
            embedding_method: EmbeddingMethod,
            eps: float = 0.5
    ):
        g = Graph()
        g.parse(input_file_path)

        tmp_file_path = tempfile.mktemp()
        nt_to_tsv(input_file_path, tmp_file_path)

        dataset_name = input_file_path.split(os.path.sep)[-1]

        self.outlier_detector = OutlierDetector(
            input_file_path=tmp_file_path,
            dataset_name=dataset_name,
            rdf_graph=g,
            embedding_method=embedding_method.value,
            eps=eps
        )

        os.remove(tmp_file_path)

    def generate_shacl(self) -> Graph:
        g_wo_outliers = self.outlier_detector.remove_outliers()

        # Using Shaclgen under the hood for now
        shaclgen = data_graph(g_wo_outliers)
        return shaclgen.gen_graph()


# if __name__ == '__main__':
#     input_file_path = '../data/Training74/mergedGraph257.nt'
#
#     shacl_generator = PyKEENAdapter(input_file_path, EmbeddingMethod.RGCN, eps=3.45)
#     res: Graph = shacl_generator.generate_shacl()
#     res.serialize('/tmp/res.nt', format='ntriples')

# fig, ax = plt.subplots()
# tp_plot, = ax.plot(epss, tps, ':', label='TP')
# fp_plot, = ax.plot(epss, fps, ':', label='FP')
# tn_plot, = ax.plot(epss, tns, ':', label='TN')
# fn_plot, = ax.plot(epss, fns, ':', label='FN')
# legend1 = ax.legend(handles=[tp_plot, fp_plot, tn_plot, fn_plot], loc='upper right')
# ax.add_artist(legend1)
#
# ax2 = ax.twinx()
# acc_plot, = ax2.plot(epss, accuracies, label='Accuracy')
# f1_score_plot, = ax2.plot(epss, f1_scores, label='F1-Score')
# ax2.legend(handles=[acc_plot, f1_score_plot], loc='lower right')
#
# plt.show()
