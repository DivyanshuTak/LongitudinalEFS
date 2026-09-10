"""Fixed encoding of the clinical covariates into a 13-d vector. Vocabularies and
age normalisation are constants, not fit per split, so the encoding is identical
across cohorts.

Layout:
  0     age at diagnosis, z-scored
  1     age missing
  2     sex (male=1, female=0)
  3-5   resection: gross/near total / subtotal|partial / biopsy
  6     resection missing
  7-11  location: posterior fossa|brainstem / cerebral hemisphere / midline|deep
        / optic pathway / other
  12    location unknown
"""
from typing import List

import pandas as pd

# CBTN records gross/near total as one value, so BCH's are merged to match
RESECTION = ("gross/near total", "subtotal/partial", "biopsy")
LOCATION = ("posterior fossa/brainstem", "cerebral hemisphere",
            "midline/deep", "optic pathway", "other")

CLINICAL_COLS = ("age_dx", "sex", "resection", "tumor_location")
CLINICAL_DIM = 2 + 1 + len(RESECTION) + 1 + len(LOCATION) + 1

# BCH training-cohort age statistics (years)
AGE_MEAN, AGE_SD = 8.8, 4.5


def encode_row(row) -> List[float]:
    v = [0.0] * CLINICAL_DIM
    age = pd.to_numeric(row.get("age_dx"), errors="coerce")
    if pd.isna(age):
        v[1] = 1.0
    else:
        v[0] = (float(age) - AGE_MEAN) / AGE_SD

    sex = str(row.get("sex", "")).strip().lower()
    v[2] = 1.0 if sex.startswith("m") else 0.0

    res = str(row.get("resection", "")).strip().lower()
    v[6] = 1.0 if res not in RESECTION else 0.0
    if res in RESECTION:
        v[3 + RESECTION.index(res)] = 1.0

    loc = str(row.get("tumor_location", "")).strip().lower()
    v[12] = 1.0 if loc not in LOCATION else 0.0
    if loc in LOCATION:
        v[7 + LOCATION.index(loc)] = 1.0
    return v
