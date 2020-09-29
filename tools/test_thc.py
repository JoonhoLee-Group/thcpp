#!/usr/bin/env python

import numpy
from thc.transform_basis import dump_orbitals_kpoints
from thc.atomic_orbitals import AOTHC

isdf = AOTHC(scf_dump='scf.kpoint.2x2x2.ngs.20.dump', ortho=True)
isdf.single(10)
