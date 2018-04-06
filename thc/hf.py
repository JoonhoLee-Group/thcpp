#!/usr/bin/env python

from thc.transform_basis import compute_thc_hf_energy
import sys

ehf, ehf_ref, fock = compute_thc_hf_energy(sys.argv[1], sys.argv[2])
print (ehf, ehf_ref)
