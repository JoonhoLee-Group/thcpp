#!/usr/bin/env python

from thc.transform_basis import compute_thc_hf_energy, compute_thc_hf_energy_wfn
import sys

if sys.argv[3] == "True":
    print ("Computing HF energy in orthogonalised AO basis.")
    ehf, ehf_ref = compute_thc_hf_energy_wfn(sys.argv[1], sys.argv[2])
else:
    print ("Computing HF energy in non-orthogonal AO basis.")
    ehf, ehf_ref, fock = compute_thc_hf_energy(sys.argv[1], sys.argv[2])
print (ehf, ehf_ref)
