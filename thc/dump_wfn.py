#!/usr/bin/env python

from thc.transform_basis import dump_thc_data, dump_thc_data_sc
import sys

if sys.argv[2] == 'ortho':
    ortho = True
    mos =  False
elif sys.argv[2] == 'mos':
    mos = True
    ortho = False
else:
    ortho = False
    mos = False
if sys.argv[3] == 'supercell':
    dump_thc_data_sc(sys.argv[1], ortho_ao=ortho,
                     orbital_file="aos.h5", wfn_file="wfn_thc.dat")
else:
    dump_thc_data(sys.argv[1], mos=mos, ortho_ao=ortho, orbital_file="aos.h5",
                  wfn_file="wfn_thc.dat")
