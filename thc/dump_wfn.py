#!/usr/bin/env python

from thc.transform_basis import dump_thc_data
import sys

if sys.argv[2] == 'True':
    ortho = True
else:
    ortho = False
dump_thc_data(sys.argv[1], ortho_ao=ortho, ao_file="aos.h5")
