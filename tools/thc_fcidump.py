#!/usr/bin/env python

import h5py
import numpy
import sys
from pyscf.tools.fcidump import write_head, write_hcore, from_integrals
from pauxy.utils.io import dump_qmcpack_trial_wfn


def get_data(filename):
    with h5py.File(filename, 'r') as fh5:
        Luv = fh5['Hamiltonian/THC/Luv'][:].view(numpy.complex128)
        nbasis = Luv.shape[0]
        Luv = Luv.reshape(nbasis,nbasis)
        Muv = numpy.dot(Luv, Luv.conj().T)
        P = fh5['Hamiltonian/THC/Orbitals'][:].view(numpy.complex128)
        P = P.reshape(P.shape[0], P.shape[1])
        hcore = fh5['Hamiltonian/hcore'][:].view(numpy.complex128)
        hcore = hcore.reshape(hcore.shape[0], hcore.shape[1])
        enuc = fh5['Hamiltonian/Energies'][:][0]
        dims = fh5['Hamiltonian/dims'][:]
    return (Muv, Luv, P, hcore, enuc, dims)

def reconstruct_eri_simple(filename):
    Muv, Luv, P, hcore, enuc, dims = get_data(filename)
    T1 = numpy.einsum('im,km,mn->ikn', P.conj(), P, Muv)
    nb = hcore.shape[0]
    eri = numpy.einsum('ikn,jn,ln->ikjl', T1, P.conj(), P)
    from_integrals('FCIDUMP', hcore.real, eri.reshape(nb*nb,nb*nb).real, hcore.shape[0],
                   (dims[4], dims[5]), nuc=enuc, tol=1e-6)
    dump_qmcpack_trial_wfn(numpy.eye(hcore.shape[0]), 4)

def reconstruct_eri_opt(filename, nactive, out='FCIDUMP'):
    Muv, Luv, P, hcore, enuc, dims = get_data(filename)
    nel = dims[4] + dims[5]
    chol = numpy.einsum('im,km,ma->ika', P[:nactive,:].conj(), P[:nactive,:], Luv)
    with open(out, 'w') as fout:
        write_head(fout, nactive, nel, 0, [])
        ik = 0
        for i in range(0,nactive):
            for k in range(0,i+1):
                jl = 0
                for j in range(0,nactive):
                    for l in range(0,j+1):
                        eri_ikjl =  numpy.dot(chol[i,k], chol[l,j].conj())
                        if abs(eri_ikjl) > 1e-8:
                            fout.write('(%13.8e,%13.8e) %d %d %d %d\n'%(eri_ikjl.real,
                                            eri_ikjl.imag, i+1, k+1, j+1, l+1))
                        jl += 1
                ik += 1

        for i in range(nactive):
            for k in range(0, i+1):
                if abs(hcore[i,k]) > 1e-8:
                    fout.write('(%13.8e,%13.8e) %d %d 0 0\n'%(hcore[i,k].real,
                        hcore[i,k].imag, i+1, k+1))
        fout.write('(%13.8e,%13.8e) 0 0 0 0'%(enuc.real, enuc.imag))

def main(filename, nactive):
    reconstruct_eri_opt(filename, nactive)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
