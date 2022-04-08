#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

'''
Convert the k-sampled MO/integrals to corresponding Gamma-point supercell
MO/integrals.

Zhihao Cui <zcui@caltech.edu>
Qiming Sun <osirpt.sun@gmail.com>

Slightly modified to stop failures for complex orbitals / assertion erros for
metals.

See also the original implementation at
https://github.com/zhcui/local-orbital-and-cdft/blob/master/k2gamma.py
'''

from functools import reduce
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.pbc import tools


def kpts_to_kmesh(cell, kpts):
    '''Guess kmesh'''
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    kmesh = (len(np.unique(scaled_k[:,0])),
             len(np.unique(scaled_k[:,1])),
             len(np.unique(scaled_k[:,2])))
    return kmesh

def translation_vectors_for_kmesh(cell, kmesh):
    '''
    Translation vectors to construct super-cell of which the gamma point is
    identical to the k-point mesh of primitive cell
    '''
    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0])
    R_rel_b = np.arange(kmesh[1])
    R_rel_c = np.arange(kmesh[2])
    R_vec_rel = lib.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_vec_abs = np.einsum('nu, uv -> nv', R_vec_rel, latt_vec)
    return R_vec_abs

def get_phase(cell, kpts, kmesh=None):
    '''
    The unitary transformation that transforms the supercell basis k-mesh
    adapted basis.
    '''
    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts)
    R_vec_abs = translation_vectors_for_kmesh(cell, kmesh)

    NR = len(R_vec_abs)
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk', R_vec_abs, kpts))
    phase /= np.sqrt(NR)  # normalization in supercell

    # R_rel_mesh has to be construct exactly same to the Ts in super_cell function
    scell = tools.super_cell(cell, kmesh)
    return scell, phase

def double_translation_indices(kmesh):
    '''Indices to utilize the translation symmetry in the 2D matrix.

    D[M,N] = D[N-M]

    The return index maps the 2D subscripts to 1D subscripts.

    D2 = D1[double_translation_indices()]

    D1 holds all the symmetry unique elements in D2
    '''

    tx = translation_map(kmesh[0])
    ty = translation_map(kmesh[1])
    tz = translation_map(kmesh[2])
    idx = np.ravel_multi_index([tx[:,None,None,:,None,None],
                                ty[None,:,None,None,:,None],
                                tz[None,None,:,None,None,:]], kmesh)
    nk = np.prod(kmesh)
    return idx.reshape(nk, nk)

def translation_map(nk):
    ''' Generate
    [0    1 .. n  ]
    [n    0 .. n-1]
    [n-1  n .. n-2]
    [...  ...  ...]
    [1    2 .. 0  ]
    '''
    idx = np.repeat(np.arange(nk)[None,:], nk-1, axis=0)
    strides = idx.strides
    t_map = np.ndarray((nk, nk), strides=(strides[0]-strides[1], strides[1]),
                       dtype=int, buffer=np.append(idx.ravel(), 0))
    return t_map

def mo_k2gamma(cell, mo_energy, mo_coeff, kpts, kmesh=None, make_real=True):
    scell, phase = get_phase(cell, kpts, kmesh)

    E_g = np.hstack(mo_energy)
    C_k = np.asarray(mo_coeff)
    Nk, Nao, Nmo = C_k.shape
    NR = phase.shape[0]

    # Transform AO indices
    C_gamma = np.einsum('Rk, kum -> Rukm', phase, C_k)
    C_gamma = C_gamma.reshape(Nao*NR, Nk*Nmo)

    E_sort_idx = np.argsort(E_g)
    E_g = E_g[E_sort_idx]
    C_gamma = C_gamma[:,E_sort_idx]
    s = scell.pbc_intor('int1e_ovlp')
    try:
        delta_ovlp = abs(reduce(np.dot, (C_gamma.conj().T, s, C_gamma)) - np.eye(Nmo*Nk)).max()
        assert delta_ovlp < 1e-5
    except AssertionError:
        print("max(C^ S C - 1) = {}".format(delta_ovlp))

    # For degenerated MOs, the transformed orbitals in super cell may not be
    # real. Construct a sub Fock matrix in super-cell to find a proper
    # transformation that makes the transformed MOs real.
    E_k_degen = abs(E_g[1:] - E_g[:-1]) < 1e-3
    degen_mask = np.append(False, E_k_degen) | np.append(E_k_degen, False)
    if make_real:
        if np.any(E_k_degen):
            if abs(C_gamma[:,~degen_mask].imag).max() < 1e-4:
                shift = min(E_g[degen_mask]) - .1
                f = np.dot(C_gamma[:,degen_mask] * (E_g[degen_mask] - shift),
                           C_gamma[:,degen_mask].conj().T)
                assert(abs(f.imag).max() < 1e-4)
                e, na_orb = scipy.linalg.eigh(f.real, s, type=2)
                C_gamma = C_gamma.real
                C_gamma[:,degen_mask] = na_orb[:, e>1e-7]
            else:
                f = np.dot(C_gamma * E_g, C_gamma.conj().T)
                assert(abs(f.imag).max() < 1e-4)
                e, C_gamma = scipy.linalg.eigh(f.real, s, type=2)

    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    # overlap between k-point unitcell and gamma-point supercell
    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase.conj()).reshape(Nk,Nao,NR*Nao)
    # The unitary transformation from k-adapted orbitals to gamma-point orbitals
    mo_phase = lib.einsum('kum,kuv,vi->kmi', C_k.conj(), s_k_g, C_gamma)

    return scell, E_g, C_gamma, mo_phase

def k2gamma(kmf, kmesh=None, make_real=True):
    r'''
    convert the k-sampled mean-field object to the corresponding supercell
    gamma-point mean-field object.

    math:
         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \frac{1}{\sqrt{N_{\UC}}}
         \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}
    '''
    from pyscf.pbc import scf
    def transform(mo_energy, mo_coeff, mo_occ):
        scell, E_g, C_gamma = mo_k2gamma(kmf.cell, mo_energy, mo_coeff,
                                         kmf.kpts, kmesh,
                                         make_real=make_real)[:3]
        E_sort_idx = np.argsort(np.hstack(mo_energy))
        mo_occ = np.hstack(mo_occ)[E_sort_idx]
        return scell, E_g, C_gamma, mo_occ

    if isinstance(kmf, scf.khf.KRHF):
        scell, E_g, C_gamma, mo_occ = transform(kmf.mo_energy, kmf.mo_coeff, kmf.mo_occ)
        mf = scf.RHF(scell)
    elif isinstance(kmf, scf.kuhf.KUHF):
        scell, Ea, Ca, occ_a = transform(kmf.mo_energy[0], kmf.mo_coeff[0], kmf.mo_occ[0])
        scell, Eb, Cb, occ_b = transform(kmf.mo_energy[1], kmf.mo_coeff[1], kmf.mo_occ[1])
        mf = scf.UHF(scell)
        E_g = [Ea, Eb]
        C_gamma = [Ca, Cb]
        mo_occ = [occ_a, occ_b]
    else:
        raise NotImplementedError('SCF object %s not supported' % kmf)

    mf.mo_coeff = C_gamma
    mf.mo_energy = E_g
    mf.mo_occ = mo_occ
    return mf


def to_supercell_ao_integrals(cell, kpts, ao_ints):
    '''Transform from the unitcell k-point AO integrals to the supercell
    gamma-point AO integrals.
    '''
    scell, phase = get_phase(cell, kpts)
    NR, Nk = phase.shape
    nao = cell.nao
    scell_ints = np.einsum('Rk,kij,Sk->RiSj', phase, ao_ints, phase.conj())
    return scell_ints.reshape(NR*nao,NR*nao).real


def to_supercell_mo_integrals(kmf, mo_ints):
    '''Transform from the unitcell k-point MO integrals to the supercell
    gamma-point MO integrals.
    '''
    cell = kmf.cell
    kpts = kmf.kpts

    mo_k = np.array(kmf.mo_coeff)
    Nk, nao, nmo = mo_k.shape
    e_k = np.array(kmf.mo_energy)
    scell, E_g, C_gamma, mo_phase = mo_k2gamma(cell, e_k, mo_k, kpts)

    scell_ints = lib.einsum('xui,xuv,xvj->ij', mo_phase.conj(), mo_ints, mo_phase)
    assert(abs(scell_ints.imag).max() < 1e-7)
    return scell_ints.real


if __name__ == '__main__':
    from pyscf.pbc import gto, dft
    cell = gto.Cell()
    cell.atom = '''
    H 0.  0.  0.
    H 0.5 0.3 0.4
    '''

    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 4.
    cell.unit='B'
    cell.build()

    kmesh = [2, 2, 1]
    kpts = cell.make_kpts(kmesh)

    print("Transform k-point integrals to supercell integral")
    scell, phase = get_phase(cell, kpts)
    NR, Nk = phase.shape
    nao = cell.nao
    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    s = scell.pbc_intor('int1e_ovlp')
    s1 = np.einsum('Rk,kij,Sk->RiSj', phase, s_k, phase.conj())
    print(abs(s-s1.reshape(s.shape)).max())

    s = scell.pbc_intor('int1e_ovlp').reshape(NR,nao,NR,nao)
    s1 = np.einsum('Rk,RiSj,Sk->kij', phase.conj(), s, phase)
    print(abs(s1-s_k).max())

    kmf = dft.KRKS(cell, kpts)
    ekpt = kmf.run()

    mf = k2gamma(kmf, kmesh)
    c_g_ao = mf.mo_coeff

    # The following is to check whether the MO is correctly coverted:

    print("Supercell gamma MO in AO basis from conversion:")
    scell = tools.super_cell(cell, kmesh)
    mf_sc = dft.RKS(scell)

    s = mf_sc.get_ovlp()
    mf_sc.run()
    sc_mo = mf_sc.mo_coeff

    nocc = scell.nelectron // 2
    print("Supercell gamma MO from direct calculation:")
    print(np.linalg.det(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc])))
    print(np.linalg.svd(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc]))[1])

