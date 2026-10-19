"""
Eight-band k.p model (conduction band + three valence bands, both spins)
for wurtzite (c-axis growth) and zincblende ([001] growth) heterostructures.

    parameters   per-node k.p parameters from the material model
    hamiltonian  the bulk 8x8 Hamiltonian H(kx, ky, kz) of one node
    solver       the 1D (growth-direction) discretisation, eigenstates at
                 each in-plane wavevector and the carrier densities
"""
