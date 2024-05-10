import pennylane as qml
from pennylane import numpy as np


def find_neighbors(lattice_size,periodic=False):
    """
    takes in a lattice size and returns a dictionary whose keys are sites
    and whose values are all adjacent neighboring sites
    """
    all_neighbors = {i:set() for i in range(lattice_size**2)}
    is_a_valid_site = lambda x,y: x>=0 and x<=lattice_size-1 and y>=0 and y<=lattice_size-1 #makes sure a site is in the square lattice


    for i in range(lattice_size):
        for j in range(lattice_size):

            site = i*lattice_size + j
            neighbors = [[i+1,j],[i-1,j],[i,j+1],[i,j-1]]

            for x,y in neighbors: 
                if is_a_valid_site(x,y): #make sure neighbors are in the lattice
                    neighbor_site = x*lattice_size + y
                    if neighbor_site > site:
                         all_neighbors[site].add(neighbor_site)
            if periodic: #for periodic boundary conditions
                if i == 0:
                    neighbor_site = lattice_size**2-lattice_size+site
                    all_neighbors[site].add(neighbor_site)
                if j == 0:
                    neighbor_site = site+lattice_size-1
                    all_neighbors[site].add(neighbor_site)
    return all_neighbors

def construct_hamiltonian(J,b,nb_rows,nb_columns,periodic,neighbors):
    """
    Given the coupling strength term and the magnetic field strength
    returns the 2-D transverse field ising model hamiltonian for a 
    square lattice
    """
    #neighbors = find_neighbors(lattice_size,periodic)
    hamiltonian_terms = []
    coeffs = []

    for i in range(nb_rows*nb_columns):
        coeffs.append(-b) 
        hamiltonian_terms.append(qml.PauliX(i))
        for j in neighbors[i]:

            coeffs.append(-J) 
            hamiltonian_terms.append(qml.PauliZ(i)@qml.PauliZ(j))
            hamiltonian_terms = list(set(hamiltonian_terms))

    hamiltonian = qml.Hamiltonian(coeffs,hamiltonian_terms, grouping_type='qwc')
    return hamiltonian

def construct_full_hamiltonian(J,b,lattice_size,periodic,neighbors):
    """
    Given the coupling strength term and the magnetic field strength
    returns the 2-D transverse field ising model hamiltonian for a 
    square lattice
    """
    #neighbors = find_neighbors(lattice_size,periodic)
    hamiltonian_terms = []
    coeffs = []

    for i in range(lattice_size**2):
        coeffs.append(-b) 
        hamiltonian_terms.append(qml.PauliX(i))
        for j in neighbors[i]:
#
            coeffs.append(-J) 
            hamiltonian_terms.append(qml.PauliZ(i)@qml.PauliZ(j))
            hamiltonian_terms = list(set(hamiltonian_terms))

    hamiltonian = qml.Hamiltonian(coeffs,hamiltonian_terms, grouping_type='qwc')
    return hamiltonian
def ising_variational_circuit(params, num_qubits, neighbors):
    """
    Variational circuit taken from Pennylane tutorial
    """
    for i in range(num_qubits):
        qml.RY(params[3*i], wires=i)
        qml.RX(params[3*i+1],wires=i)
        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])
        qml.RY(params[3*i+2],wires=i)


def ising_2_level_variational_circuit(params, num_qubits, neighbors):
    """
    Variational circuit taken from Pennylane tutorial
    """
    for i in range(num_qubits):
        qml.RY(params[6*i], wires=i)
        qml.RX(params[6*i+1],wires=i)
        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])
        qml.RX(params[6*i+2],wires=i)
        qml.RY(params[6*i+3], wires=i)
        qml.RX(params[6*i+4],wires=i)
        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])
        qml.RY(params[6*i+5],wires=i)

#print(construct_full_hamiltonian(J, b, lattice_size, periodic, neighbors))