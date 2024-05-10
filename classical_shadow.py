import pennylane as qml
from pennylane import numpy as np
from shadow_functions import (
    find_neighbors, construct_hamiltonian 
    )
import argparse
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import scipy



#np.random.seed(11)
def get_config():
    parser = ArgumentParser()

    parser.add_argument('--num_qubits', type=int, default=1)
    parser.add_argument('--num_snapshots', type = int, default=1)
    parser.add_argument('--circuit', type=str, default='zero')
    parser.add_argument('--num_trials', type=int, default=10)


    args = parser.parse_args()
    return args

    zero = np.array([[1,0]])
    one = np.array([[0,1]])
    zero_state = np.array([[1,0],[0,0]])
    one_state = np.array([[0,0],[0,1]])
    phase_z = np.array([[1,0],[0,-1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))
def state_dict(state):
    dic = {
        'zero' : np.array([[1,0]]),
        'one' : np.array([[0,1]]),
        'pi_0' : np.array([[1,0],[0,0]]),
        'pi_1' : np.array([[0,0],[0,1]]),
        'phase' : np.array([[1,0],[0,-1j]], dtype=complex),
        'H' : qml.matrix(qml.Hadamard(0)),
        'I' : qml.matrix(qml.Identity(0))
     }
    return dic[state]


def snapshot_state(b_list = None, obs_list = None, im_time_evo_op = None):

    num_qubits = len(b_list)

    zero = np.array([[1,0]])
    one = np.array([[0,1]])
    zero_state = np.array([[1,0],[0,0]])
    one_state = np.array([[0,0],[0,1]])
    phase_z = np.array([[1,0],[0,-1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))

    unitaries = [hadamard, hadamard@phase_z, identity]
    #rhos = []
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot




def calculate_classical_shadow(circuit_template, shadow_size, num_qubits, prob=0.8):
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))

    for num_shadows in range(shadow_size):
        obs = [unitary_ensemble[int(unitary_ids[num_shadows, i])](i) for i in range(num_qubits)]
        if np.random.rand() > prob:
            outcomes[num_shadows, :] = 1 - circuit_template(num_qubits, obs)  # Flip state with 1-prob
        else:
            outcomes[num_shadows, :] = circuit_template(num_qubits, obs)

    return outcomes, unitary_ids


def shadow_state_reconstruction(shadow, num_qubits):
    num_snapshots, _ = shadow[0].shape
    b_lists, obs_lists = shadow
    shadow_rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])  # assuming snapshot_state is defined elsewhere

    return shadow_rho / np.trace(shadow_rho)


def fidelity(rho, sigma):
    sqrt_rho = scipy.linalg.sqrtm(rho)
    product = np.dot(sqrt_rho, np.dot(sigma, sqrt_rho))
    
    sqrt_product = scipy.linalg.sqrtm(product)
    
    fidelity_value = (np.trace(sqrt_product))**2
    return np.real(fidelity_value) - .05




def main():
    config = get_config()

    num_qubits = config.num_qubits
    dev = qml.device("default.qubit", wires=num_qubits, shots=1)

    circuit = config.circuit

    if circuit == 'zero':
        @qml.qnode(dev)
        def ansatz_circuit(num_qubits, obs): 
            out = [qml.expval(o) for o in obs]
            return out
        check_state=np.array([[1,0],[0,0]])

    elif circuit == 'plus':
        @qml.qnode(dev)
        def ansatz_circuit(num_qubits, obs):
            qml.Hadamard(wires=0) 
            out = [qml.expval(o) for o in obs]
            return out       
        check_state = 1/2*np.array([[1,1],[1,1]])

    elif circuit == 'quED':
        @qml.qnode(dev)
        def ansatz_circuit(num_qubits, obs):
            theta = 2 * np.arccos(np.sqrt(0.72))
            qml.RY(theta, wires=0)
            out = [qml.expval(o) for o in obs]
            return out
        check_state = np.array([[.72,.449],[.449,.28]])
    
    fidelities = []
    rho_final = np.zeros((2,2))
    for i in range(config.num_trials):

        num_snapshots = config.num_snapshots
        #print(num_snapshots)
        rho_hat = create_classical_shadow(ansatz_circuit, num_snapshots, num_qubits)

        #print(rho_hat.real)
        #print(np.trace(rho_hat@rho_hat))

        fidelities.append(fidelity(rho_hat,check_state))
        rho_final = rho_final+rho_hat
    print(f'\n Fidelity Mean: {np.mean(fidelities)}, Fidelity Var: {np.var(fidelities)}\n')
    print(f'Reconstructed density matrix: \n{rho_final / config.num_trials}')



if __name__ == '__main__':
    main()

    
    



#next --> i think MPS is kinda fucked
#implement --> shadow VQE and compare with tdvp
