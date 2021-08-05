import numpy as np
from qiskit.opflow import SummedOp

from qiskit import QuantumCircuit, execute
from qiskit import Aer
simulator = Aer.get_backend('qasm_simulator')

def generateBasis(H: SummedOp) -> str:
    n = H.num_qubits
    qubits_shift = list(np.random.choice(range(n), size=n, replace=False))
    bases_shift = []
    for j in range(n):
        basisSingle = _generateBasisSingle(j, qubits_shift, bases_shift, H)
        bases_shift.append(basisSingle)
    B = '' # measurement basis
    for i in range(n):
        j = qubits_shift.index(i)
        B = B + bases_shift[j]
    return B

def generateBasis2(H: SummedOp) -> str:
    n = H.num_qubits
    # Choose fixed best qubit ordering instead of randomizing:
    qubit_weights = [[i,0] for i in range(n)]
    for x in H:
        coeff, pauli = x.coeff, str(x.primitive)
        for i in range(n):
            if pauli[i] != 'I':
                qubit_weights[i][1] += abs(coeff)
    qubit_weights.sort(key=lambda x: x[1])
    qubits_shift = [w[0] for w in qubit_weights]
    # print('order =',qubits_shift)
    bases_shift = []
    for j in range(n):
        basisSingle = _generateBasisSingle(j, qubits_shift, bases_shift, H)
        bases_shift.append(basisSingle)
    B = '' # measurement basis
    for i in range(n):
        j = qubits_shift.index(i)
        B = B + bases_shift[j]
    return B

def _generateBasisSingle(j: int, qubits_shift: list, bases_shift: list, H: SummedOp) -> str:
    assert len(bases_shift) == j
    beta = _generateBeta(j, qubits_shift, bases_shift, H)
    basis = np.random.choice(['X', 'Y', 'Z'], p=beta)
    return basis

def _generateBeta(j, qubits_shift, bases_shift, H):
    constants = [0.0, 0.0, 0.0]
    p2index = {'X': 0, 'Y': 1, 'Z': 2}
    for x in H:
        coeff, pauli = x.coeff, str(x.primitive)
        if _isCompatible(pauli, j, qubits_shift, bases_shift):
            p = pauli[qubits_shift[j]]
            index = p2index[p]
            constants[index] += coeff**2
    if np.sum(constants) == 0.0:
        # generate uniform distribution
        beta = [1/3, 1/3, 1/3]
    else:
        beta_unnormalized = np.sqrt(constants)
        beta = beta_unnormalized / np.sum(beta_unnormalized)
    return beta

def _isCompatible(pauli, j, qubits_shift, bases_shift):
    if pauli[qubits_shift[j]] == 'I':
        return False
    for k in range(j):
        i = qubits_shift[k]
        if not pauli[i] in ('I', bases_shift[k]):
            return False
    return True

def precomputePauliFrequencies(H: SummedOp, M: int) -> dict:
    out = {}
    for x in H:
        coeff, P = x.coeff, str(x.primitive)
        out[P] = 0
    for i in range(10*M):
        B = generateBasis(H)
        for P in out.keys():
            if all([P[j]==B[j] or P[j]=='I' for j in range(len(P))]):
                out[P] += 1
    for P in out.keys():
        out[P] = out[P]/10
    return out

def updateHamiltonian(H: SummedOp, MPs: dict, B: str) -> SummedOp:
    MPs_new = {}
    scaleFactors = []
    
    for x in H:
        coeff, P = x.coeff, str(x.primitive)
        if all([P[j]==B[j] or P[j]=='I' for j in range(len(P))]):
            MPs_new[P] = MPs[P] - 1
            if MPs[P] > 1:
                scaleFactors.append(np.sqrt(1-1/MPs[P]))
                # scaleFactors.append(1)
            else:
                scaleFactors.append(0)
        else:
            MPs_new[P] = MPs[P]
            # if MPs[P] > 1:
            #     scaleFactors.append(np.sqrt(1+1/MPs[P]))
            # else:
            #     scaleFactors.append(1)
            scaleFactors.append(1)
            
    return SummedOp([H[i].mul(scaleFactors[i]) for i in range(len(H))]), MPs_new

def updateHamiltonian2(H: SummedOp, counts: dict, B: str) -> SummedOp:
    scaleFactors = []
    
    for x in H:
        coeff, P = x.coeff, str(x.primitive)
        if all([P[j]==B[j] or P[j]=='I' for j in range(len(P))]):
            if counts[P] > 0:
                scaleFactors.append(np.sqrt(counts[P]/(counts[P]+2)))
            else:
                scaleFactors.append(1/np.sqrt(2))
        else:
            scaleFactors.append(1)
            
    return SummedOp([H[i].mul(scaleFactors[i]) for i in range(len(H))])

# Includes state and shot info in variance estimates.
def rescaledHamiltonian(H: SummedOp, pauliEstimates: dict) -> SummedOp:
    scaleFactors = []

    for x in H:
        coeff, P = x.coeff, str(x.primitive)
        samples = pauliEstimates[P]['number']
        estimate = pauliEstimates[P]['running'][-1]
        if samples > 10:
            scaleFactors.append(np.sqrt((1-estimate**2)/(samples*(samples+1))))
        else:
            scaleFactors.append(1)
            
    return SummedOp([H[i].mul(scaleFactors[i]) for i in range(len(H))])

def runAndMeasure(state, basis):
    n = len(basis)
    circ = QuantumCircuit(n, n)
    circ.initialize(state, range(n))

    circ = circ.compose(_measurementCircuit(basis))

    # run experiment
    result = execute(circ, simulator, shots=1).result()
    counts = result.get_counts(circ)
    # counts is a dictionary with only one entry (since shots=1)
    bitString = counts.popitem()[0]  # physics ordering

    # return +/- evalues
    evalues = [(-1)**int(bit) for bit in bitString]
    return evalues

def _measurementCircuit(basis: str):
    n = len(basis)
    circ = QuantumCircuit(n, n)
    # qiskit ordering
    for qubit, pauli in enumerate(basis[::-1]):
        circ = _measurementPauli(circ, pauli, qubit)
    return circ


def _measurementPauli(circ, pauli, qubit):
    '''
    modify circuit by appending measurement.
    return modified circuit
    '''
    if pauli == 'X':
        circ.h(qubit)
    elif pauli == 'Y':
        circ.sdg(qubit)
        circ.h(qubit)
    elif pauli == 'Z':
        pass
    circ.measure(qubit, qubit)
    return circ


def buildPauliEstimates(H):
    pauliEstimates = {}
    # key = Pauli appearing in H
    # value = dict where
        # number = number of times a basis has allowed Pauli to be estimated
        # running = list of running best estimates of Pauli value
    for x in H:
        pauli = str(x.primitive)
        pauliEstimates[pauli] = {'number': 0, 'running': [0.0]}
    return pauliEstimates

def _isEstimatible(pauli, basis):
    for qubit in range(len(basis)):
        if not pauli[qubit] in ('I', basis[qubit]):
            return False
    return True

def _estimate(pauli, evalues):
    est = 1.0
    for qubit, p in enumerate(pauli):
        if p != 'I':
            est *= evalues[qubit]
    return est

def updatePauliEstimates(pauliEstimates, evalues, basis):
    for pauli, estimates in pauliEstimates.items():
        lastEstimate = estimates['running'][-1]
        if _isEstimatible(pauli, basis):
            est = _estimate(pauli, evalues)
            n = estimates['number']
            newEstimate = 1/(n+1) * (n * lastEstimate + est)
            estimates['number'] += 1
            estimates['running'].append(newEstimate)
        else:
            estimates['running'].append(lastEstimate)
    pass
