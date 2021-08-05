#!/usr/bin/python

import sys
import time
import multiprocessing
import numpy as np

from qiskit.opflow import SummedOp
from hamiltonian import hamiltonians
from circuits import generateBasis, generateBasis2, precomputePauliFrequencies, updateHamiltonian, updateHamiltonian2, rescaledHamiltonian, runAndMeasure, buildPauliEstimates, updatePauliEstimates

def simulation(x):
    name, ground_state, n_shots_per_worker = x
    H = hamiltonians[name].SummedOp()
    pauliEstimates = buildPauliEstimates(H)
    for shot in range(n_shots_per_worker):
        basis = generateBasis(H)
        evalues = runAndMeasure(ground_state, basis)
        updatePauliEstimates(pauliEstimates, evalues, basis)
    return pauliEstimates

def simulation2(x):
    name, ground_state, n_shots_per_worker = x
    H = hamiltonians[name].SummedOp()
    pauliEstimates = buildPauliEstimates(H)
    for shot in range(n_shots_per_worker):
        basis = generateBasis2(H)
        evalues = runAndMeasure(ground_state, basis)
        updatePauliEstimates(pauliEstimates, evalues, basis)
    return pauliEstimates

def stitch_simulations(H, pauliEstimatesMultiple):
    pauliEstimatesBest = {}
    for x in H:
        pauli = str(x.primitive)
        estimates = []
        for pE in pauliEstimatesMultiple:
            number = pE[pauli]['number']
            estimate = pE[pauli]['running'][-1]
            estimates.append((number, estimate))
        pauliEstimatesBest[pauli] = _stitch_estimates(estimates)
    return pauliEstimatesBest

def _stitch_estimates(estimates):
    number_total = sum(estimates[i][0] for i in range(len(estimates)))
    if number_total == 0:
        estimate = 0.0
    else:
        estimate = 1/number_total * sum(np.prod(estimates[i]) for i in range(len(estimates)))
    return estimate

def energyEstimate(H, pauliEstimatesBest):
    energyRunning = 0.0
    for x in H:
        coeff, pauli = x.coeff, str(x.primitive)
        energyRunning += coeff * pauliEstimatesBest[pauli]
    return energyRunning

def simulate(name, n_shots, n_reps, v):
    assert name in hamiltonians, "Molecule not recognized."
    ham = hamiltonians[name]
    print("\nMolecule is {} in {} encoding.".format(name.split('_')[0],name.split('_')[1]))
    print("Number of shots is set to {}.".format(n_shots))
    print("Number of reps per Hamiltonian is set to {}.".format(n_reps))
    if v == 1:
        print("APS version 1.")
    elif v == 2:
        print("APS version 2 (A2PS).")
    elif v == 3:
        print("APS version 3 (fixed qubit ordering).")
    
    ### Ground information
    print("Calculating ground energy and ground state...")
    t0 = time.time()
    ground_energy, ground_state = ham.ground(sparse=True)
    t1 = time.time()
    diff = int(t1-t0)
    print("Done\nGround information calculated in {}min{}sec.".format(diff // 60, diff % 60))

    H = ham.SummedOp()
    all_results = []

    if v == 1:
        # Original version of APS (no shot-to-shot adaptivity)
        for i in range(n_reps):
            print("\nRep {}...".format(i))

            ### Simulation
            n_workers = 15
            n_shots_per_worker = int(np.ceil(n_shots / n_workers))
            #assert n_workers * n_shots_per_worker == n_shots, "Your math doesn't check out!"

            print("Pooling {} workers to simulate roughly {} shots each...".format(n_workers, n_shots_per_worker))
            t0 = time.time()
            p = multiprocessing.Pool(n_workers)
            x = (name, ground_state, n_shots_per_worker)
            y = (name, ground_state, n_shots - (n_workers-1)*n_shots_per_worker)
            inputs = [x for _ in range(n_workers-1)]
            inputs.append(y)
            outputs = p.map(simulation, inputs)
            t1 = time.time()
            diff = int(t1-t0)
            print("Done\nPooling took {}min{}sec.".format(diff // 60, diff % 60))

            ### Return energy estimate
            pauliEstimatesMultiple = outputs
            pauliEstimatesBest = stitch_simulations(H, pauliEstimatesMultiple)
            estimate = energyEstimate(H, pauliEstimatesBest)

            print('true       :', ground_energy)
            print('estimate   :', estimate)
            print('difference :', estimate - ground_energy)
            all_results.append([estimate, estimate - ground_energy])
    
    elif v == 2:
        # Version 2 (shot-to-shot adaptivity)
        for i in range(n_reps):
            print("\nRep {}...".format(i))

            # print("Precomputing term frequency estimates...")
            # t0 = time.time()
            # MPs = precomputePauliFrequencies(H, n_shots)
            # # print("MPs =",MPs)
            # t1 = time.time()
            # diff = int(t1-t0)
            # print("Done\nPrecomputation took {}min{}sec.".format(diff // 60, diff % 60))

            H_current = SummedOp([H[i] for i in range(len(H))])

            print("Main simulation...")
            t0 = time.time()
            pauliEstimates = buildPauliEstimates(H)
            counts = {str(x.primitive):0 for x in H}
            for shot in range(n_shots):
                basis = generateBasis(H_current)
                # H_current, MPs = updateHamiltonian(H_current, MPs, basis)
                H_current = updateHamiltonian2(H_current, counts, basis)
                evalues = runAndMeasure(ground_state, basis)
                updatePauliEstimates(pauliEstimates, evalues, basis)
                for x in H:
                    P = str(x.primitive)
                    if all([P[j]==basis[j] or P[j]=='I' for j in range(len(P))]):
                        counts[P] += 1
            # pauliEstimatesBest = {P:pauliEstimates[P]['running'][-1] for P in pauliEstimates.keys()}
            pauliEstimatesBest = stitch_simulations(H, [pauliEstimates])
            estimate = energyEstimate(H, pauliEstimatesBest)
            t1 = time.time()
            diff = int(t1-t0)
            # print("MPs =",MPs)
            print("Done\nSimulation took {}min{}sec.".format(diff // 60, diff % 60))

            print('true       :', ground_energy)
            print('estimate   :', estimate)
            print('difference :', estimate - ground_energy)
            # for P in MPs.keys():
            #     if MPs[P] > 0:
            #         print(P,MPs[P],MPs[P]-counts[P])
            # print(sum([MPs[P]-counts[P] for P in MPs.keys()]))
            # print(sum([(MPs[P]-counts[P])**2 for P in MPs.keys()])**0.5)
            all_results.append([estimate, estimate - ground_energy])

    elif v == 3:
        # Version 3 (fixed qubit ordering)
        for i in range(n_reps):
            print("\nRep {}...".format(i))

            ### Simulation
            n_workers = 15
            n_shots_per_worker = int(np.ceil(n_shots / n_workers))
            #assert n_workers * n_shots_per_worker == n_shots, "Your math doesn't check out!"

            print("Pooling {} workers to simulate roughly {} shots each...".format(n_workers, n_shots_per_worker))
            t0 = time.time()
            p = multiprocessing.Pool(n_workers)
            x = (name, ground_state, n_shots_per_worker)
            y = (name, ground_state, n_shots - (n_workers-1)*n_shots_per_worker)
            inputs = [x for _ in range(n_workers-1)]
            inputs.append(y)
            outputs = p.map(simulation2, inputs)
            t1 = time.time()
            diff = int(t1-t0)
            print("Done\nPooling took {}min{}sec.".format(diff // 60, diff % 60))

            ### Return energy estimate
            pauliEstimatesMultiple = outputs
            pauliEstimatesBest = stitch_simulations(H, pauliEstimatesMultiple)
            estimate = energyEstimate(H, pauliEstimatesBest)

            print('true       :', ground_energy)
            print('estimate   :', estimate)
            print('difference :', estimate - ground_energy)
            all_results.append([estimate, estimate - ground_energy])

    elif v == 4:
        # Version 4 (shot-to-shot adaptivity including state estimate info)
        for i in range(n_reps):
            print("\nRep {}...".format(i))
            print("Main simulation...")
            t0 = time.time()
            pauliEstimates = buildPauliEstimates(H)
            counts = {str(x.primitive):0 for x in H}
            H_rescaled = SummedOp([H[i] for i in range(len(H))])
            for shot in range(n_shots):
                basis = generateBasis(H_rescaled)
                evalues = runAndMeasure(ground_state, basis)
                updatePauliEstimates(pauliEstimates, evalues, basis)
                H_rescaled = rescaledHamiltonian(H_rescaled, pauliEstimates)
            pauliEstimatesBest = stitch_simulations(H, [pauliEstimates])
            estimate = energyEstimate(H, pauliEstimatesBest)
            t1 = time.time()
            diff = int(t1-t0)
            print("Done\nSimulation took {}min{}sec.".format(diff // 60, diff % 60))

            print('true       :', ground_energy)
            print('estimate   :', estimate)
            print('difference :', estimate - ground_energy)
            all_results.append([estimate, estimate - ground_energy])
    
    else:
        raise ValueError("Bad version number.")
        
    print("all results for {}:".format(name), all_results)
    print("\n##################################")


def main():
    ### Preparation
    name = sys.argv[1]
    n_shots = int(sys.argv[2])
    n_reps = int(sys.argv[3])
    v = int(sys.argv[4])

    if name == "all":
        for eachname in hamiltonians:
            simulate(eachname, n_shots, n_reps, v)
    else:
        simulate(name, n_shots, n_reps, v)

    

if __name__ == "__main__":
    main()
