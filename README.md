# VQE_ansatz_for_the_J1J2-model


This is the code for one example model for the paper "Variational Quantum Eigensolver Ansatz for the J1-J2-model": https://journals.aps.org/prb/pdf/10.1103/PhysRevB.106.144426

The simulation of the Variational Quantum Eigensolver for the J1 − J2-model was done in Python 3.8.5 with the help of NumPy 1.20.2 using Cirq 0.10.0. For the optimization, the built-in optimizers from SciPy 1.6.3 were used. We used the gate cirq.X(q), cirq.Y(q),cirq.Z(q) cirq.XX(q1,q2), cirq.YY(q1,q2) and cirq.ZZ(q1,q2) built in Cirq to a power of the respective parameter. And defined the Hamiltonian via Pauli operators in Cirq via cirq.PauliSum. After applying a sufficient amount of gate-repetitions, the circuit is simulated. We used qsimcirq, a full wave function simulator written in C++ which is much faster than the normal simulator in Cirq.

The code shown is a sample code for the 16 qubit model without the gates for the diagonal interactions in the J1-J2-model.
