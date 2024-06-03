### As an example the python code for the J1J2-model with 16 qubits without the gates for the diagonal interactions ###


import cirq

import numpy as np
import sympy as sp
import random
from numpy import pi 
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qsimcirq

J1=-1 #AFM
J2 = -0.5


#Pauli-Matrices
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])
I = np.eye(2)
s =[sx,sy,sz]

#analytical eigenvalue calculated in extra code via sparse matrices 

q1, q2, q3, q4, q5,q6, q7, q8, q9, q10,q11, q12, q13, q14, q15, q16 = cirq.GridQubit.rect(4,4)  #qubits in rectangle
qubits = [q1,q2,q3,q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16]

#Pauli operators
pauli = cirq.Z
pauli_x = cirq.X
pauli_y = cirq.Y

#list to save optimization steps
l =[]

#Define the different gate sequences 

def xxyyzz_layer(t):
  yield cirq.XX(q1,q2)**t[0]
  yield cirq.YY(q1,q2)**t[0]
  yield cirq.ZZ(q1,q2)**t[0]
  yield cirq.XX(q1,q5)**t[1]
  yield cirq.YY(q1,q5)**t[1]
  yield cirq.ZZ(q1,q5)**t[1]
  yield cirq.XX(q2,q3)**t[2]
  yield cirq.YY(q2,q3)**t[2]
  yield cirq.ZZ(q2,q3)**t[2]
  yield cirq.XX(q2,q6)**t[3]
  yield cirq.YY(q2,q6)**t[3]
  yield cirq.ZZ(q2,q6)**t[3]
  yield cirq.XX(q3,q4)**t[4]
  yield cirq.YY(q3,q4)**t[4]
  yield cirq.ZZ(q3,q4)**t[4]
  yield cirq.XX(q4,q8)**t[5]
  yield cirq.YY(q4,q8)**t[5]
  yield cirq.ZZ(q4,q8)**t[5]
  yield cirq.XX(q5,q6)**t[6]
  yield cirq.YY(q5,q6)**t[6]
  yield cirq.ZZ(q5,q6)**t[6]
  yield cirq.XX(q3,q7)**t[7]
  yield cirq.YY(q3,q7)**t[7]
  yield cirq.ZZ(q3,q7)**t[7]
  yield cirq.XX(q6,q7)**t[8]
  yield cirq.YY(q6,q7)**t[8]
  yield cirq.ZZ(q6,q7)**t[8]
  yield cirq.XX(q5,q9)**t[9]
  yield cirq.YY(q5,q9)**t[9]
  yield cirq.ZZ(q5,q9)**t[9]
  yield cirq.XX(q7,q8)**t[10]
  yield cirq.YY(q7,q8)**t[10]
  yield cirq.ZZ(q7,q8)**t[10]
  yield cirq.XX(q8,q12)**t[11]
  yield cirq.YY(q8,q12)**t[11]
  yield cirq.ZZ(q8,q12)**t[11]
  yield cirq.XX(q7,q11)**t[12]
  yield cirq.YY(q7,q11)**t[12]
  yield cirq.ZZ(q7,q11)**t[12]
  yield cirq.XX(q6,q10)**t[13]
  yield cirq.YY(q6,q10)**t[13]
  yield cirq.ZZ(q6,q10)**t[13]
  yield cirq.XX(q9,q10)**t[14]
  yield cirq.YY(q9,q10)**t[14]
  yield cirq.ZZ(q9,q10)**t[14]
  yield cirq.XX(q10,q11)**t[15]
  yield cirq.YY(q10,q11)**t[15]
  yield cirq.ZZ(q10,q11)**t[15]
  yield cirq.XX(q12,q11)**t[16]
  yield cirq.YY(q12,q11)**t[16]
  yield cirq.ZZ(q12,q11)**t[16]
  yield cirq.XX(q9,q13)**t[17]
  yield cirq.YY(q9,q13)**t[17]
  yield cirq.ZZ(q9,q13)**t[17]
  yield cirq.XX(q14,q10)**t[18]
  yield cirq.YY(q14,q10)**t[18]
  yield cirq.ZZ(q14,q10)**t[18]
  yield cirq.XX(q11,q15)**t[19]
  yield cirq.YY(q11,q15)**t[19]
  yield cirq.ZZ(q11,q15)**t[19]
  yield cirq.XX(q12,q16)**t[20]
  yield cirq.YY(q12,q16)**t[20]
  yield cirq.ZZ(q12,q16)**t[20]
  yield cirq.XX(q13,q14)**t[21]
  yield cirq.YY(q13,q14)**t[21]
  yield cirq.ZZ(q13,q14)**t[21]
  yield cirq.XX(q14,q15)**t[22]
  yield cirq.YY(q14,q15)**t[22]
  yield cirq.ZZ(q14,q15)**t[22]
  yield cirq.XX(q15,q16)**t[23]
  yield cirq.YY(q15,q16)**t[23]
  yield cirq.ZZ(q15,q16)**t[23]

def z_layer(t):
  yield cirq.Z(q1)**t[0]    
  yield cirq.Z(q2)**t[1]
  yield cirq.Z(q3)**t[2]
  yield cirq.Z(q4)**t[3]
  yield cirq.Z(q5)**t[4]
  yield cirq.Z(q6)**t[5]
  yield cirq.Z(q7)**t[6]
  yield cirq.Z(q8)**t[7]
  yield cirq.Z(q9)**t[8]
  yield cirq.Z(q10)**t[9]
  yield cirq.Z(q11)**t[10]
  yield cirq.Z(q12)**t[11]
  yield cirq.Z(q13)**t[12]
  yield cirq.Z(q14)**t[13]
  yield cirq.Z(q15)**t[14]
  yield cirq.Z(q16)**t[15]
  
  
  
#Defining the H_x, H_y and H_z via PauliSum
operator_x: cirq.PauliSum = -J1*(pauli_x(q1)*pauli_x(q2)+pauli_x(q1)*pauli_x(q5)+pauli_x(q2)*pauli_x(q3)+pauli_x(q2)*pauli_x(q6) +pauli_x(q3)*pauli_x(q4)+pauli_x(q3)*pauli_x(q7)+pauli_x(q5)*pauli_x(q6)    +pauli_x(q4)*pauli_x(q8)+pauli_x(q6)*pauli_x(q7)+pauli_x(q5)*pauli_x(q9)+pauli_x(q7)*pauli_x(q8)+pauli_x(q6)*pauli_x(q10)   +pauli_x(q11)*pauli_x(q7)+pauli_x(q8)*pauli_x(q12)+pauli_x(q9)*pauli_x(q10)+pauli_x(q10)*pauli_x(q11)+pauli_x(q11)*pauli_x(q12)     +pauli_x(q9)*pauli_x(q13)+pauli_x(q10)*pauli_x(q14)+pauli_x(q11)*pauli_x(q15)+pauli_x(q12)*pauli_x(q16)+pauli_x(q13)*pauli_x(q14)+pauli_x(q14)*pauli_x(q15)+pauli_x(q15)*pauli_x(q16))-J2*(pauli_x(q1)*pauli_x(q6)+pauli_x(q2)*pauli_x(q5)+pauli_x(q2)*pauli_x(q7)+pauli_x(q3)*pauli_x(q6)+pauli_x(q3)*pauli_x(q8)+pauli_x(q4)*pauli_x(q7)+pauli_x(q6)*pauli_x(q9)+pauli_x(q5)*pauli_x(q10)+pauli_x(q6)*pauli_x(q11)+pauli_x(q7)*pauli_x(q10)+pauli_x(q7)*pauli_x(q12)+pauli_x(q8)*pauli_x(q11)                   +pauli_x(q10)*pauli_x(q13)+pauli_x(q9)*pauli_x(q14)+pauli_x(q10)*pauli_x(q15)+pauli_x(q11)*pauli_x(q14)+pauli_x(q11)*pauli_x(q16)+pauli_x(q12)*pauli_x(q15))
print("operator_x", operator_x)

operator_z: cirq.PauliSum = -J1*(pauli(q1)*pauli(q2)+pauli(q1)*pauli(q5)+pauli(q2)*pauli(q3)+pauli(q2)*pauli(q6) +pauli(q3)*pauli(q4)+pauli(q3)*pauli(q7)+pauli(q5)*pauli(q6)    +pauli(q4)*pauli(q8)+pauli(q6)*pauli(q7)+pauli(q5)*pauli(q9)+pauli(q7)*pauli(q8)+pauli(q6)*pauli(q10)   +pauli(q11)*pauli(q7)+pauli(q8)*pauli(q12)+pauli(q9)*pauli(q10)+pauli(q10)*pauli(q11)+pauli(q11)*pauli(q12)     +pauli(q9)*pauli(q13)+pauli(q10)*pauli(q14)+pauli(q11)*pauli(q15)+pauli(q12)*pauli(q16)+pauli(q13)*pauli(q14)+pauli(q14)*pauli(q15)+pauli(q15)*pauli(q16))-J2*(pauli(q1)*pauli(q6)+pauli(q2)*pauli(q5)+pauli(q2)*pauli(q7)+pauli(q3)*pauli(q6)+pauli(q3)*pauli(q8)+pauli(q4)*pauli(q7)+pauli(q6)*pauli(q9)+pauli(q5)*pauli(q10)+pauli(q6)*pauli(q11)+pauli(q7)*pauli(q10)+pauli(q7)*pauli(q12)+pauli(q8)*pauli(q11)                   +pauli(q10)*pauli(q13)+pauli(q9)*pauli(q14)+pauli(q10)*pauli(q15)+pauli(q11)*pauli(q14)+pauli(q11)*pauli(q16)+pauli(q12)*pauli(q15))
print("operator_z", operator_z)

operator_y: cirq.PauliSum = -J1*(pauli_y(q1)*pauli_y(q2)+pauli_y(q1)*pauli_y(q5)+pauli_y(q2)*pauli_y(q3)+pauli_y(q2)*pauli_y(q6) +pauli_y(q3)*pauli_y(q4)+pauli_y(q3)*pauli_y(q7)+pauli_y(q5)*pauli_y(q6)    +pauli_y(q4)*pauli_y(q8)+pauli_y(q6)*pauli_y(q7)+pauli_y(q5)*pauli_y(q9)+pauli_y(q7)*pauli_y(q8)+pauli_y(q6)*pauli_y(q10)   +pauli_y(q11)*pauli_y(q7)+pauli_y(q8)*pauli_y(q12)+pauli_y(q9)*pauli_y(q10)+pauli_y(q10)*pauli_y(q11)+pauli_y(q11)*pauli_y(q12)     +pauli_y(q9)*pauli_y(q13)+pauli_y(q10)*pauli_y(q14)+pauli_y(q11)*pauli_y(q15)+pauli_y(q12)*pauli_y(q16)+pauli_y(q13)*pauli_y(q14)+pauli_y(q14)*pauli_y(q15)+pauli_y(q15)*pauli_y(q16))-J2*(pauli_y(q1)*pauli_y(q6)+pauli_y(q2)*pauli_y(q5)+pauli_y(q2)*pauli_y(q7)+pauli_y(q3)*pauli_y(q6)+pauli_y(q3)*pauli_y(q8)+pauli_y(q4)*pauli_y(q7)+pauli_y(q6)*pauli_y(q9)+pauli_y(q5)*pauli_y(q10)+pauli_y(q6)*pauli_y(q11)+pauli_y(q7)*pauli_y(q10)+pauli_y(q7)*pauli_y(q12)+pauli_y(q8)*pauli_y(q11)                   +pauli_y(q10)*pauli_y(q13)+pauli_y(q9)*pauli_y(q14)+pauli_y(q10)*pauli_y(q15)+pauli_y(q11)*pauli_y(q14)+pauli_y(q11)*pauli_y(q16)+pauli_y(q12)*pauli_y(q15))
print("operator_y", operator_y)  


def vqe(parameters,measure):
     
  circuit = cirq.Circuit()
  gate = cirq.CXPowGate(exponent=1) 

   
  theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11, theta12, theta13, theta14, theta15, theta16, theta17, theta18, theta19, theta20, theta21, theta22, theta23, theta24, theta25, theta26, theta27, theta28, theta29, theta30, theta31, theta32,theta33, theta34, theta35, theta36, theta37, theta38, theta39, theta40, theta41, theta42, theta43, theta44, theta45, theta46, theta47, theta48,theta49, theta50, theta51, theta52, theta53, theta54, theta55, theta56, theta57, theta58, theta59, theta60, theta61, theta62, theta63, theta64, theta65, theta66, theta67, theta68, theta69, theta70, theta71, theta72, theta73, theta74, theta75, theta76, theta77, theta78, theta79, theta80, theta81, theta82, theta83, theta84, theta85, theta86, theta87, theta88, theta89, theta90, theta91, theta92, theta93, theta94, theta95, theta96, theta97, theta98, theta99, theta100, theta101, theta102, theta103, theta104, theta105,theta106, theta107, theta108, theta109, theta110, theta111, theta112, theta113, theta114, theta115, theta116, theta117, theta118, theta119, theta120, theta121, theta122, theta123, theta124, theta125, theta126, theta127, theta128, theta129, theta130, theta131, theta132, theta133, theta134, theta135, theta136, theta137, theta138, theta139, theta140, theta141, theta142, theta143, theta144, theta145, theta146,theta147, theta148, theta149, theta150, theta151, theta152, theta153, theta154, theta155, theta156, theta157, theta158, theta159, theta160, theta161, theta162,theta163, theta164, theta165, theta166, theta167, theta168, theta169, theta170, theta171, theta172, theta173, theta174, theta175, theta176, theta177, theta178, theta179, theta180, theta181, theta182, theta183, theta184, theta185, theta186, theta187, theta188, theta189, theta190, theta191, theta192, theta193, theta194, theta195, theta196, theta197, theta198, theta199, theta200, theta201, theta202, theta203,theta204, theta205, theta206, theta207, theta208, theta209, theta210, theta211, theta212, theta213, theta214, theta215, theta216, theta217, theta218, theta219,theta220, theta221, theta222, theta223, theta224, theta225, theta226, theta227, theta228, theta229, theta230, theta231, theta232, theta234, theta235, theta236, theta237, theta238, theta239, theta240, theta241, theta242, theta243, theta244, theta245, theta246, theta247, theta248, theta249, theta250, theta251, theta252, theta253, theta254, theta255, theta256, theta257, theta258, theta259, theta260, theta261,theta262, theta263, theta264, theta265, theta266, theta267, theta268, theta269, theta270, theta271, theta272, theta273, theta274, theta275, theta276, theta277,theta278, theta279, theta280, theta281, theta282, theta283, theta284, theta285, theta286, theta287, theta288, theta289, theta290, theta291, theta292, theta293, theta294, theta295, theta296, theta297, theta298, theta299, theta300, theta301, theta302, theta303, theta304, theta305, theta306, theta307, theta308, theta309, theta310, theta311, theta312, theta313, theta314, theta315, theta316, theta317, theta318,theta319, theta320, theta321, theta322, theta323, theta324, theta325, theta326, theta327, theta328, theta329, theta330, theta331, theta332, theta333, theta334,theta335, theta336, theta337, theta338, theta339, theta340, theta341, theta342, theta343, theta344, theta345, theta346, theta347, theta348, theta349, theta350, theta351, theta352, theta353, theta354, theta355, theta356, theta357, theta358, theta359, theta360, theta361, theta362, theta363, theta364, theta365, theta366, theta367, theta368, theta369, theta370, theta371, theta372, theta373, theta374, theta375,theta376, theta377, theta378, theta379, theta380, theta381, theta382, theta383, theta384, theta385, theta386, theta387, theta388, theta389, theta390, theta391,theta392, theta393, theta394, theta395, theta396, theta397, theta398, theta399, theta400, theta401, theta402, theta403, theta404, theta405, theta406, theta407, theta408, theta409, theta410, theta411, theta412, theta413, theta414, theta415, theta416, theta417, theta418, theta419, theta420, theta421, theta422, theta423, theta424, theta425, theta426, theta427, theta428, theta429, theta430, theta431, theta432 = parameters
  


  circuit.append(cirq.X(q1)**parameters[1])
  circuit.append(cirq.X(q2)**parameters[2])  
  circuit.append(cirq.X(q3)**parameters[3])
  circuit.append(cirq.X(q4)**parameters[4])  
  circuit.append(cirq.X(q5)**parameters[5])
  circuit.append(cirq.X(q6)**parameters[6])  
  circuit.append(cirq.X(q7)**parameters[7])
  circuit.append(cirq.X(q8)**parameters[8])  
  circuit.append(cirq.X(q9)**parameters[9])
  circuit.append(cirq.X(q10)**parameters[10])  
  circuit.append(cirq.X(q11)**parameters[11])
  circuit.append(cirq.X(q12)**parameters[12])  
  circuit.append(cirq.X(q13)**parameters[13])
  circuit.append(cirq.X(q14)**parameters[14])
  circuit.append(cirq.X(q15)**parameters[15])
  circuit.append(cirq.X(q16)**parameters[16]) 

  circuit.append(cirq.Y(q1)**parameters[17])
  circuit.append(cirq.Y(q2)**parameters[18])  
  circuit.append(cirq.Y(q3)**parameters[19])
  circuit.append(cirq.Y(q4)**parameters[20])  
  circuit.append(cirq.Y(q5)**parameters[21])
  circuit.append(cirq.Y(q6)**parameters[22])  
  circuit.append(cirq.Y(q7)**parameters[23])
  circuit.append(cirq.Y(q8)**parameters[24])  
  circuit.append(cirq.Y(q9)**parameters[25])
  circuit.append(cirq.Y(q10)**parameters[26])  
  circuit.append(cirq.Y(q11)**parameters[27])
  circuit.append(cirq.Y(q12)**parameters[28])  
  circuit.append(cirq.Y(q13)**parameters[29])
  circuit.append(cirq.Y(q14)**parameters[30])
  circuit.append(cirq.Y(q15)**parameters[31])
  circuit.append(cirq.Y(q16)**parameters[32]) 
  
  
  circuit.append(z_layer(parameters[33:49]))
  circuit.append(xxyyzz_layer(parameters[49:75]))

  circuit.append(z_layer(parameters[75:91]))
  circuit.append(xxyyzz_layer(parameters[91:117]))

  circuit.append(z_layer(parameters[117:133]))
  circuit.append(xxyyzz_layer(parameters[133:159]))

  circuit.append(z_layer(parameters[159:175]))
  circuit.append(xxyyzz_layer(parameters[175:201]))

  circuit.append(z_layer(parameters[201:217]))
  circuit.append(xxyyzz_layer(parameters[217:243]))

  circuit.append(z_layer(parameters[243:259]))
  circuit.append(xxyyzz_layer(parameters[259:285]))

  circuit.append(z_layer(parameters[285:301]))
  circuit.append(xxyyzz_layer(parameters[301:330]))
  

  
  qsim_simulator = qsimcirq.QSimSimulator()
  qsim_results_s = qsim_simulator.simulate(circuit).final_state_vector
  qsim_results_sv = qsim_results_s/np.linalg.norm(qsim_results_s)

 
 
  if measure == 'Z':
    expectation = operator_z.expectation_from_state_vector(qsim_results_sv, qubit_map={i: qubits.index(i) for i in [q1,q2,q3,q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16]})
  elif measure == 'X':
    expectation = operator_x.expectation_from_state_vector(qsim_results_sv, qubit_map={i: qubits.index(i) for i in [q1,q2,q3,q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16]})
  elif measure == 'Y':
    expectation = operator_y.expectation_from_state_vector(qsim_results_sv, qubit_map={i: qubits.index(i) for i in [q1,q2,q3,q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16]})
  else: 
    raise ValueError('Not valid input for measurement: Input should be "X" or "Y" or "Z"')  
  return np.real(expectation)



def expect(parameters):

  Z = vqe(np.real(parameters),'Z')
  Y = vqe(np.real(parameters),'Y')
  X = vqe(np.real(parameters),'X')
  add = (X+Y+Z)
  l.append(add)
  return add


#random initial guess
initial_guess = [random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi),random.uniform(-np.pi, np.pi)]


print(initial_guess)
print(expect(initial_guess))

#classical minimizer


minimizer_kwargs = {"method": "COBYLA","options": {"tol":1e-8, "maxiter":50000}}
'''
For Basinhopping, e.g.
ret = basinhopping(expect,initial_guess, minimizer_kwargs=minimizer_kwargs,niter=2)
print(ret)
'''

res = minimize(expect , initial guess ,method= ’COBYLA’ , tol =1e−08, options={’maxiter’: 50000})
print(res)


string ='data_optimization_J1J2_16-qubits_wo_diaggates.dat'
np.savetxt(string, l)


plt.plot(l, 'r')
plt.show()
