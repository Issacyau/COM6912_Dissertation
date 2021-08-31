import numpy as np
import itertools
from scipy.optimize import minimize

best_optimal_sol = [21946,21716,20754,21464,21814,22176,21799,21397,22493,20983,40767,41304,41560,41041,40872,41058,41062,42719,42230,41700,57494,60027,58025,60776,58884,60011,58132,59064,58975,60603]

def load_file_3(file):
    lines = []
    data = {}
    with open(file) as f:
        for line in f:
            lines.append(int(x) for x in line.split())

    flat_list = list(itertools.chain.from_iterable(lines))
    number_test_problems = flat_list[0]
    del flat_list[0]

    step = len(flat_list)/number_test_problems
    split_list = [flat_list[i:i + int(step)] for i in range(0, len(flat_list), int(step))]
    
    for prb,x in enumerate(split_list):
        data_prb = {}
        n = x[0] ### num variables
        m = x[1] ### num constraint

        data_prb["number_variables_problem"] = n ### 100
        data_prb["number_constraint_problem"] = m ###30
        data_prb["opt_solution_problem"] = x[2]

        data_prb["pj"] = x[3:3+n]
        data_prb["r1j"] = x[3+n:3+2*n]
        data_prb["r2j"] = x[3+2*n:3+3*n]
        data_prb["r3j"] = x[3+3*n:3+4*n]
        data_prb["r4j"] = x[3+4*n:3+5*n]
        data_prb["r5j"]= x[3+5*n:3+6*n]
        data_prb["r6j"]= x[3+6*n:3+7*n]
        data_prb["r7j"]= x[3+7*n:3+8*n]
        data_prb["r8j"]= x[3+8*n:3+9*n]
        data_prb["r9j"]= x[3+9*n:3+10*n]
        data_prb["r10j"]= x[3+10*n:3+11*n]
        data_prb["r11j"]= x[3+11*n:3+12*n]
        data_prb["r12j"]= x[3+12*n:3+13*n]
        data_prb["r13j"]= x[3+13*n:3+14*n]
        data_prb["r14j"]= x[3+14*n:3+15*n]
        data_prb["r15j"]= x[3+15*n:3+16*n]
        data_prb["r16j"]= x[3+16*n:3+17*n]
        data_prb["r17j"]= x[3+17*n:3+18*n]
        data_prb["r18j"]= x[3+18*n:3+19*n]
        data_prb["r19j"]= x[3+19*n:3+20*n]
        data_prb["r20j"]= x[3+20*n:3+21*n]
        data_prb["r21j"]= x[3+21*n:3+22*n]
        data_prb["r22j"]= x[3+22*n:3+23*n]
        data_prb["r23j"]= x[3+23*n:3+24*n]
        data_prb["r24j"]= x[3+24*n:3+25*n]
        data_prb["r25j"]= x[3+25*n:3+26*n]
        data_prb["r26j"]= x[3+26*n:3+27*n]
        data_prb["r27j"]= x[3+27*n:3+28*n]
        data_prb["r28j"]= x[3+28*n:3+29*n]
        data_prb["r29j"]= x[3+29*n:3+30*n]
        data_prb["r30j"]= x[3+30*n:3+31*n]
        data_prb["bi"] = x[3+31*n:]
        data["prb"+str(prb)] = data_prb
    return number_test_problems,data
    
num_test_prb_7,mknapcb7_data = load_file_3("mknapcb7.txt") ### loaded the first problem file
print(len(mknapcb7_data),len(best_optimal_sol))

def lp_relaxed(weights, values, capacity, N): ### weights = rij; capacity=bi; N = 100;
    cons = ({'type': 'ineq', 'fun': lambda x: np.dot(x, weights) - capacity})
    bnds = [(0, 1) for x in range(N)]
    return minimize(lambda x: np.dot(x, values), \
                    np.zeros(N),
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons).fun

def sort_index(data):
    sort_index = []
    for i in range(30):
        coefficient = [data["prb"+str(i)]["r1j"],data["prb"+str(i)]["r2j"],data["prb"+str(i)]["r3j"],data["prb"+str(i)]["r4j"],data["prb"+str(i)]["r5j"],
                      data["prb"+str(i)]["r6j"],data["prb"+str(i)]["r7j"],data["prb"+str(i)]["r8j"],data["prb"+str(i)]["r9j"],data["prb"+str(i)]["r10j"],
                      data["prb"+str(i)]["r11j"],data["prb"+str(i)]["r12j"],data["prb"+str(i)]["r13j"],data["prb"+str(i)]["r14j"],data["prb"+str(i)]["r15j"],
                      data["prb"+str(i)]["r16j"],data["prb"+str(i)]["r17j"],data["prb"+str(i)]["r18j"],data["prb"+str(i)]["r19j"],data["prb"+str(i)]["r20j"],
                      data["prb"+str(i)]["r21j"],data["prb"+str(i)]["r22j"],data["prb"+str(i)]["r23j"],data["prb"+str(i)]["r24j"],data["prb"+str(i)]["r25j"],
                      data["prb"+str(i)]["r26j"],data["prb"+str(i)]["r27j"],data["prb"+str(i)]["r28j"],data["prb"+str(i)]["r29j"],data["prb"+str(i)]["r30j"]]
        capacities = data["prb"+str(i)]["bi"]
        N = 100
        values = data["prb"+str(i)]["pj"]
        Omega = [lp_relaxed(weights, values, capacity, N) for weights, capacity in zip(coefficient, capacities)]
        u1 = np.dot(np.asarray(Omega)[0],data["prb"+str(i)]["r1j"])
        u2 = np.dot(np.asarray(Omega)[1],data["prb"+str(i)]["r2j"])
        u3 = np.dot(np.asarray(Omega)[2],data["prb"+str(i)]["r3j"])
        u4 = np.dot(np.asarray(Omega)[3],data["prb"+str(i)]["r4j"])
        u5 = np.dot(np.asarray(Omega)[4],data["prb"+str(i)]["r5j"])
        u6 = np.dot(np.asarray(Omega)[5],data["prb"+str(i)]["r6j"])
        u7 = np.dot(np.asarray(Omega)[6],data["prb"+str(i)]["r7j"])
        u8 = np.dot(np.asarray(Omega)[7],data["prb"+str(i)]["r8j"])
        u9 = np.dot(np.asarray(Omega)[8],data["prb"+str(i)]["r9j"])
        u10 = np.dot(np.asarray(Omega)[9],data["prb"+str(i)]["r10j"])
        u11 = np.dot(np.asarray(Omega)[10],data["prb"+str(i)]["r11j"])
        u12 = np.dot(np.asarray(Omega)[11],data["prb"+str(i)]["r12j"])
        u13 = np.dot(np.asarray(Omega)[12],data["prb"+str(i)]["r13j"])
        u14 = np.dot(np.asarray(Omega)[13],data["prb"+str(i)]["r14j"])
        u15 = np.dot(np.asarray(Omega)[14],data["prb"+str(i)]["r15j"])
        u16 = np.dot(np.asarray(Omega)[15],data["prb"+str(i)]["r16j"])
        u17 = np.dot(np.asarray(Omega)[16],data["prb"+str(i)]["r17j"])
        u18 = np.dot(np.asarray(Omega)[17],data["prb"+str(i)]["r18j"])
        u19 = np.dot(np.asarray(Omega)[18],data["prb"+str(i)]["r19j"])
        u20 = np.dot(np.asarray(Omega)[19],data["prb"+str(i)]["r20j"])
        u21 = np.dot(np.asarray(Omega)[20],data["prb"+str(i)]["r21j"])
        u22 = np.dot(np.asarray(Omega)[21],data["prb"+str(i)]["r22j"])
        u23 = np.dot(np.asarray(Omega)[22],data["prb"+str(i)]["r23j"])
        u24 = np.dot(np.asarray(Omega)[23],data["prb"+str(i)]["r24j"])
        u25 = np.dot(np.asarray(Omega)[24],data["prb"+str(i)]["r25j"])
        u26 = np.dot(np.asarray(Omega)[25],data["prb"+str(i)]["r26j"])
        u27 = np.dot(np.asarray(Omega)[26],data["prb"+str(i)]["r27j"])
        u28 = np.dot(np.asarray(Omega)[27],data["prb"+str(i)]["r28j"])
        u29 = np.dot(np.asarray(Omega)[28],data["prb"+str(i)]["r29j"])
        u30 = np.dot(np.asarray(Omega)[29],data["prb"+str(i)]["r30j"])
        uj = data["prb"+str(i)]["pj"]/(u1+u2+u3+u4+u5+u6+u7+u8+u9+u10+u11+u12+u13+u14+u15+u16+u17+u18+u19+u20+
                                      u21+u22+u23+u24+u25+u26+u27+u28+u29+u30)
        sorted_index = np.argsort(uj)
        sort_index.append(sorted_index)
    return sort_index
    
sorted_index = sort_index(mknapcb7_data)

def repair_operator(data,C,sorted_index): ### Transform infeasible child to feasible for test problem 1
    R1 = np.dot(data["r1j"],C)
    R2 = np.dot(data["r2j"],C)
    R3 = np.dot(data["r3j"],C)
    R4 = np.dot(data["r4j"],C)
    R5 = np.dot(data["r5j"],C)
    R6 = np.dot(data["r6j"],C)
    R7 = np.dot(data["r7j"],C)
    R8 = np.dot(data["r8j"],C)
    R9 = np.dot(data["r9j"],C)
    R10 = np.dot(data["r10j"],C)
    R11 = np.dot(data["r11j"],C)
    R12 = np.dot(data["r12j"],C)
    R13 = np.dot(data["r13j"],C)
    R14 = np.dot(data["r14j"],C)
    R15 = np.dot(data["r15j"],C)
    R16 = np.dot(data["r16j"],C)
    R17 = np.dot(data["r17j"],C)
    R18 = np.dot(data["r18j"],C)
    R19 = np.dot(data["r19j"],C)
    R20 = np.dot(data["r20j"],C)
    R21 = np.dot(data["r21j"],C)
    R22 = np.dot(data["r22j"],C)
    R23 = np.dot(data["r23j"],C)
    R24 = np.dot(data["r24j"],C)
    R25 = np.dot(data["r25j"],C)
    R26 = np.dot(data["r26j"],C)
    R27 = np.dot(data["r27j"],C)
    R28 = np.dot(data["r28j"],C)
    R29 = np.dot(data["r29j"],C)
    R30 = np.dot(data["r30j"],C)

    b1 = data["bi"][0]
    b2 = data["bi"][1]
    b3 = data["bi"][2]
    b4 = data["bi"][3]
    b5 = data["bi"][4]
    b6 = data["bi"][5]
    b7 = data["bi"][6]
    b8 = data["bi"][7]
    b9 = data["bi"][8]
    b10 = data["bi"][9]
    b11 = data["bi"][10]
    b12 = data["bi"][11]
    b13 = data["bi"][12]
    b14 = data["bi"][13]
    b15 = data["bi"][14]
    b16 = data["bi"][15]
    b17 = data["bi"][16]
    b18 = data["bi"][17]
    b19 = data["bi"][18]
    b20 = data["bi"][19]
    b21 = data["bi"][20]
    b22 = data["bi"][21]
    b23 = data["bi"][22]
    b24 = data["bi"][23]
    b25 = data["bi"][24]
    b26 = data["bi"][25]
    b27 = data["bi"][26]
    b28 = data["bi"][27]
    b29 = data["bi"][28]
    b30 = data["bi"][29]

    for j in (sorted_index):
        if (C[j] == 1) and (R1>b1 or R2>b2 or R3>b3 or R4>b4 or R5>b5 or R6>b6 or R7>b7 or R8>b8 or R9>b9 or R10>b10
                           or R11>b11 or R12>b12 or R13>b13 or R14>b14 or R15>b15 or R16>b16 or R17>b17 or R18>b18 or R19>b19 or R20>b20
                           or R21>b21 or R22>b22 or R23>b23 or R24>b24 or R25>b25 or R26>b26 or R27>b27 or R28>b28 or R29>b29 or R30>b30):
            C[j] = 0
            #print(j,C)
            R1 -= data["r1j"][j]
            R2 -= data["r2j"][j]
            R3 -= data["r3j"][j]
            R4 -= data["r4j"][j]
            R5 -= data["r5j"][j]
            R6 -= data["r6j"][j]
            R7 -= data["r7j"][j]
            R8 -= data["r8j"][j]
            R9 -= data["r9j"][j]
            R10 -= data["r10j"][j]
            R11 -= data["r11j"][j]
            R12 -= data["r12j"][j]
            R13 -= data["r13j"][j]
            R14 -= data["r14j"][j]
            R15 -= data["r15j"][j]
            R16 -= data["r16j"][j]
            R17 -= data["r17j"][j]
            R18 -= data["r18j"][j]
            R19 -= data["r19j"][j]
            R20 -= data["r20j"][j]
            R21 -= data["r21j"][j]
            R22 -= data["r22j"][j]
            R23 -= data["r23j"][j]
            R24 -= data["r24j"][j]
            R25 -= data["r25j"][j]
            R26 -= data["r26j"][j]
            R27 -= data["r27j"][j]
            R28 -= data["r28j"][j]
            R29 -= data["r29j"][j]
            R30 -= data["r30j"][j]
            
    for j in reversed(sorted_index):
        if (C[j] == 0):
            if (R1 + data["r1j"][j] <= b1) and\
                (R2 + data["r2j"][j] <= b2) and\
                (R3 + data["r3j"][j] <= b3) and\
                (R4 + data["r4j"][j] <= b4) and\
                (R5 + data["r5j"][j] <= b5) and\
                (R6 + data["r6j"][j] <= b6) and\
                (R7 + data["r7j"][j] <= b7) and\
                (R8 + data["r8j"][j] <= b8) and\
                (R9 + data["r9j"][j] <= b9) and\
                (R10 + data["r10j"][j] <= b10) and\
                (R11 + data["r11j"][j] <= b11) and\
                (R12 + data["r12j"][j] <= b12) and\
                (R13 + data["r13j"][j] <= b13) and\
                (R14 + data["r14j"][j] <= b14) and\
                (R15 + data["r15j"][j] <= b15) and\
                (R16 + data["r16j"][j] <= b16) and\
                (R17 + data["r17j"][j] <= b17) and\
                (R18 + data["r18j"][j] <= b18) and\
                (R19 + data["r19j"][j] <= b19) and\
                (R20 + data["r20j"][j] <= b20) and\
                (R21 + data["r21j"][j] <= b21) and\
                (R22 + data["r22j"][j] <= b22) and\
                (R23 + data["r23j"][j] <= b23) and\
                (R24 + data["r24j"][j] <= b24) and\
                (R25 + data["r25j"][j] <= b25) and\
                (R26 + data["r26j"][j] <= b26) and\
                (R27 + data["r27j"][j] <= b27) and\
                (R28 + data["r28j"][j] <= b28) and\
                (R29 + data["r29j"][j] <= b29) and\
                (R30 + data["r30j"][j] <= b30):
                
                C[j] = 1
                R1 += data["r1j"][j]
                R2 += data["r2j"][j]
                R3 += data["r3j"][j]
                R4 += data["r4j"][j]
                R5 += data["r5j"][j]
                R6 += data["r6j"][j]
                R7 += data["r7j"][j]
                R8 += data["r8j"][j]
                R9 += data["r9j"][j]
                R10 += data["r10j"][j]
                R11 += data["r11j"][j]
                R12 += data["r12j"][j]
                R13 += data["r13j"][j]
                R14 += data["r14j"][j]
                R15 += data["r15j"][j]
                R16 += data["r16j"][j]
                R17 += data["r17j"][j]
                R18 += data["r18j"][j]
                R19 += data["r19j"][j]
                R20 += data["r20j"][j]
                R21 += data["r21j"][j]
                R22 += data["r22j"][j]
                R23 += data["r23j"][j]
                R24 += data["r24j"][j]
                R25 += data["r25j"][j]
                R26 += data["r26j"][j]
                R27 += data["r27j"][j]
                R28 += data["r28j"][j]
                R29 += data["r29j"][j]
                R30 += data["r30j"][j]
    return C
    
def init_pop(j,N,data):

    population = [] ### store the inititalise population
    b1 = data["bi"][0]
    b2 = data["bi"][1]
    b3 = data["bi"][2]
    b4 = data["bi"][3]
    b5 = data["bi"][4]
    b6 = data["bi"][5]
    b7 = data["bi"][6]
    b8 = data["bi"][7]
    b9 = data["bi"][8]
    b10 = data["bi"][9]
    b11 = data["bi"][10]
    b12 = data["bi"][11]
    b13 = data["bi"][12]
    b14 = data["bi"][13]
    b15 = data["bi"][14]
    b16 = data["bi"][15]
    b17 = data["bi"][16]
    b18 = data["bi"][17]
    b19 = data["bi"][18]
    b20 = data["bi"][19]
    b21 = data["bi"][20]
    b22 = data["bi"][21]
    b23 = data["bi"][22]
    b24 = data["bi"][23]
    b25 = data["bi"][24]
    b26 = data["bi"][25]
    b27 = data["bi"][26]
    b28 = data["bi"][27]
    b29 = data["bi"][28]
    b30 = data["bi"][29]

    for k in range(N): ### Make 100 populations
        s = np.zeros(j) ### initiate all zeros with shape = 100
        t = np.arange(j) ### add all the index to dummy set
        index = np.random.choice(t, 1, replace=False) ### randomly select an element and change to 1
        t = t[t!=index]
        R1_rij = 0
        R2_rij = 0
        R3_rij = 0
        R4_rij = 0
        R5_rij = 0
        R6_rij = 0
        R7_rij = 0
        R8_rij = 0
        R9_rij = 0
        R10_rij = 0
        R11_rij = 0
        R12_rij = 0
        R13_rij = 0
        R14_rij = 0
        R15_rij = 0
        R16_rij = 0
        R17_rij = 0
        R18_rij = 0
        R19_rij = 0
        R20_rij = 0
        R21_rij = 0
        R22_rij = 0
        R23_rij = 0
        R24_rij = 0
        R25_rij = 0
        R26_rij = 0
        R27_rij = 0
        R28_rij = 0
        R29_rij = 0
        R30_rij = 0
        
        while True:
            if (R1_rij + data["r1j"][index[0]] <= b1) and\
                (R2_rij + data["r2j"][index[0]] <= b2) and\
                (R3_rij + data["r3j"][index[0]] <= b3) and\
                (R4_rij + data["r4j"][index[0]] <= b4) and\
                (R5_rij + data["r5j"][index[0]] <= b5) and\
                (R6_rij + data["r6j"][index[0]] <= b6) and\
                (R7_rij + data["r7j"][index[0]] <= b7) and\
                (R8_rij + data["r8j"][index[0]] <= b8) and\
                (R9_rij + data["r9j"][index[0]] <= b9) and\
                (R10_rij + data["r10j"][index[0]] <= b10) and\
                (R11_rij + data["r11j"][index[0]] <= b11) and\
                (R12_rij + data["r12j"][index[0]] <= b12) and\
                (R13_rij + data["r13j"][index[0]] <= b13) and\
                (R14_rij + data["r14j"][index[0]] <= b14) and\
                (R15_rij + data["r15j"][index[0]] <= b15) and\
                (R16_rij + data["r16j"][index[0]] <= b16) and\
                (R17_rij + data["r17j"][index[0]] <= b17) and\
                (R18_rij + data["r18j"][index[0]] <= b18) and\
                (R19_rij + data["r19j"][index[0]] <= b19) and\
                (R20_rij + data["r20j"][index[0]] <= b20) and\
                (R21_rij + data["r21j"][index[0]] <= b21) and\
                (R22_rij + data["r22j"][index[0]] <= b22) and\
                (R23_rij + data["r23j"][index[0]] <= b23) and\
                (R24_rij + data["r24j"][index[0]] <= b24) and\
                (R25_rij + data["r25j"][index[0]] <= b25) and\
                (R26_rij + data["r26j"][index[0]] <= b26) and\
                (R27_rij + data["r27j"][index[0]] <= b27) and\
                (R28_rij + data["r28j"][index[0]] <= b28) and\
                (R29_rij + data["r29j"][index[0]] <= b29) and\
                (R30_rij + data["r30j"][index[0]] <= b30):

                R1_rij += data["r1j"][index[0]]
                R2_rij += data["r2j"][index[0]]
                R3_rij += data["r3j"][index[0]]
                R4_rij += data["r4j"][index[0]]
                R5_rij += data["r5j"][index[0]]
                R6_rij += data["r6j"][index[0]]
                R7_rij += data["r7j"][index[0]]
                R8_rij += data["r8j"][index[0]]
                R9_rij += data["r9j"][index[0]]
                R10_rij += data["r10j"][index[0]]
                R11_rij += data["r11j"][index[0]]
                R12_rij += data["r12j"][index[0]]
                R13_rij += data["r13j"][index[0]]
                R14_rij += data["r14j"][index[0]]
                R15_rij += data["r15j"][index[0]]
                R16_rij += data["r16j"][index[0]]
                R17_rij += data["r17j"][index[0]]
                R18_rij += data["r18j"][index[0]]
                R19_rij += data["r19j"][index[0]]
                R20_rij += data["r20j"][index[0]]
                R21_rij += data["r21j"][index[0]]
                R22_rij += data["r22j"][index[0]]
                R23_rij += data["r23j"][index[0]]
                R24_rij += data["r24j"][index[0]]
                R25_rij += data["r25j"][index[0]]
                R26_rij += data["r26j"][index[0]]
                R27_rij += data["r27j"][index[0]]
                R28_rij += data["r28j"][index[0]]
                R29_rij += data["r29j"][index[0]]
                R30_rij += data["r30j"][index[0]]
                s[index] = 1
            else:
                s[index] = 0
                break
            index = np.random.choice(t,1,replace=False)
            t = t[t!=index]
        population.append(s)
    return population
    
def crossover_operator(P1,P2): ### uniform crossover -> 2 parents give 1 child
    child = np.zeros(len(P1))
    prob = np.random.randint(2,size=len(P1))
    for i in range(len(P1)):
        if prob[i] == 0:
            child[i] = P1[i]
        else:
            child[i] = P2[i]
    return child
    
def mutation_operator(child): ### two bits per child string
    index = np.random.choice(child.shape[0], 2, replace=False)
    #print("Index to mutate",index)
    if child[index[0]] == 0:
        child[index[0]] = 1
    else:
        child[index[0]] = 0
        
    if child[index[1]] == 0:
        child[index[1]] = 1
    else:
        child[index[1]] = 0
    #print("After_mutation", child)
    return child

def fitness_function(data,population): ### need to include constraint??
    fitness = []
    pj = data["pj"]
    b1 = data["bi"][0]
    b2 = data["bi"][1]
    b3 = data["bi"][2]
    b4 = data["bi"][3]
    b5 = data["bi"][4]
    b6 = data["bi"][5]
    b7 = data["bi"][6]
    b8 = data["bi"][7]
    b9 = data["bi"][8]
    b10 = data["bi"][9]
    b11 = data["bi"][10]
    b12 = data["bi"][11]
    b13 = data["bi"][12]
    b14 = data["bi"][13]
    b15 = data["bi"][14]
    b16 = data["bi"][15]
    b17 = data["bi"][16]
    b18 = data["bi"][17]
    b19 = data["bi"][18]
    b20 = data["bi"][19]
    b21 = data["bi"][20]
    b22 = data["bi"][21]
    b23 = data["bi"][22]
    b24 = data["bi"][23]
    b25 = data["bi"][24]
    b26 = data["bi"][25]
    b27 = data["bi"][26]
    b28 = data["bi"][27]
    b29 = data["bi"][28]
    b30 = data["bi"][29]

    for i in range(len(population)):
        profit = 0
        constraint_1 = 0
        constraint_2 = 0
        constraint_3 = 0
        constraint_4 = 0
        constraint_5 = 0
        constraint_6 = 0
        constraint_7 = 0
        constraint_8 = 0
        constraint_9 = 0
        constraint_10 = 0
        constraint_11 = 0
        constraint_12 = 0
        constraint_13 = 0
        constraint_14 = 0
        constraint_15 = 0
        constraint_16 = 0
        constraint_17 = 0
        constraint_18 = 0
        constraint_19 = 0
        constraint_20 = 0
        constraint_21 = 0
        constraint_22 = 0
        constraint_23 = 0
        constraint_24 = 0
        constraint_25 = 0
        constraint_26 = 0
        constraint_27 = 0
        constraint_28 = 0
        constraint_29 = 0
        constraint_30 = 0
        for j in range(len(population[i])):
            if population[i][j] == 1:
                profit += pj[j]
                constraint_1 += data["r1j"][j]
                constraint_2 += data["r2j"][j]
                constraint_3 += data["r3j"][j]
                constraint_4 += data["r4j"][j]
                constraint_5 += data["r5j"][j]
                constraint_6 += data["r6j"][j]
                constraint_7 += data["r7j"][j]
                constraint_8 += data["r8j"][j]
                constraint_9 += data["r9j"][j]
                constraint_10 += data["r10j"][j]
                constraint_11 += data["r11j"][j]
                constraint_12 += data["r12j"][j]
                constraint_13 += data["r13j"][j]
                constraint_14 += data["r14j"][j]
                constraint_15 += data["r15j"][j]
                constraint_16 += data["r16j"][j]
                constraint_17 += data["r17j"][j]
                constraint_18 += data["r18j"][j]
                constraint_19 += data["r19j"][j]
                constraint_20 += data["r20j"][j]
                constraint_21 += data["r21j"][j]
                constraint_22 += data["r22j"][j]
                constraint_23 += data["r23j"][j]
                constraint_24 += data["r24j"][j]
                constraint_25 += data["r25j"][j]
                constraint_26 += data["r26j"][j]
                constraint_27 += data["r27j"][j]
                constraint_28 += data["r28j"][j]
                constraint_29 += data["r29j"][j]
                constraint_30 += data["r30j"][j]
        fitness.append(profit)
    return fitness
    
def child_fitness_function(data,child,print_constraint = False):
    fitness = 0
    pj = data["pj"]
    b1 = data["bi"][0]
    b2 = data["bi"][1]
    b3 = data["bi"][2]
    b4 = data["bi"][3]
    b5 = data["bi"][4]
    b6 = data["bi"][5]
    b7 = data["bi"][6]
    b8 = data["bi"][7]
    b9 = data["bi"][8]
    b10 = data["bi"][9]
    b11 = data["bi"][10]
    b12 = data["bi"][11]
    b13 = data["bi"][12]
    b14 = data["bi"][13]
    b15 = data["bi"][14]
    b16 = data["bi"][15]
    b17 = data["bi"][16]
    b18 = data["bi"][17]
    b19 = data["bi"][18]
    b20 = data["bi"][19]
    b21 = data["bi"][20]
    b22 = data["bi"][21]
    b23 = data["bi"][22]
    b24 = data["bi"][23]
    b25 = data["bi"][24]
    b26 = data["bi"][25]
    b27 = data["bi"][26]
    b28 = data["bi"][27]
    b29 = data["bi"][28]
    b30 = data["bi"][29]
    
    profit = 0
    constraint_1 = 0
    constraint_2 = 0
    constraint_3 = 0
    constraint_4 = 0
    constraint_5 = 0
    constraint_6 = 0
    constraint_7 = 0
    constraint_8 = 0
    constraint_9 = 0
    constraint_10 = 0
    constraint_11 = 0
    constraint_12 = 0
    constraint_13 = 0
    constraint_14 = 0
    constraint_15 = 0
    constraint_16 = 0
    constraint_17 = 0
    constraint_18 = 0
    constraint_19 = 0
    constraint_20 = 0
    constraint_21 = 0
    constraint_22 = 0
    constraint_23 = 0
    constraint_24 = 0
    constraint_25 = 0
    constraint_26 = 0
    constraint_27 = 0
    constraint_28 = 0
    constraint_29 = 0
    constraint_30 = 0
    
    for i in range(len(child)):
        if child[i] == 1:
            profit += pj[i]
            constraint_1 += data["r1j"][i]
            constraint_2 += data["r2j"][i]
            constraint_3 += data["r3j"][i]
            constraint_4 += data["r4j"][i]
            constraint_5 += data["r5j"][i]
            constraint_6 += data["r6j"][i]
            constraint_7 += data["r7j"][i]
            constraint_8 += data["r8j"][i]
            constraint_9 += data["r9j"][i]
            constraint_10 += data["r10j"][i]
            constraint_11 += data["r11j"][i]
            constraint_12 += data["r12j"][i]
            constraint_13 += data["r13j"][i]
            constraint_14 += data["r14j"][i]
            constraint_15 += data["r15j"][i]
            constraint_16 += data["r16j"][i]
            constraint_17 += data["r17j"][i]
            constraint_18 += data["r18j"][i]
            constraint_19 += data["r19j"][i]
            constraint_20 += data["r20j"][i]
            constraint_21 += data["r21j"][i]
            constraint_22 += data["r22j"][i]
            constraint_23 += data["r23j"][i]
            constraint_24 += data["r24j"][i]
            constraint_25 += data["r25j"][i]
            constraint_26 += data["r26j"][i]
            constraint_27 += data["r27j"][i]
            constraint_28 += data["r28j"][i]
            constraint_29 += data["r29j"][i]
            constraint_30 += data["r30j"][i]
    if print_constraint == True:
        print("Limit_1: %s ; Child_constraint_1: %s" %(b1,constraint_1))
        print("Limit_2: %s ; Child_constraint_2: %s" %(b2,constraint_2))
        print("Limit_3: %s ; Child_constraint_3: %s" %(b3,constraint_3))
        print("Limit_4: %s ; Child_constraint_4: %s" %(b4,constraint_4))
        print("Limit_5: %s ; Child_constraint_5: %s" %(b5,constraint_5))
        print("Limit_6: %s ; Child_constraint_6: %s" %(b6,constraint_6))
        print("Limit_7: %s ; Child_constraint_7: %s" %(b7,constraint_7))
        print("Limit_8: %s ; Child_constraint_8: %s" %(b8,constraint_8))
        print("Limit_9: %s ; Child_constraint_9: %s" %(b9,constraint_9))
        print("Limit_10: %s ; Child_constraint_10: %s" %(b10,constraint_10))
        print("Limit_11: %s ; Child_constraint_11: %s" %(b11,constraint_11))
        print("Limit_12: %s ; Child_constraint_12: %s" %(b12,constraint_12))
        print("Limit_13: %s ; Child_constraint_13: %s" %(b13,constraint_13))
        print("Limit_14: %s ; Child_constraint_14: %s" %(b14,constraint_14))
        print("Limit_15: %s ; Child_constraint_15: %s" %(b15,constraint_15))
        print("Limit_16: %s ; Child_constraint_16: %s" %(b16,constraint_16))
        print("Limit_17: %s ; Child_constraint_17: %s" %(b17,constraint_17))
        print("Limit_18: %s ; Child_constraint_18: %s" %(b18,constraint_18))
        print("Limit_19: %s ; Child_constraint_19: %s" %(b19,constraint_19))
        print("Limit_20: %s ; Child_constraint_20: %s" %(b20,constraint_20))
        print("Limit_21: %s ; Child_constraint_21: %s" %(b21,constraint_21))
        print("Limit_22: %s ; Child_constraint_22: %s" %(b22,constraint_22))
        print("Limit_23: %s ; Child_constraint_23: %s" %(b23,constraint_23))
        print("Limit_24: %s ; Child_constraint_24: %s" %(b24,constraint_24))
        print("Limit_25: %s ; Child_constraint_25: %s" %(b25,constraint_25))
        print("Limit_26: %s ; Child_constraint_26: %s" %(b26,constraint_26))
        print("Limit_27: %s ; Child_constraint_27: %s" %(b27,constraint_27))
        print("Limit_28: %s ; Child_constraint_28: %s" %(b28,constraint_28))
        print("Limit_29: %s ; Child_constraint_29: %s" %(b29,constraint_29))
        print("Limit_30: %s ; Child_constraint_30: %s" %(b30,constraint_30))
    fitness = profit
    return fitness

def GA_MKP(j,N,data,sorted_index,best_opt):
    population = init_pop(j,N,data)
    fitness = fitness_function(data,population)
    S_star = fitness.index(max(fitness)) ### Find the best individual
    best_fitness = max(fitness)
    t = len(population)
    print("Length of population",t)
    worst_fitness = min(fitness)
    tmax = 2000000
    
    while t < tmax:
        valid = False
        while not valid:
            P1 = population[np.random.choice(len(population),1)[0]]
            P2 = population[np.random.choice(len(population),1)[0]]
            C = crossover_operator(P1,P2)
            C = mutation_operator(C)
            C = repair_operator(data,C,sorted_index)
            check_all = []
            for i in population:
                check = (C == i) ### check element-wise
                check_all.append(np.all(check)) ### check if all true
            if any(check_all) == True:
                valid = False
            else:
                valid = True
        fitness_C = child_fitness_function(data,C)
    
        S_prime = fitness.index(min(fitness)) ### steady state replacement
        if fitness_C >= worst_fitness: ### reject if child worst than original population
            population[S_prime] = C
            fitness[S_prime] = fitness_C ### update fitness
        worst_fitness = min(fitness)
        if fitness_C > best_fitness:
            S_star = S_prime
            best_fitness = fitness_C
            print(best_fitness,t)
            if best_fitness >= best_opt:
                child_fitness_function(data,C,print_constraint=True)
                print("Child: ",C)
        t += 1
    return S_star,best_fitness,t

for i in range(30):
    S_star,best_fitness,t = GA_MKP(100,100,mknapcb7_data["prb"+str(i)],sorted_index[i],best_optimal_sol[i])
    print(i,best_fitness,t)
    print("----------------------------------------------------------------------------Finished Problem",i)

print("-----------------------------------------------Finished---------------------------------------------------------")
