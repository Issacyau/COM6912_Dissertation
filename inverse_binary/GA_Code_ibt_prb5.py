import numpy as np
import itertools
from scipy.optimize import minimize

best_optimal_sol = [59187,58662,58094,61000,58092,58803,58607,58917,59384,59193,110863,108659,108932,110037,108423,110841,106075,106686,109825,106723,151790,148772,151900,151275,151948,152109,153131,153520,149155,149704]

def load_file_2(file):
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

        data_prb["number_variables_problem"] = n
        data_prb["number_constraint_problem"] = m
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
        data_prb["bi"] = x[3+11*n:]
        data["prb"+str(prb)] = data_prb
    return number_test_problems,data
    
num_test_prb_5,mknapcb5_data = load_file_2("mknapcb5.txt") ### loaded the first problem file
print(len(mknapcb5_data),len(best_optimal_sol))

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
                      data["prb"+str(i)]["r6j"],data["prb"+str(i)]["r7j"],data["prb"+str(i)]["r8j"],data["prb"+str(i)]["r9j"],data["prb"+str(i)]["r10j"]]
        capacities = data["prb"+str(i)]["bi"]
        N = 250
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
        uj = data["prb"+str(i)]["pj"]/(u1+u2+u3+u4+u5+u6+u7+u8+u9+u10)
        sorted_index = np.argsort(uj)
        sort_index.append(sorted_index)
    return sort_index
    
sorted_index = sort_index(mknapcb5_data)

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

    
    for j in (sorted_index):
        if (C[j] == 1) and (R1>b1 or R2>b2 or R3>b3 or R4>b4 or R5>b5 or R6>b6 or R7>b7 or R8>b8 or R9>b9 or R10>b10):
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
                (R10 + data["r10j"][j] <= b10):
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
    return C
    
def init_pop(j,N,data):

    population = [] ### store the inititalise population
    b1 = data["bi"][0] ### Constraint 1
    b2 = data["bi"][1] ### Constraint 2
    b3 = data["bi"][2] ### Constraint 3
    b4 = data["bi"][3] ### Constraint 4
    b5 = data["bi"][4] ### Constraint 5
    b6 = data["bi"][5]
    b7 = data["bi"][6]
    b8 = data["bi"][7]
    b9 = data["bi"][8]
    b10 = data["bi"][9]
    

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
                (R10_rij + data["r10j"][index[0]] <= b10):

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
    
def inverse_binary_tournament_selection(population,fitness): ### inverse binary selection
    best = None
    for i in range(2):
        ind = np.random.choice(len(population),1)[0]
        if (best == None) or (fitness[ind] < fitness[best]): ### (fitness(ind) > fitness(best))
            best = ind
    return population[best]

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
    fitness = profit
    return fitness

def GA_MKP(j,N,data,sorted_index,best_opt):
    population = init_pop(j,N,data)
    fitness = fitness_function(data,population)
    S_star = fitness.index(max(fitness)) ### Find the best individual
    best_fitness = max(fitness)
    worst_fitness = min(fitness)
    t = len(population)
    print("Length of population",t)
    tmax = 2000000
    
    while t < tmax:
        valid = False
        while not valid:
            P1 = inverse_binary_tournament_selection(population,fitness)
            P2 = inverse_binary_tournament_selection(population,fitness)
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
    S_star,best_fitness,t = GA_MKP(250,100,mknapcb5_data["prb"+str(i)],sorted_index[i],best_optimal_sol[i])
    print(i,best_fitness,t)
    print("----------------------------------------------------------------------------Finished Problem",i)

print("-----------------------------------------------Finished---------------------------------------------------------")
