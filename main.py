
# wandb
#import wandb

# usual
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import csv

# neat
from utils import run_qvc, test_best_agent
from NEATQuantum_v0 import *

# graph
n_nodes = 8
#random
graph = [(0,1), (0,6), (0,7), (1,2), (1,6), (2,4), (2,7), (3,7), (3,6), (4,5), (4,6), (1,5), (5,6), (5,7), (6,7)]
#ladder
#graph = [(0,1), (0,7), (1,2), (2,3), (2,7), (3,4), (3,6), (4,5), (5,6), (6,7)]
#barbell
#graph = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (2,3), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7)]
#caveman
#graph = [(0,1), (0,2), (1,2), (1,3), (1,4), (2,3), (3,5), (4,5), (4,6), (5,6), (5,7), (6,7)]

def set_graph(gra):
    global graph
    graph = gra
# with these params in works:
# QUBITS = n_nodes

# N_AGENTS = 200
# TOP_LIMIT = 5
# NUM_GENERATIONS = 100

# INITIAL_LAYERS = 2
# MUTATION_POWER = 0.0001
# WEIGHTS_MUTATION_PROB = 0.6
# ADD_CNOT_PROB = 0.3
# ADD_ROT_PROB = 0.3

# COMPATIBILITY_THRESHOLD = 0.45
# TARGET_SPECIES = 20

# population parameters
QUBITS = n_nodes
GRAPH = graph
N_AGENTS = 100
TOP_LIMIT = 5
NUM_GENERATIONS = 100

INITIAL_LAYERS = 2
MUTATION_POWER = 0.01
WEIGHTS_MUTATION_PROB = 0.6
ADD_CNOT_PROB = 0.3
ADD_ROT_PROB = 0.3

COMPATIBILITY_THRESHOLD = 0.45
TARGET_SPECIES = 20

# CVaR loss func (False; otherwise in (0, 1]=True )
ALPHA = 0.0
#ALPHA = 0.3
# change mut_power
DYNAMIC_MUT_POWER = True


# options
Options.set_options(num_inputs=QUBITS,
                    graph = GRAPH,
                    population_size=N_AGENTS,
                    
                    num_initial_layers=INITIAL_LAYERS,
                    
                    weight_mutate_prob=WEIGHTS_MUTATION_PROB,
                    weight_mutate_power=MUTATION_POWER,
                    
                    add_cnot_prob=ADD_CNOT_PROB,
                    add_rot_prob=ADD_ROT_PROB,
        
                    compatibility_threshold=COMPATIBILITY_THRESHOLD,
                    target_species=TARGET_SPECIES,
                    
                    num_generations=NUM_GENERATIONS,
                    
                    dynamic_mut_power=DYNAMIC_MUT_POWER,
                    
                    cvar=ALPHA,
                    
                    # info
                    show_best_distribution=False,
                    show_elite=False, # False or int
                   )
                   
                   
                   
mean_scores = []     
best_fitness = []              
                   
def evaluate_main(agents, graph):
    scores = []
    global mean_scores
    global best_fitness
    # for every agent
    for i, agent in enumerate(agents):
        # save score
        score = run_qvc(agent, graph, alpha=ALPHA)
        # append to the list
        scores.append(score)
        # set the fitness of the agent
        agent.fitness = score
        
        # info
        print("{}%".format(100*i/len(agents)), end = '\r')
            
    # convert to np array
    scores = np.array(scores)
    # sort and get the indices
    sorted_indices = np.argsort(scores)
    # get the scores of the top elite
    top_limit_score = sum(scores[sorted_indices[-TOP_LIMIT:]])/TOP_LIMIT
    # best agent
    best_agent = agents[sorted_indices[-1]]
    # number of gates in the best agent
    n_rots = len(best_agent.gates['rot'])
    n_cnots = len(best_agent.gates['cnot'])
    
    ### show the best agent (print circuit)
    drawer = qml.draw(best_agent.qnode)
    print(drawer([0 for i in range(Options.num_inputs)]))

    
    ### save innovation table from the final circuit
    with open ('FinalCircuit.csv', 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerow(best_agent.gates['rot'])
        writer.writerow(best_agent.gates['cnot'])
    
    # test best agent
    average_energy = test_best_agent(best_agent, n_nodes, graph)
    
    print(n_rots, n_cnots)
    
    #wandb.log(
    #        {
    #            'Score': average_energy,
    #            'n. Rots': n_rots,
    #            'n. Cnots' : n_cnots,
    #            'n. gates' : n_rots + n_cnots
    #        }
    #)

    print('Average score of population: ', scores.mean(), ' Avg top {}: '.format(TOP_LIMIT), top_limit_score, ' Best {}'.format(best_agent.fitness))
    mean = scores.mean()
    mean_scores.append(mean)
    
    ### make sure that the values for best_agent.fitness are saved in the correct format
    if type(best_agent.fitness) == type(2.7):
        fit = best_agent.fitness
    else:
        fit = best_agent.fitness.item()
    best_fitness.append(fit)

    ### save mean score
    # with open ('MeanScore.csv', 'w', newline='') as file1:
    #     writer = csv.writer(file1)
    #     writer.writerow([mean_scores])

    ### save best fitness    
    # with open ('BestFitness.csv', 'w', newline='') as file2:
    #     writer = csv.writer(file2)
    #     writer.writerow([best_fitness])    
    
# main
### the original QNEAT version does not contain a 'real' main function, but it is necessary to wrap it for TL and slurm
def mainfunction(graph):
#for ALPHA in [False, 0.3]:
    for ALPHA in [False]:
        #for MUTATION_POWER in [0.0001, 0.001, 0.01]:
        for MUTATION_POWER in [0.01]:    
            # options
            Options.set_options(num_inputs=QUBITS,
                        graph=graph,
                        population_size=N_AGENTS,
                    
                        num_initial_layers=INITIAL_LAYERS,
                    
                        weight_mutate_prob=WEIGHTS_MUTATION_PROB,
                        weight_mutate_power=MUTATION_POWER,
                    
                        add_cnot_prob=ADD_CNOT_PROB,
                        add_rot_prob=ADD_ROT_PROB,
        
                        compatibility_threshold=COMPATIBILITY_THRESHOLD,
                        target_species=TARGET_SPECIES,
                    
                        num_generations=NUM_GENERATIONS,
                    
                        dynamic_mut_power=DYNAMIC_MUT_POWER,
                    
                        cvar=ALPHA,
                    
                        # info
                        show_best_distribution=False,
                        show_elite=False, # False or int
                    )

            for i in range(1):
            #    run = wandb.init(
            #        project='Final-Experiment-Comb-Optim-QNEAT',
            #        entity='s-egger',
            #        config={
#
            #            'num_inputs': QUBITS,
            #            'graph': GRAPH,
            #            'population_size': N_AGENTS,

            #            'num_initial_layers': INITIAL_LAYERS,

#                        'weight_mutate_prob': WEIGHTS_MUTATION_PROB,
#                        'weight_mutate_power': MUTATION_POWER,

#                        'add_cnot_prob': ADD_CNOT_PROB,
#                        'add_rot_prob': ADD_ROT_PROB,

#                        'compatibility_threshold': COMPATIBILITY_THRESHOLD,
#                        'target_species': TARGET_SPECIES,

#                        'alpha': ALPHA,

#                        'dynamic_mut_power': DYNAMIC_MUT_POWER,
#                    },

#                    reinit=True,
#                )
#                run.name = 'QNEAT/Alpha{}/MutPower{}'.format(ALPHA, MUTATION_POWER)

                p = Population()

                best = p.evaluate(evaluate_main, graph)

#     run.finish()
#mainfunction(graph=graph)