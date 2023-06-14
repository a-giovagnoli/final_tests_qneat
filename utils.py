


import numpy as np
import gym
import matplotlib.pyplot as plt
import math
import csv

# convert the array of measurements e.g. [0,1,1,0] into a string 0110
def bitstring_to_int(bit_string_sample):
    # transform tensor of measurements into a list
    bit_string_sample = bit_string_sample.tolist()
    
    # join the elements of the list to make a string
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    
    return int(bit_string, base=2)


def run_qvc(qvc, graph, alpha=0):
    """
    evaluate qvc
    """   
    # alpha for CVaR
    if alpha:
        single_edge_samples = []
        for edge in graph:
            prediction = qvc.predict(edge)
            single_edge_samples.append(prediction)
        hamiltonians = []
        for list_of_measurements in zip(single_edge_samples):
            
            hamiltonians.append( sum( [0.5 * (1- z_iz_j) for z_iz_j in list_of_measurements] ).item() )
            
        
        hamiltonians.sort(reverse=True)


        # CVaR
        alphaK = math.ceil(alpha * len(hamiltonians))
        score = 1/alphaK * sum([hamiltonians[i] for i in range( alphaK )])
        

    else:
        obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            h_average = qvc.predict(edge)
            obj += 0.5 * (1 - h_average)

        score = obj

    return score


def hamiltonian(config, graph):
    
    H = 0
    for i, j in graph:
        
        zi = config[i]
        zj = config[j]
        H += 0.5 * (1 - zi*zj)
    
    return H

avg = []

def test_best_agent(qvc, num_nodes, graph):
    
    # sample measured bitstrings 100 times
    bit_strings = []
    hamiltonians = []
    n_samples = 100
    
    simple_measurements = []
    for i in range(0, n_samples):
        measurements = qvc.predict(None, test=True)
        simple_measurements.append(measurements)
        bit_strings.append(bitstring_to_int(measurements))
        
    samples_energies = []
    for m in simple_measurements:
        m = m.tolist()
        m = [int(x) if int(x)==1 else -1 for x in m]
        samples_energies.append(hamiltonian(m, graph))
       
    average_energy = sum(samples_energies)/len(samples_energies)
    print('average energy:', average_energy)
    global avg
    avg.append(average_energy)
    #with open ('Caveman3TestAverageEnergy.csv', 'w', newline='') as file:
     #   writer = csv.writer(file)
      #  writer.writerow(avg)   

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    
    print("Most frequently sampled bit string is: {:08b}".format(most_freq_bit_string))
    
#    with open ('RandomTestBitstring.csv', 'w', newline='') as file1:
#        writer = csv.writer(file1)
#        writer.writerow([most_freq_bit_string])
    
    xticks = range(0, 2**num_nodes)
    xtick_labels = list(map(lambda x: format(x, "0{}b".format(num_nodes)), xticks))
    
    xtick_plus_minus_1 = []
    for label in xtick_labels:
        arr = list(label)
        arr = [int(x) if int(x)==1 else -1 for x in arr]
        xtick_plus_minus_1.append(arr)
    
    energies = []
    for tick in xtick_plus_minus_1:
        energies.append(hamiltonian(tick, graph))
        
                            

    final_xticks_labels = []
    for i in range(len(energies)):
        bits = ''.join(str(x) for x in xtick_plus_minus_1[i])
        final_xticks_labels.append(bits + " - " + str(energies[i]))

            
    bins = np.arange(0, 2**num_nodes + 1) - 0.5

    # Write on top of bar: https://www.tutorialspoint.com/how-to-write-text-above-the-bars-on-a-bar-plot-python-matplotlib

    #from matplotlib.pyplot import figure
    #figure(figsize=(10, 6))

    #plt.title("n_layers=1")
    #plt.xlabel("bitstrings")
    #plt.ylabel("freq.")
    #plt.xticks(xticks, final_xticks_labels, rotation="vertical", fontsize=4)
    #plt.hist(bit_strings, bins=bins)
    #plt.show()


    
    
def scan_best_agent(env, pool):
    """
    Find the best agent and evaluate it many times to find its distribution
    """
    
    # find best agent
    scores = [float(brain.fitness) for brain in pool]
    indices = np.argsort(np.array(scores))[-1:]
    print('Best score: ', np.array(scores)[indices][0])
    # select it
    agent = np.array(pool)[indices][0]
    
    # evaluate it 100 times
    scores_agent = run_episode(env, agent, 100)
    
    # plot the distribution
    print('Evaluated again: ', sum(scores_agent)/len(scores_agent))
    
    # calculate average score
    plt.hist(scores_agent, bins = 100, range=[0, 500])
    plt.title('Best agent averaged 100 times')
    plt.show()
