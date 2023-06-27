# here there is the CVaR loss function
# also only one shor for each run of the circuit

import random
import math
import copy
import numpy as np
import pennylane as qml
#import gym
import matplotlib.pyplot as plt
import csv

from utils import scan_best_agent

random.seed()


class Options:
    @staticmethod
    def set_options(
        num_inputs,
        graph,
        population_size,

        num_initial_layers=1,
        
        max_fitness=float('inf'),
        
        max_rots=float('inf'),
        max_cnots=float('inf'),

        disjoint_coeff=1,
        weight_coeff=0.5,

        add_rot_prob=0.02,
        add_cnot_prob=0.02,

        weight_mutate_prob=0.1,
        new_weight_prob=0.1,
        weight_init_range=np.pi,
        weight_mutate_power=0.5,

        feature_selection=False,

        compatibility_threshold=1,
        dynamic_compatibility_threshold=True,

        target_species=15,
        dropoff_age=300,
        survival_rate=0.2,
        species_elitism=True,

        crossover_rate=1,
        tries_tournament_selection=3,

        young_age_threshhold=5,
        young_age_fitness_bonus=1.3,
        old_age_threshold=15,
        old_age_fitness_penalty=0.7,
        
        num_generations=100,
        
        dynamic_mut_power=False,
        
        cvar=True,
        
                
        
        # info
        show_best_distribution=False,
        show_elite = 5,
        
    ):
        Options.num_inputs = num_inputs
        Options.graph = graph
        Options.population_size = population_size
        
        Options.num_initial_layers = num_initial_layers

        Options.max_fitness = max_fitness
        Options.max_rots = max_rots
        Options.max_cnots = max_cnots

        Options.disjoint_coeff = disjoint_coeff
        Options.weight_coeff = weight_coeff

        Options.add_cnot_prob = add_cnot_prob
        Options.add_rot_prob = add_rot_prob

        Options.weight_mutate_prob = weight_mutate_prob
        Options.new_weight_prob = new_weight_prob
        Options.weight_init_range = weight_init_range
        Options.weight_mutate_power = weight_mutate_power
        
        Options.initial_weight_mutate_power = weight_mutate_power

        Options.feature_selection = feature_selection

        Options.compatibility_threshold = compatibility_threshold
        Options.dynamic_compatibility_threshold = dynamic_compatibility_threshold

        Options.target_species = target_species
        Options.dropoff_age = dropoff_age
        Options.survival_rate = survival_rate
        Options.species_elitism = species_elitism

        Options.crossover_rate = crossover_rate
        Options.tries_tournament_selection = tries_tournament_selection

        Options.young_age_threshhold = young_age_threshhold
        Options.young_age_fitness_bonus = young_age_fitness_bonus
        Options.old_age_threshold = old_age_threshold
        Options.old_age_fitness_penalty = old_age_fitness_penalty
        
        Options.num_generations = num_generations
        
        Options.dynamic_mut_power = dynamic_mut_power
        
        Options.cvar = cvar
        
        # info
        Options.show_best_distribution = show_best_distribution
        Options.show_elite = show_elite



class Gate:
    # dictionaries of type: {number of the gate: layer in the circuit}
    gates_pos = {
        'rot': {},
        'cnot': {},
    }
    
    # total gates so far, incremented whenever a new gate is added
    gates_id = {
        'rot': 0,
        'cnot': 0,
    }
    
    # dictionary of type {[layer, wire]: number assigned to the gate}
    gates_hist = {
        'rot': {},
        'cnot': {},
    }
    
    @staticmethod
    def init_pos():
        for i in range(Options.num_inputs):
            Gate.gates_pos['rot'][i] = (0, i)
            Gate.gates_pos['cnot'][i] = (0, i)

        Gate.gates_id['rot'] = Options.num_inputs
        Gate.gates_id['cnot'] = Options.num_inputs

    @staticmethod
    def set_pos(gate_id, layer, wire, gate_type):
        # when a new gate is being added, it's always along the connection between two existing ones, so:
        if Gate.gates_pos[gate_type].get(gate_id) is None:
            # we save the layer and wire where it is located
            Gate.gates_pos[gate_type][gate_id] = layer, wire

    @staticmethod
    def get_innov(layer, wire, gate_type):
        # add the info
        if Gate.gates_hist[gate_type].get((layer, wire)) is None:
            Gate.gates_hist[gate_type][layer, wire] = Gate.gates_id[gate_type]
            Gate.gates_id[gate_type] += 1

        return Gate.gates_hist[gate_type][layer, wire]

def new_cnot(wire, layer=0):
    # Note: the wire of a cnot is considered to be the one where is starts
    return {
        'layer': layer,
        'wire': wire,
        'to': wire+1 if wire<(Options.num_inputs-1) else 0,
        'innov': InnovTable.get_innov(layer, wire, 'cnot'),
        'enabled': True
    }

def new_rot(wire, layer=0, weights=None):
    return {
        'layer': layer,
        'wire': wire,
        'weights': weights or [random.uniform(-1, 1)*Options.weight_init_range for _ in range(3)],
        'innov': InnovTable.get_innov(layer, wire, 'rot'),
        'enabled': True
    }
    


class InnovTable:
    
    inn_gates_id = {'rot': 0, 'cnot': 0}
    inn_gates_hist = {'rot': {}, 'cnot': {}}
    
    @staticmethod
    def get_innov(layer, wire, gate_type):
        
        if InnovTable.inn_gates_hist[gate_type].get((layer, wire)) is None:
            InnovTable.inn_gates_hist[gate_type][layer, wire] = InnovTable.inn_gates_id[gate_type]
            InnovTable.inn_gates_id[gate_type] += 1

        return InnovTable.inn_gates_hist[gate_type][layer, wire]

mut_weights = [0,0, np.pi/2]    
class Brain:   
    def __init__(self, gates=None, layers=[None]):
        self.fitness = 0
        
        self.device = qml.device('default.qubit', wires=Options.num_inputs, shots=1000)
        self.device_test = qml.device('default.qubit', wires=Options.num_inputs, shots=1)
        
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")
        self.qnode_test = qml.QNode(self._circuit, self.device_test, interface="torch")
    
        # Remember: by default I start with two initial layers available. One that is actually full of gates, the other ones is at the end and empty, but available.
        
        self.layers_indices = set(layers)
        self.gates= gates # dictionary
                
        # if no information is given
        if gates is not None:
            return

        initial_layers = [i for i in range(Options.num_initial_layers + 1)] # the +1 is due to the reminder

        self.layers_indices = set(initial_layers)
        # dictionary containing all the info on the rotations and gates
        self.gates = {
            'rot': [new_rot(wire, layer) for wire in range(Options.num_inputs) for layer in range(Options.num_initial_layers)],
            'cnot': [new_cnot(wire, layer) for wire in range(Options.num_inputs) for layer in range(Options.num_initial_layers)],
        }
        
    def _add_gate(self, gate_type):
   
        valid = []

        # check for all the valid connections
        for layer in self.layers_indices:
            for wire in range(Options.num_inputs):
                if self._valid_gate(layer, wire, gate_type):
                    valid.append((layer, wire))
                                        
        # if there is at least a possible one
        if valid:
            # pick a random one
            layer, wire = random.choice(valid)
            
            # if it's empty, there are no rules
            if len(self.gates['rot']) == 0 and len(self.gates['cnot']) == 0:
                if gate_type == 'rot':
                    self.gates[gate_type].append(new_rot(wire, layer))
                else:
                    self.gates[gate_type].append(new_cnot(wire, layer))
                        
            # if a rot has to be added
            if gate_type == 'rot':
                # check if there is a cnot in the same layer
                cnot_layers = []
                for cnot in self.gates['cnot']:
                    cnot_layers.append(cnot['layer'])
                if layer in cnot_layers:
                    self.gates[gate_type].append(new_rot(wire, layer))
                else:
                    return
                  
            # if a cnot has to be added
            elif gate_type == 'cnot':
                # check if there is a rotation in the layer before
                rot_layers = []
                for rot in self.gates['rot']:
                    rot_layers.append(rot['layer'])
                cnot_layers = []
                for cnot in self.gates['cnot']:
                    cnot_layers.append(cnot['layer'])
                if layer-1 in rot_layers:
                    self.gates[gate_type].append(new_cnot(wire, layer))
                else:
                    return
                
             
            # get the gate id depending on the history and type
            gate_id = Gate.get_innov(layer, wire, gate_type)

            # set the info on where is located
            Gate.set_pos(gate_id, layer, wire, gate_type)


            # if the last (empty) layer was selected then add a new one
            last_layer = max(self.layers_indices)
            if layer == last_layer:
                self.layers_indices.add(last_layer + 1)
                
    def set_weights(weights):
        global mut_weights
        mut_weights = weights 
    
    def mutate(self):
        # with a probability threshold, add a new rotation gate
        weight = []
        if random.random() < Options.add_rot_prob and len(self.gates['rot']) < Options.max_rots:
            self._add_gate('rot')

        # with a probability threshold, add a new cnot
        if random.random() < Options.add_cnot_prob and len(self.gates['cnot']) < Options.max_cnots:
            self._add_gate('cnot')
        for rot in self.gates['rot']:
            # with a probability threshold, change the weight, between
            if random.random() < Options.weight_mutate_prob:
                # giving a completely new one
                if random.random() < Options.new_weight_prob:
                    rot['weights'] = [random.uniform(-1, 1)*Options.weight_init_range for _ in range(3)]
                    weight = rot['weights']                # or adding on the existing one
                else:
                    rot['weights'] += np.array([random.uniform(-1, 1)*Options.weight_mutate_power for _ in range(3)])
                    weight = rot['weights']          
        global mut_weights
        mut_weights = [weight[0], weight[1], weight[2]]
        
        ## save mutation weights for the next iteration
        with open('Weights.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(mut_weights)    

    def _valid_gate(self, layer, wire, gate_type):
        
        
        # for every gate already present
        for gate in self.gates[gate_type]:
            # if it's already taken
            if gate['layer'] == layer and gate['wire'] == wire:
                # then it's not available
                return False

        # else it's available only if (this returns a product (and) between Trues and Falses):
        return (
            # here the conditions which allow a gate to be placed inside a layer
            # right now nothing comes to my mind why a gate should not be placed unless it's already taken
            True
        )
    
    def _encode(self, inputs):
        qml.BasisState(inputs, wires= [i for i in range(Options.num_inputs)])
        
#         # encode the input
#         assert len(inputs) == Options.num_inputs
        
#         for i, s in enumerate(inputs):
#             qml.RX(np.arctan(s), wires=i)

    def _feature_map(self, theta, graph):
        for edge in graph:
            wire1 = edge[0]
            wire2 = edge[1]
            qml.CNOT(wires = [wire1, wire2])
            qml.RZ(theta, wires=wire2 )            
            qml.CNOT(wires = [wire1, wire2])

    def _place_gates(self):
        
        # get all the layers of each cnot
        cnot_layers = [cnot['layer'] for cnot in self.gates['cnot']]
        
        # get all the layers of each rot
        rot_layers = [rot['layer'] for rot in self.gates['rot']]
        
        for layer in self.layers_indices:
                                        
            # CNOTS
            # find the indices of the cnots supposed to be in that layer
            cnots_indices = np.where(np.array(cnot_layers) == layer)[0] # the [0] is as before
        
            cnots_to_place = []
            # go thorugh the indices
            for cnot_index in cnots_indices:
                # save the 'from' and 'to' positions
                fr = self.gates['cnot'][cnot_index]['wire']
                to = self.gates['cnot'][cnot_index]['to']
                                
                cnots_to_place.append([fr, to])
                
            # sort the list of cnots wrt the 'from' wire
            cnots_to_place.sort(key=lambda x: x[0])
            
            # place the cnots
            for fr, to in cnots_to_place:
                qml.CNOT(wires=[fr, to])
                
            
            # ROTS
            # find the indices of the rots supposed to be in that layer
            rots_indices = np.where(np.array(rot_layers) == layer)[0] # the [0] is to get the array, since no.where returns the couple (array, something)
                        
            # go thorugh the indices
            for rot_index in rots_indices:
                # place them
                
                theta_x, theta_y, theta_z = self.gates['rot'][rot_index]['weights']
                qml.Rot(theta_x, theta_y, theta_z, wires=self.gates['rot'][rot_index]['wire'])
                
## the following circuit is the original QNEAT circuit with the additional feature map                
    def _circuit(self, inputs=None):
                  
        # there is no encoding here
#         # encode the input
#         self._encode(inputs)

    
        for i in range(Options.num_inputs):
            qml.Hadamard(wires=i)

        
       # get the edges, if there are any
        if inputs is not None:
            edge_1 = inputs[0]
            edge_2 = inputs[1]

        # encoding to generalize
        self._feature_map(mut_weights[2] , Options.graph)
        

        # apply rots and cnots
        self._place_gates()
        
        # se non vengono passati edges, allora si sta solo valutando
        if inputs is None:
            # measurement phase
           return qml.sample()
    
       
        # during the optimization phase we are evaluating the term Zi*Zj which will go into 1/2 * (1 - Zi*Zj)
        H = qml.PauliZ(edge_1) @ qml.PauliZ(edge_2)

        
        # return the expected values
        # results = qml.sample(H)
        
        if Options.cvar:
            return qml.sample(H)

        return qml.expval(H)

################################################################################################
####                                                                                        ####
####                      the following circuit is pretrained                               ####    
####                                                                                        ####    
################################################################################################

##    def _circuit(self, inputs=None):
##        for i in range(Options.num_inputs):
##            qml.Hadamard(wires=i)
##
##
##        if inputs is not None:
##          edge_1 = inputs[0]
##          edge_2 = inputs[1]
##
##       for edge in Options.graph:
##            wire1 = edge[0]
##            wire2 = edge[1]
##            qml.CNOT(wires = [wire1, wire2])
##            qml.RZ(np.pi/2, wires=wire2 )            
##            qml.CNOT(wires = [wire1, wire2])
##
##        for i in range(Options.num_inputs):
##            qml.CNOT(wires=[i, (i+1)%8])
##
##        qml.CNOT(wires = [0, 1])
##        qml.CNOT(wires = [1, 2])
##        qml.CNOT(wires = [2, 3])
##        qml.CNOT(wires = [3, 4])
##        qml.CNOT(wires = [4, 5])
##        qml.CNOT(wires = [5, 6])
##        qml.CNOT(wires = [6, 7])
##        qml.CNOT(wires = [7, 0])
##        qml.Rot(-1.81877198, -2.09880025, 1.62806948, wires=0)
##        qml.Rot(-0.85396962, -2.93316033, 1.74560563, wires=1)
##        qml.Rot(-0.89040408, -1.66499548, 1.45023039, wires=2)
##        qml.Rot(-1.34591333, 1.80183085, -1.28780316, wires=3)
##        qml.Rot(-0.33684414, -2.2518475, 0.68849814, wires=4)
##        qml.Rot(-2.09879252, -2.74982311, -1.30287686, wires=5)
##        qml.Rot(2.56501416, -0.73304696, -2.69551005, wires=6)
##        qml.Rot(-2.24185585, -2.06997122, -2.5541789, wires=7)
##        qml.CNOT(wires = [0, 1])
##        qml.CNOT(wires = [1, 2])
##        qml.CNOT(wires = [2, 3])
##        qml.CNOT(wires = [3, 4])
##        qml.CNOT(wires = [4, 5])
##        qml.CNOT(wires = [5, 6])
##        qml.CNOT(wires = [6, 7])
##        qml.CNOT(wires = [7, 0])
##        qml.Rot(-0.26389069437688056, 2.076059338085625, 3.0319820541671656, wires=0)
##        qml.Rot(-2.29650283, -1.02404911, 2.46227138, wires=1)
##        qml.Rot(-0.74759364, 0.59249434, -2.38139785, wires=2)
##        qml.Rot(2.97621168, 1.74863029, -2.35798027, wires=3)
##        qml.Rot(2.23790175, -1.37511776, 0.13180021, wires=4)
##        qml.Rot(-0.82947198, 0.27575124, -1.74537692, wires=5)
##        qml.Rot(-1.06195298, -0.76374445, -2.34961979, wires=6)
##        qml.Rot(-1.04020438, 2.26549634, -2.24423832, wires=7)
##        qml.CNOT(wires = [0, 1])
##        qml.CNOT(wires = [3, 4])
##        qml.Rot(-1.5977368944353703, 2.7178315418655656, 0.7335230171448104, wires=0)
##        qml.CNOT(wires = [1, 2])
##        qml.Rot(0.18426389, -0.18651029, 1.67065533, wires=3)
##        qml.CNOT(wires = [4, 5])
##        qml.Rot(3.04003561, -2.87929468, -1.51457137, wires=1)
##        qml.Rot(-2.69249263, -0.36376189, 0.67718989, wires=2)
##        qml.Rot(-2.17757397, 1.28063702, 0.53500865, wires=4)
##        qml.CNOT(wires = [5, 6])
##        qml.CNOT(wires = [1, 2])
##        qml.Rot(2.1097648, -0.21134067, 0.01650762, wires=4)
##        qml.Rot(-0.66616656, 2.20301045, 2.99157453, wires=5)
##        qml.CNOT(wires = [6, 7])
##        qml.CNOT(wires = [0, 1])
##        qml.Rot(2.72851749, 1.7211524, -2.42143562, wires=2)
##        qml.Rot(-2.54315627, -0.33492014, 1.27553156, wires=6)
##        qml.Rot(-0.8387802, 0.34395651, -1.85443775, wires=7)
##        qml.Rot(-0.58805529, -2.95251772, -0.87859343, wires=0)
##        qml.CNOT(wires = [1, 2])
##        qml.CNOT(wires = [5, 6])
##        qml.Rot(1.4224279879206834, 2.3212750120222347, 3.073044358025544, wires=1)
##        qml.Rot(-1.047579123384796, -3.12150501870214, 2.9226151690877993, wires=5)
##        qml.CNOT(wires = [6, 7])
##        qml.CNOT(wires = [4, 5])
##        qml.Rot(-1.99886359, 2.11186623, 0.11415383, wires=7)
##        qml.CNOT(wires = [5, 6])
##        qml.Rot(-1.41909344, -0.78769634, -3.00959879, wires=6)
##        qml.CNOT(wires = [5, 6])
##        qml.CNOT(wires = [6, 7])
##        qml.Rot(0.79597108, -1.3282444, 0.43886158, wires=6)
##        qml.CNOT(wires = [7, 0])
##        qml.Rot(-1.9279876025991163, 0.45620883459685296, 1.323207750036913, wires=0)
##        qml.Rot(-1.18421405, -3.02044304, 2.75112554, wires=4)
##        qml.CNOT(wires = [5, 6])
##        qml.Rot(2.80446356, 2.94015694, 0.88364029, wires=5)
##        if inputs is None:
##            return qml.sample()
##
##        H = qml.PauliZ(edge_1) @ qml.PauliZ(edge_2)
##
##        return qml.expval(H)       

    def predict(self, inputs, test=False):
        if test:
            return self.qnode_test()
        
        return self.qnode(inputs)
        
    
    @staticmethod
    def crossover(mom, dad):
        
        baby_layers_indices = set()
        baby_gates = {
                'rot':[],
                'cnot': [],
        }
        
        for gate_type in ['rot', 'cnot']:
        
            n_mom = len(mom.gates[gate_type])
            n_dad = len(dad.gates[gate_type])
            

            # if they have the same fitness
            if mom.fitness == dad.fitness:
                # the best one is the one with the simpler architecture
                if n_mom == n_dad:
                    better = random.choice([mom, dad])
                elif n_mom < n_dad:
                    better = mom
                else:
                    better = dad
                    

            # else simply select the one with the highest fitness
            elif mom.fitness > dad.fitness:
                better = mom
            else:
                better = dad
            
            
            # starting from 0
            i_mom = i_dad = 0

            # until one of them is still not over
            while i_mom < n_mom or i_dad < n_dad:
                mom_gate = mom.gates[gate_type][i_mom] if i_mom < n_mom else None
                dad_gate = dad.gates[gate_type][i_dad] if i_dad < n_dad else None

                selected_gate = None

                if mom_gate and dad_gate:
                    # if they have the same inn number
                    if mom_gate['innov'] == dad_gate['innov']:
                        # pick randomly
                        selected_gate = random.choice([mom_gate, dad_gate])

                        # slide on both
                        i_mom += 1
                        i_dad += 1

                    # if the mom is missing that connection/mutation
                    elif dad_gate['innov'] < mom_gate['innov']:
                        # if the dad is better
                        if dad is better:
                            # then select that connection
                            selected_gate = dad.gates[gate_type][i_dad]
                        i_dad += 1

                    # the opposite here
                    elif mom_gate['innov'] < dad_gate['innov']:
                        if mom is better:
                            selected_gate = mom_gate
                        i_mom += 1

                # if the mom has finished the connections, but the index is still sliding on the father, and the father is better, then take the father. Else take nothing.
                elif mom_gate is None and dad_gate and dad is better:
                    selected_gate = dad_gate
                    i_dad += 1

                # else same for the mom
                elif mom_gate and dad_gate is None and mom is better:
                    selected_gate = mom_gate
                    i_mom += 1

                else:
                    break

                # add the connection and the nodes to the baby
                if selected_gate is not None:
                    baby_layers_indices.add(selected_gate['layer'])
                    baby_gates[gate_type].append(copy.copy(selected_gate))
                            
        # add the extra final empty layer
        if len(baby_layers_indices):
            baby_last_layer = max(baby_layers_indices)
            baby_layers_indices.add(baby_last_layer+1)
        else:
            baby_layers_indices.add(0)
            
        return Brain(baby_gates, baby_layers_indices)
    
    def distance(b1, b2):
        weight_difference = 0
        n_disjoint = 0
            
        for gate_type in ['rot', 'cnot']:
            
            n_match = 0
    
            n_g1 = len(b1.gates[gate_type])
            n_g2 = len(b2.gates[gate_type])

            for g1 in b1.gates[gate_type]:
                for g2 in b2.gates[gate_type]:
                    if g1['innov'] == g2['innov']:
                        n_match += 1
                        if gate_type == 'rot':
                            weight_difference += abs( np.mean( np.array(g1['weights']) - np.array(g2['weights']) ) )

#             # INFO
#             print('n_g1 = {}, n_g2 = {}, n_match {} '.format(n_g1, n_g2, n_match))
            
            n_disjoint += (n_g1 - n_match) + (n_g2 - n_match)
        
        # in case it's still 0, since there is a division
        n_match += 1
        
        
        return (Options.disjoint_coeff * n_disjoint) / max([n_g1, n_g2, 1]) + Options.weight_coeff * weight_difference / n_match

class Species:
    def __init__(self, member):
        
        self.best = member
        self.pool = [member]

        self.age = 0
        self.stagnation = 0

        self.spawns_required = 0

        self.max_fitness = 0.0
        self.average_fitness = 0.0

    def purge(self):
        self.age += 1
        self.stagnation += 1
        self.pool[:] = []
        
    def get_brain(self):
        # DOUBT: What does this function actually do? It looks like it _tries_ to get the best performing member with some random trials.
        
        best = random.choice(self.pool)
        for _ in range( min(len(self.pool), Options.tries_tournament_selection) ):
            g = random.choice(self.pool)
            if g.fitness > best.fitness:
                best = g
                
        return best
    
    def cull(self):
        # DOUBT: Not sure what array[:] = array[:max()] does. How can it reshape the array like this?
        # depending of the survival_rate select the fitst n members (min 1, because this is a specie that has to survive)
        self.pool[:] = self.pool[:max(1, round(len(self.pool) * Options.survival_rate))]
    
    def adjust_fitnesses(self):
        # new species are penalised, since it takes some time to optimize them
        # so if the specie is young, the fitness of its members should be increased
        
        total = 0
        # for every agent in the specie
        for m in self.pool:
            # select fitness
            fitness = m.fitness

            # if the age of the specie is less than the age threshold
            if self.age < Options.young_age_threshhold:
                # increse the fitness of the specie, since the new ones are p
                fitness *= Options.young_age_fitness_bonus

            # otherwise give a penalty
            if self.age > Options.old_age_threshold:
                fitness *= Options.old_age_fitness_penalty

            # get average fitness
            total += fitness / len(self.pool)

        # save average fitness
        self.average_fitness = total
        
    def make_leader(self):
        # sort the agents of the species wrt to fitnesses
        self.pool.sort(key=lambda x: x.fitness, reverse=True)
        # select the best agent
        self.best = self.pool[0]

        # if there is a new leader
        if self.best.fitness > self.max_fitness:
            # if the specie produced a best score ever, it's not stagnating at all, so reset stagnation
            self.stagnation = 0
            # save max fitness
            self.max_fitness = self.best.fitness

    def same_species(self, brain):
        # gets an agent and determines the distance between him and the best agent in the current specie; depending on the compatibility returns True of False
    
#         # INFO
#         print('Distance: ', Brain.distance(brain, self.best))

        return Brain.distance(brain, self.best) <= Options.compatibility_threshold

class Population:
    def __init__(self):

        self.pool = [Brain() for _ in range(Options.population_size)]
        self.species = []

        self.best = self.pool[0]
        self.gen = 0

        Gate.init_pos()

    def evaluate(self, eval_func, graph, report=True):
        
        for generation in range(Options.num_generations):
            
            print('\n\n')
            print('-------------- GENERATION N. {} -------------'.format(self.gen + 1))
            
            # evaluate the task
            eval_func(self.pool, graph)
            # info
            if Options.show_best_distribution:
                scan_best_agent(Options.environment , self.pool)
            
            # mutate and reproduce
            self.epoch()
            
            if Options.dynamic_mut_power:
                # reduce mutation power
                Options.weight_mutate_power = min(Options.initial_weight_mutate_power*0.99**(generation-30), Options.initial_weight_mutate_power)
                print(Options.weight_mutate_power)
            
        # return best agent
        return self.best

    def _speciate(self):
        for brain in self.pool:
            added = False

            for sp in self.species:
                # if the current agent belongs to the current specie
                if sp.same_species(brain):
                    # append to the pool of the current space
                    sp.pool.append(brain)
                    added = True
                    break

            # if the neural network doens't belong to any of the already existing species
            if not added:
                self.species.append(Species(brain))

        # update the list of species
        self.species[:] = [sp for sp in self.species if len(sp.pool) > 0]
        
        # INFO
        print(self)

    def _calc_spawns(self):
        total = max(1, sum([sp.average_fitness for sp in self.species]))
        for sp in self.species:
            # the spwans are the population_size proportioned to the ratio or the total fitness
            # the species that best perform will be larger
            sp.spawns_required = Options.population_size * sp.average_fitness / total

    def _reproduce(self):
        self.pool[:] = []
        
        # in the specie now there are only the best performimg members, who survived
        for s in self.species:
            new_pool = []

            if Options.species_elitism:
                new_pool.append(s.best)

            while len(new_pool) < s.spawns_required:
                # select a brain from the pool of the brains that survived, so the best performin ones
                brain = s.get_brain()

                # do the crossover
                if random.random() < Options.crossover_rate:
                    child = Brain.crossover(brain, s.get_brain())
                else:
                    child = Brain.crossover(brain, brain)

                # mutate the child
                child.mutate()
                
                new_pool.append(child)

            # add the new members
            self.pool.extend(new_pool)
            
            # increase the age and stagnation of the specie and make each specie empty
            s.purge()

        while len(self.pool) < Options.population_size:
            self.pool.append(Brain())

    def _sort_pool(self):

        # sort pool in decreasing order of scores of members
        self.pool.sort(key=lambda x: x.fitness, reverse=True)

        assert self.pool[-1].fitness >= 0, "Cannot handle negative fitnesses"

        if self.best.fitness < self.pool[0].fitness:
            self.best = self.pool[0]
            
    def _adjust_fitnesses(self):
        for s in self.species:
            # select the best agent
            s.make_leader()
            # adjust fitness wrt the age of the specie
            s.adjust_fitnesses()

    def _change_compatibility_threshold(self):
        if len(self.species) < Options.target_species:
            Options.compatibility_threshold *= 0.95

        elif len(self.species) > Options.target_species:
            Options.compatibility_threshold *= 1.05

    def _reset_and_kill(self):
        new_species = []

        for sp in self.species:
            # if it's time to let the specie disappear
            if sp.stagnation > Options.dropoff_age or sp.spawns_required == 0:
                # then do nothing (no append, so the specie disappears)
                continue

            # otherwise kill one portion of the specie, leaving at max 1 member
            sp.cull()
            
            new_species.append(sp)

        # here there are only the old species where an amount of members have been killed. New spowns and new species are not yet there.
        self.species[:] = new_species

    def epoch(self):
        # function managing the mutation and reproduction ecc
        
        # sort the population with the fitness
        self._sort_pool()
        
        # divide the population in species
        self._speciate()

        # the desired amout of species is set at the beginning; if there are more species than the target one then the threshold gets less selective,
        # otherswise, if there are less species, then the threshold gets more selective to diversify more.
        if Options.dynamic_compatibility_threshold:
            self._change_compatibility_threshold()

        # select the leader and adjust the fitnesses wrt the age of the specie, so that newborn species are not penalized
        self._adjust_fitnesses()
        
        # calculate the number of new member of the specie for every specie
        self._calc_spawns()

        # kill the current species depending of their scores
        self._reset_and_kill()

        # repdoducte: crossover and mutation
        self._reproduce()
        

        self.gen += 1

    def __str__(self):
    
                        
        for i, specie in enumerate(self.species):
            scores = []
            for brain in specie.pool:
                scores.append(float(brain.fitness))
                
            print('Specie n. {} | Elements {} | Specie avg score: {} | Best of specie: {}'.format(i+1, len(specie.pool), sum(scores)/len(scores), max(scores)))
            
            
            if Options.show_elite:
                indices = np.argsort(np.array(scores))[-Options.show_elite:]

                for index in indices:
                    brain = specie.pool[index]
                    drawer = qml.draw(brain.qnode)
                    print(drawer([0 for i in range(Options.num_inputs)]))

            

        # wait
        import time
        time.sleep(0)
        
        return '\n\n'
