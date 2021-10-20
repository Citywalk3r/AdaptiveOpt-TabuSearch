import itertools
import numpy as np
from random import sample

class Tabu_search:

    def __init__(self, is_debug):
        self.is_debug = is_debug

    def generate_init_solution(self, seed):
        """
        Generates the initial solution given random seed.
        """
        np.random.seed(seed=seed)
        init_solution = np.random.permutation(range(1,21))
        if self.is_debug:
            print("Initial solution: ", init_solution)
        return init_solution
    
    def generate_neighborhood(self, currState, eval_f, neighborhood_strategy):
        """Implements the move operator.
           Picks 2 random elements from the array and swaps them.
        """

        neighborhood = []
        swaps = []
        neighborhood_evaluation_list = []

        combinations = list(itertools.combinations(currState, 2))

        if neighborhood_strategy=="partial":
            combinations = sample(combinations, len(combinations)//2)
        
        for pair in combinations:

            c = np.ndarray.tolist(currState).copy()
            
            idx1 = np.where(c == pair[0])[0][0]
            idx2 = np.where(c == pair[1])[0][0]

            c[idx1] , c[idx2] = c[idx2] , c[idx1]

            if self.is_debug:
                print("Pair to swap: ", pair)
                print("Index of element 1: ", idx1)
                print("Index of element 2: ", idx2)
                print("Current state : ", currState)
                print("Permutation: ", c)

            neighborhood.append(c)
            swaps.append(pair)
            neighborhood_evaluation_list.append(eval_f(c))

        # Combine solution, swap, and evaluation into one list
        zipped_list = list(zip(neighborhood, swaps, neighborhood_evaluation_list))

        # Sort the combined list based on the evaluation
        sorted_neighborhood = sorted(zipped_list, key=lambda x: x[2])

        if self.is_debug:
            print("Neighborhood:{} \n Neighborhood size: {} ".format(neighborhood, len(neighborhood)))
            print("Swaps:{} \n Swaps size: {} ".format(swaps, len(swaps)))
            print("Neighborhood evaluation list:{} \n Neighborhood evaluation list size: {} ".format(neighborhood_evaluation_list, len(neighborhood_evaluation_list)))
            print("Zipped list:{} \n Zipped list size: {} ".format(zipped_list, len(zipped_list)))
            print("Sorted neighborhood:{} \n Sorted neighborhood size: {} ".format(sorted_neighborhood, len(sorted_neighborhood)))

        return sorted_neighborhood

    def ts(self, tabu_size, iterations, eval_f, seed, t_strategy=None, neighborhood_strategy = None, restart_strategy = None):

            """Tabu search

            Parameters:
                tabu_size: size of the tabu list
                iterations : maximum number of iterations
                eval_f : evaluation function
                seed: Random seed for population initialization
                t_strategy: strategy for the tabu size
                neighborhood_strategy: neighborhood exploration strategy
                restart_strategy: restart strategy when stagnant

            Returns:
                best_so_far : best found solution
                best_list : list of historical best solutions through the iterations
            """
            
            print("Running the tabu search algorithm...")

            # initialization
            self.tabu_list = []
            self.best_so_far = 999999
            best_list = []
            iterations_ctr = 0
            stagnant_counter = 0
            initial_population = self.generate_init_solution(seed)
            currState = initial_population

            while iterations_ctr < iterations:
                if (t_strategy == "random"):
                    if iterations_ctr%100 == 0 or iterations_ctr==0:
                        tabu_size = np.random.randint(7,16)
                    
                neighborhood = self.generate_neighborhood(currState, eval_f, neighborhood_strategy)

                for neighbor in neighborhood:
                    if neighbor[1] not in self.tabu_list and (neighbor[1][1], neighbor[1][0]) not in self.tabu_list:
                        break
                    elif neighbor[2] < self.best_so_far:
                        break
                
                self.tabu_list.append(neighbor[1])

                while len(self.tabu_list) > tabu_size:
                    self.tabu_list.pop(0)

                if neighbor[2] < self.best_so_far:
                    self.best_so_far = neighbor[2]
                    stagnant_counter = 0
                else:
                    stagnant_counter+=1
    
                if restart_strategy == "random" and stagnant_counter == 10:
                    currState = np.random.permutation(range(1,21))
                else:
                    currState = np.array(neighbor[0])

                best_list.append(self.best_so_far)
                iterations_ctr+=1

            return  best_list, self.best_so_far