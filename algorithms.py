import sys
import numpy as np
import ndomsort

class MOEADVA(AbstractGeneticAlgorithm):

    def __init__(self, problem,
                 neighborhood_size = 10,
                 generator = RandomGenerator(),
                 variator = None,
                 delta = 0.8,
                 eta = 1,
                 update_utility = None,
                 weight_generator = random_weights,
                 scalarizing_function = chebyshev,
                 **kwargs):
        super(MOEAD, self).__init__(problem, 0, generator, **remove_keys(kwargs, "population_size")) # population_size is set after generating weights
        self.neighborhood_size = neighborhood_size
        self.variator = variator
        self.delta = delta
        self.eta = eta
        self.update_utility = update_utility
        self.weight_generator = weight_generator
        self.scalarizing_function = scalarizing_function
        self.generation = 0
        self.weight_generator_kwargs = only_keys_for(kwargs, weight_generator)

        # MOEA/D currently only works on minimization problems
        if any([d != Problem.MINIMIZE for d in problem.directions]):
            raise PlatypusError("MOEAD currently only works with minimization problems")

        # If using the default weight generator, random_weights, use a default
        # population_size
        if weight_generator == random_weights and "population_size" not in self.weight_generator_kwargs:
            self.weight_generator_kwargs["population_size"] = 100

    def control_variable_analysis(self, n_vars, function_evaluations, NCA, lower_bound, upper_bound, function):
        diversionIndexes = []
        converIndexes = []
        sampleSet = []

        for i in range(n_vars):
            # Generate random vector
            X = np.random.rand(n_vars) * upper_bound
            for j in range(NCA):
                X[i] = X[i].lower_bound + (j-1 + random.random()) * (X[i].upper_bound - X[i].lower_bound) / (NCA)
                sampleSet.append(function(X))
                function_evaluations += 1

            #Do non-dominated sorting on sampleSet and then check for first front
            fronts = ndomsort.non_domin_sort(sampleSet, function)

            if len(fronts[0]) == NCA:
                diversionIndexes.append(i)
            else:
                flag = False
                for key in fronts.keys():
                    if len(fronts[key] != 1):
                        diversionIndexes.append(i)
                        flag = True
                        break

                if not flag:
                    converIndexes.append(i)

        return diversionIndexes, converIndexes

    def interaction_analysis(self, converIndexes, diverIndexes, NIA, n_vars):
        interaction = [False] * n_vars