# ======================================= #
#   PROBLEM PAKOWANIA / PLECAKOWY W 2D    #
#      Algorytmy optymalizacji (AO)       #
#        Filip Chmielowski, 252836        #
#           Jacek Czyż, 259265            #
#          Adam Hensler, 259298           #
#  Informatyczne Systemy Automatyki (ISA) #
# ======================================= #

# Libraries
import csv
import random

# A class representing the genetic algorithm for solving the knapsack problem
class KnapsackGeneticAlgorithm:
    def __init__(self, data_file, capacity, population_size=100, mutation_rate=0.01, generations=100):
        self.weights, self.values = self.read_data(data_file)
        self.capacity = capacity
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []

    # Creates an initial population of random chromosomes
    def initialize_population(self):
        for _ in range(self.population_size):
            chromosome = [random.choice([0, 1]) for _ in range(len(self.weights))]
            self.population.append(chromosome)

    # Calculates the fitness of a chromosome, which is the total value of the items
    # in the knapsack if the weight does not exceed the capacity, otherwise 0
    def fitness(self, chromosome):
        total_weight = sum(chromosome[i] * self.weights[i] for i in range(len(chromosome)))
        total_value = sum(chromosome[i] * self.values[i] for i in range(len(chromosome)))
        if total_weight > self.capacity:
            return 0
        else:
            return total_value

    # Performs crossover between two parent chromosomes to produce two child chromosomes
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    # Applies mutation to a chromosome with a certain probability
    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    # Selects two parent chromosomes based on their fitness
    def select_parents(self, population):
        fitnesses = [self.fitness(chromosome) for chromosome in population]
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        parents = random.choices(population, weights=probabilities, k=2)
        return parents

    # Runs the genetic algorithm for a certain number of generations, performing selection, crossover,
    # and mutation in each generation. Finally, it returns the best value found and the corresponding chromosome
    def evolve(self):
        self.initialize_population()
        for _ in range(self.generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(self.population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population
        best_chromosome = max(self.population, key=self.fitness)
        best_value = self.fitness(best_chromosome)
        return best_value, best_chromosome
    
    # Reads the data from the .csv file in the appropriate way
    def read_data(self, data_file):
        weights = []
        values = []
        with open(data_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)   # Skips needless header
            for row in reader:
                weights.append(int(row[0]))
                values.append(int(row[1]))
        return weights, values

# Usage of the Knapsack Algorithm
data_file = "knapsack_problem_1D_data.csv"
capacity = 5
genetic_algorithm = KnapsackGeneticAlgorithm(data_file, capacity)
max_value, selected_items = genetic_algorithm.evolve()
print("Maximum value: ", max_value)
print("Selected items: ", selected_items)