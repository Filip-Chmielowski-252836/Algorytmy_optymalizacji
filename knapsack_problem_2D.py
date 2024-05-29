# ======================================= #
#   PROBLEM PAKOWANIA / PLECAKOWY W 2D    #
#      Algorytmy optymalizacji (AO)       #
#        inż. Filip Chmielowski, 252836   #
#        inż. Jacek Czyż, 259265          #
#        inż. Adam Hensler, 259298        #
#  Informatyczne Systemy Automatyki (ISA) #
# ======================================= #

# Libraries
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Represents an item with a name, width, and height
class Item:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height

# Reads items from a csv file and returns a list of Item objects
def read_items_from_file(file_path):
    items = []
    with open(file_path, "r") as file:
        for line in file:
            name, width, height = line.strip().split(",")
            items.append(Item(name, int(width), int(height)))
    return items

# Generates an initial population of solutions for the knapsack problem
# where each solution is represented as a list of binary values
# indicating whether an item is included or not
def generate_population(items, population_size):
    population = []
    for _ in range(population_size):
        solution = [random.randint(0, 1) for _ in range(len(items))]
        population.append(solution)
    return population

# Calculates the fitness of a solution based on how well the items fit
# into the knapsack without exceeding its width and height constraints
def fitness(solution, items, knapsack_width, knapsack_height):
    total_value = 0
    columns = [0] * knapsack_width  # Initialize column heights with zeros

    for i in range(len(items)):
        if solution[i] == 1:
            item = items[i]

            # Find the lowest possible y position for this item
            min_y = float("inf")

            position_x = 0
            for x in range(knapsack_width - item.width + 1):
                max_column_height = max(columns[x : x + item.width])
                if max_column_height < min_y:
                    min_y = max_column_height
                    position_x = x

            # Check if the item fits in the knapsack height
            if min_y + item.height > knapsack_height:
                return 0   # If any item doesn't fit, the solution is invalid

            # Place the item at the lowest possible y position
            for x in range(position_x, position_x + item.width):
                columns[x] = min_y + item.height

            # Add the value of the item to the total value
            total_value += (item.width * item.height)

    return total_value

# Selects the best-performing solutions from the population based on their fitness scores to proceed to the next generation
def selection(population, items, knapsack_width, knapsack_height):
    sorted_population = sorted(population, key = lambda x : fitness(x, items, knapsack_width, knapsack_height), reverse = True)
    return sorted_population[:len(sorted_population) // 2]

# Performs crossover between two parent solutions to generate two
# child solutions, aiding in the exploration of the solution space
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Applies mutation to a solution with a certain probability
# to introduce diversity and prevent premature convergence
def mutation(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]
    return solution

# Visualizes the best solution found by plotting the items packed in the knapsack using matplotlib
def visualize_solution(items, knapsack_width, knapsack_height, solution):
    packed_items = next_fit(items, knapsack_width, knapsack_height, solution)
    fig, ax = plt.subplots()
    ax.set_xlim(0, knapsack_width)
    ax.set_ylim(0, knapsack_height)
    ax.set_aspect("equal", "box")

    for item, x, y in packed_items:
        color = (random.random(), random.random(), random.random())   # Generate a random color for each item in the knapsack
        rect = patches.Rectangle((x, y), item.width, item.height, linewidth = 1, edgecolor = "r", facecolor = color)
        ax.add_patch(rect)
        ax.text(x + item.width / 2, y + item.height / 2, item.name, ha = "center", va = "center", fontsize = 8, weight = "bold", color = "white")

    plt.title("Items in Knapsack", fontsize = 12, weight = "bold")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

# Implements the genetic algorithm to solve the knapsack problem by evolving a population of solutions over multiple generations
def genetic_algorithm(items, knapsack_width, knapsack_height, population_size, generations, initial_mutation_rate):
    population = generate_population(items, population_size)
    fitness_history = []
    mutation_rate = initial_mutation_rate

    for gen in range(generations):
        population = selection(population, items, knapsack_width, knapsack_height)
        new_population = population.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(population), random.choice(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population[:population_size]
        best_solution = max(population, key = lambda x : fitness(x, items, knapsack_width, knapsack_height))
        best_fitness = fitness(best_solution, items, knapsack_width, knapsack_height)
        fitness_history.append(best_fitness)

        if gen % 10 == 0:
            print(f"Generation {gen}: Best Fitness = {best_fitness}")
        
        mutation_rate *= 0.95   # Decrease mutation rate by 5%

    return best_solution, best_fitness, fitness_history

# Packs the items into the knapsack and returns the packed items' coordinates for visualization
def next_fit(items, knapsack_width, knapsack_height, solution):
    columns = [0] * knapsack_width   # Initialize column heights to zero
    packed_items = []
    max_height = 0

    for i in range(len(items)):
        if solution[i] == 1:
            item = items[i]
            
            # Find the lowest possible y position for this item
            min_y = float("inf")

            position_x = 0
            for x in range(knapsack_width - item.width + 1):
                max_column_height = max(columns[x : x + item.width])
                if max_column_height < min_y:
                    min_y = max_column_height
                    position_x = x

            # Check if the item fits in the knapsack height and, if not, returns appropriate information
            if min_y + item.height > knapsack_height:
                print(f"Item {item.name} cannot be packed because it exceeds the knapsack height!")
                continue   # Item cannot be packed inside the knapsack

            # Prints the information about each item being packed in the knapsack with its coordinates
            print(f"Packing {item.name} at ({position_x}, {min_y})")

            # Place the item at the lowest possible y position
            for x in range(position_x, position_x + item.width):
                columns[x] = min_y + item.height

            # Update the max height of the knapsack
            max_height = max(max_height, min_y + item.height)

            # Add the item to the packed items list
            packed_items.append((item, position_x, min_y))

    return packed_items

# ==================== Main function of the program ====================
if __name__ == "__main__":

    # Path to the input data file (with items to be packed inside the knapsack)
    input_file_path = "knapsack_problem_2D_data_1.csv"   # I | II | III | IV

    # Elements to be packed inside the knapsack are being read from a csv file
    items = read_items_from_file(input_file_path)

    # Simulation parameters
    knapsack_width = 60    # I: 60 | II: 10 | III: 15 | IV: 25
    knapsack_height = 80   # I: 80 | II: 10 | III: 25 | IV: 30
    population_size = 100
    generations = 100
    initial_mutation_rate = 0.1

    best_solution, best_fitness, fitness_history = genetic_algorithm(items, knapsack_width, knapsack_height, population_size, generations, initial_mutation_rate)

    visualize_solution(items, knapsack_width, knapsack_height, best_solution)

    # Prints the best solution and the best fitness in each generation
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    # Plots the changes' history of fitness over generations
    plt.plot(fitness_history)
    plt.title("Fitness changes over Generations", fontsize = 12, weight = "bold")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()