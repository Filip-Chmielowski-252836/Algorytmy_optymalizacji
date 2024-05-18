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


# Generates an initial population of solutions for the knapsack problem where each solution
# is represented as a list of binary values indicating whether an item is included or not
def generate_population(items, population_size):
    population = []
    for _ in range(population_size):
        solution = [random.randint(0, 1) for _ in range(len(items))]
        population.append(solution)
    return population


# Calculates the fitness of a solution based on how well the items fit
# into the knapsack without exceeding its width and height constraints
def fitness(solution, items, knapsack_width, knapsack_height):
    fit = 0
    used_width = 0
    used_height = 0
    for i in range(0, len(items)):
        if solution[i] == 1:
            used_width += items[i].width
            used_height = max(used_height, items[i].height)   # Elements are being put on the heighest of all elements, even if there is a better option
            if used_width > knapsack_width:
                used_width = items[i].width
                used_height += items[i].height
            if used_height > knapsack_height:
                return 0
            fit += (items[i].width * items[i].height)
    return fit


# Selects the best-performing solutions from the population based on their fitness scores to proceed to the next generation
def selection(population, items, knapsack_width, knapsack_height):
    sorted_population = sorted(population, key = lambda x : fitness(x, items, knapsack_width, knapsack_height), reverse = True)
    return sorted_population[:len(sorted_population) // 2]


# Performs crossover between two parent solutions to generate two child solutions, aiding in the exploration of the solution space
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Applies mutation to a solution with a certain probability to introduce diversity and prevent premature convergence
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
        
        mutation_rate *= 0.95   # Decrease mutation rate
    return best_solution, best_fitness, fitness_history


# Packs the items into the knapsack using the Next_Fit heuristic and returns the packed items' coordinates for visualization
def next_fit(items, knapsack_width, knapsack_height, solution):
    columns = [0] * knapsack_width   # Initialize column heights to zero
    packed_items = []
    max_height = 0
    for i in range(len(items)):
        if solution[i] == 1:
            item = items[i]
            min_y = float("inf")   # Find the lowest possible y position for this item
            position_x = 0
            for x in range(knapsack_width - item.width + 1):
                max_column_height = max(columns[x : x + item.width])
                if max_column_height < min_y:
                    min_y = max_column_height
                    position_x = x

            # Check if the item fits in the knapsack height and if not - returns appropriate information about it
            if min_y + item.height > knapsack_height:
                print(f"Item {item.name} cannot be packed because it exceeds the knapsack height!")
                continue

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


# Main function of the program
if __name__ == "__main__":

    # Elements to be packed inside the knapsack
    items = [
        Item("Item1", 2, 3),    Item("Item2", 3, 4),    Item("Item3", 4, 5),    Item("Item4", 5, 6),
        Item("Item5", 6, 7),    Item("Item6", 7, 8),    Item("Item7", 8, 9),    Item("Item8", 9, 10),
        Item("Item9", 10, 11),  Item("Item10", 11, 12), Item("Item11", 12, 13), Item("Item12", 13, 14),
        Item("Item13", 14, 15), Item("Item14", 15, 16), Item("Item15", 16, 17), Item("Item16", 17, 18),
        Item("Item17", 18, 19), Item("Item18", 19, 20), Item("Item19", 20, 21), Item("Item20", 21, 22),
        Item("Item21", 22, 23), Item("Item22", 23, 24), Item("Item23", 24, 25), Item("Item24", 25, 26)
    ]

    # Simulation parameters
    knapsack_width = 60
    knapsack_height = 80
    population_size = 100
    generations = 50
    initial_mutation_rate = 0.1

    best_solution, best_fitness, fitness_history = genetic_algorithm(items, knapsack_width, knapsack_height, population_size, generations, initial_mutation_rate)

    # Prints the best solution and the best fitness in each generation
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    visualize_solution(items, knapsack_width, knapsack_height, best_solution)

    # Plots the changes' history of fitness over generations
    plt.plot(fitness_history)
    plt.title("Fitness changes over Generations", fontsize = 12, weight = "bold")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()