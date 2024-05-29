# ======================================= #
#   PROBLEM PAKOWANIA / PLECAKOWY W 2D    #
#      Algorytmy optymalizacji (AO)       #
#        Filip Chmielowski, 252836        #
#           Jacek Czyż, 259265            #
#          Adam Hensler, 259298           #
#  Informatyczne Systemy Automatyki (ISA) #
# ======================================= #

# Libraries
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json
# Represents an item with a name, width, and height
class Item:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.area=width*height


# Generates an initial population of solutions for the knapsack problem where each solution
# is represented as a list of binary values indicating whether an item is included or not
def generate_population(items, population_size):
    population = []
    for _ in range(population_size):
        solution = [random.randint(0, 1) for _ in range(len(items))]
        population.append(solution)
    return population


class FreeSpace:
    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right


class PackedItem:
    def __init__(self, name, bottom_left, top_right):
        self.name = name
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.width = top_right[0] - bottom_left[0]
        self.height = top_right[1] - bottom_left[1]

def does_collide(item1, item2):
    """
    Check if two items collide.
    """
    if item1.top_right[0] <= item2.bottom_left[0] or item2.top_right[0] <= item1.bottom_left[0]:
        return False
    
    # Check if one rectangle is above the other
    if item1.top_right[1] <= item2.bottom_left[1] or item2.top_right[1] <= item1.bottom_left[1]:
        return False
    
    # If both conditions are false, rectangles overlap
    return True




def fitness(solution, items, knapsack_width, knapsack_height, return_fit):
    fit = 0
    packed_items_ret = []
    packed_items = []
    for i in range(len(items)):
        if solution[i] == 1:
            item_placed = False
            for y in range(knapsack_height):
                for x in range(knapsack_width):
                    item_to_pack = PackedItem(items[i].name, [x, y], [x + items[i].width, y + items[i].height])
                    collides = False
                    for packed_item in packed_items:
                        if does_collide(item_to_pack, packed_item):
                            collides = True
                            break
                    if not collides and item_to_pack.top_right[0] <= knapsack_width and item_to_pack.top_right[1] <= knapsack_height:
                        fit += items[i].area
                        packed_items.append(item_to_pack)
                        packed_items_ret.append((items[i], x, y))
                        item_placed = True

                        break
                    elif not item_to_pack.top_right[0] <= knapsack_width and not item_to_pack.top_right[1] <= knapsack_height:
                        return 0
                if item_placed:
                    break


    if return_fit:
        return fit
    else:
        return packed_items_ret



# Selects the best-performing solutions from the population based on their fitness scores to proceed to the next generation
def selection(population, items, knapsack_width, knapsack_height):
    sorted_population = sorted(population, key=lambda x: fitness(x, items, knapsack_width, knapsack_height, True), reverse=True)
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

    sorted_items = sorted(items, key=lambda x: x.area, reverse=True) 
    packed_items = fitness(solution, sorted_items, knapsack_width, knapsack_height, False)
    fig, ax = plt.subplots()
    ax.set_xlim(0, knapsack_width)
    ax.set_ylim(0, knapsack_height)
    ax.set_aspect('equal', 'box')

    for item, x, y in packed_items:
        color = (random.random(), random.random(), random.random())   # Generate a random color for each item in the knapsack
        rect = patches.Rectangle((x, y), item.width, item.height, linewidth = 1, edgecolor = "r", facecolor = color)
        ax.add_patch(rect)
        ax.text(x + item.width / 2, y + item.height / 2, item.name, ha = "center", va = "center", fontsize = 8, weight = "bold", color = "white")

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Items in Knapsack')
    plt.show()

# Implements the genetic algorithm to solve the knapsack problem by evolving a population of solutions over multiple generations
def genetic_algorithm(items, knapsack_width, knapsack_height, population_size, generations, initial_mutation_rate):
    sorted_items = sorted(items, key=lambda x: x.area, reverse=True)    
    population = generate_population(items, population_size)
    fitness_history = []
    mutation_rate = initial_mutation_rate
    for gen in tqdm(range(generations)):
        population = selection(population, sorted_items, knapsack_width, knapsack_height)
        new_population = population.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(population), random.choice(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population[:population_size]
        best_solution = max(population, key=lambda x: fitness(x, sorted_items, knapsack_width, knapsack_height, True))
        best_fitness = fitness(best_solution, sorted_items, knapsack_width, knapsack_height, True)
        fitness_history.append(best_fitness)
        amount=0
        for i in range(len(best_solution)):
            if best_solution[i]==1:
                amount+=1
        if best_fitness==knapsack_height*knapsack_width or amount==len(items):
            print("\nfound complete solution: stopping\n")
            break
        mutation_rate *= 0.95   # Decrease mutation rate

    print(amount)
    return best_solution, best_fitness, fitness_history

def load_json(fname):
    f=open(fname, "r")
    data=f.read()
    parsed=json.loads(data)

    items_dict=parsed["items"]
    items=[]
    for item in items_dict:
        items.append(Item(item["name"], item["size_x"], item["size_y"]))

    f.close()
    return parsed["name"], items, parsed["size_x"], parsed["size_y"]

# Main function of the program
if __name__ == "__main__":

    print(load_json("dataset.json"))
    name, items, knapsack_width, knapsack_height = load_json("dataset.json")


    population_size =100
    generations = 50
    initial_mutation_rate = 0.1

    best_solution, best_fitness, fitness_history = genetic_algorithm(items, knapsack_width, knapsack_height, population_size, generations, initial_mutation_rate)
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    visualize_solution(items, knapsack_width, knapsack_height, best_solution)

    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Changes over Generations')
    plt.show()