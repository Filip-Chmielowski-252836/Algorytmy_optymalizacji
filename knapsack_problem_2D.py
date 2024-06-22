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
import alive_progress as ab
import json
import itertools
import time
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


class PackedItem:
    def __init__(self, name, bottom_left, top_right):
        self.name = name
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.width = top_right[0] - bottom_left[0]
        self.height = top_right[1] - bottom_left[1]

def does_collide(item1, item2):
    if item1.top_right[0] <= item2.bottom_left[0] or item2.top_right[0] <= item1.bottom_left[0]:
        return False
    
    # Check if one rectangle is above the other
    if item1.top_right[1] <= item2.bottom_left[1] or item2.top_right[1] <= item1.bottom_left[1]:
        return False
    
    # If both conditions are false, rectangles overlap
    return True



def fitness(solution, items, knapsack_width, knapsack_height, return_fit, method):
    if method == 'full_search':
        return fitness_full_search(solution, items, knapsack_width, knapsack_height, return_fit)
    if method == 'top_search':
        return fitness_top_search(solution, items, knapsack_width, knapsack_height, return_fit)
    raise NotImplementedError
    


def fitness_top_search(solution, items, knapsack_width, knapsack_height, return_fit):
    fit = 0
    packed_items_ret = []
    packed_items = []
    for i in range(len(items)):
        if solution[i] == 1:
            item_placed = False
            for selected_item in packed_items:
                # Checking bottom right corners of items
                x = selected_item.top_right[0]
                y = selected_item.bottom_left[1]
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
                # Checking the tops of every item
                y = selected_item.top_right[1]
                for x in range(selected_item.bottom_left[0],selected_item.top_right[0]):   # jakby chcieć ograniczyć wiszenie to tutaj zrobić jakieś -0.5długości czy coś
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
                if item_placed:
                    break
            # Checking the first row
            if i == 0:
                item_to_pack = PackedItem(items[i].name, [0, 0], [0 + items[i].width, 0 + items[i].height])
                if item_to_pack.top_right[0] <= knapsack_width and item_to_pack.top_right[1] <= knapsack_height:
                    fit += items[i].area
                    packed_items.append(item_to_pack)
                    packed_items_ret.append((items[i], 0, 0))
                    item_placed = True
            # Item does not fit - 0 value, eliminate from pool
            if not item_placed:
                    return 0
    if return_fit:
        return fit
    else:
        return packed_items_ret


def fitness_full_search(solution, items, knapsack_width, knapsack_height, return_fit):
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
def selection(population, items, knapsack_width, knapsack_height, packing_method):
    sorted_population = sorted(population, key=lambda x: fitness(x, items, knapsack_width, knapsack_height, True, packing_method), reverse=True)
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
def visualize_solution(items, knapsack_width, knapsack_height, solution, packing_method):

    sorted_items = sorted(items, key=lambda x: x.area, reverse=True) 
    packed_items = fitness(solution, sorted_items, knapsack_width, knapsack_height, False, packing_method)
    fig, ax = plt.subplots()
    ax.set_xlim(0, knapsack_width)
    ax.set_ylim(0, knapsack_height)
    ax.set_aspect('equal', 'box')

    if packed_items!=0:
        for item, x, y in packed_items:
            color = (random.random(), random.random(), random.random())   # Generate a random color for each item in the knapsack
            rect = patches.Rectangle((x, y), item.width, item.height, linewidth = 1, edgecolor = "r", facecolor = color)
            ax.add_patch(rect)
            ax.text(x + item.width / 2, y + item.height / 2, item.name, ha = "center", va = "center", fontsize = 8, weight = "bold", color = "white")

    else:
        print("knapsack is empty!")

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Items in Knapsack')
    plt.show()

# Brute force approach
def brute(items, knapsack_width, knapsack_height, packing_method):
    sorted_items = sorted(items, key=lambda x: x.area, reverse=True)    
    # population = generate_population(items, population_size)
    fitness_history = []
    best_fitness=0
    with ab.alive_bar(total=2**len(items), force_tty=True) as bar:
        for i in itertools.product([0, 1], repeat=len(items)):
            fit=fitness(i, sorted_items, knapsack_width, knapsack_height, True, packing_method)
            bar()
            if best_fitness<fit:
                best_fitness=fit
                best_solution=i
                fitness_history.append(best_fitness)
    return best_solution, best_fitness, fitness_history


# Implements the genetic algorithm to solve the knapsack problem by evolving a population of solutions over multiple generations
def genetic_algorithm(items, knapsack_width, knapsack_height, population_size, generations, initial_mutation_rate, packing_method):
    sorted_items = sorted(items, key=lambda x: x.area, reverse=True)    
    population = generate_population(items, population_size)
    fitness_history = []
    mutation_rate = initial_mutation_rate
    tmsum = 0
    with ab.alive_bar(generations, force_tty=True) as bar:
        for gen in range(generations):
            bar.title = 'Genetic Simulation Running'
            bar()
            population = selection(population, sorted_items, knapsack_width, knapsack_height, packing_method)
            new_population = population.copy()
            while len(new_population) < population_size:
                parent1, parent2 = random.choice(population), random.choice(population)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)
                new_population.extend([child1, child2])
            population = new_population[:population_size]
            tm0 = time.time()
            best_solution = max(population, key=lambda x: fitness(x, sorted_items, knapsack_width, knapsack_height, True, packing_method))
            best_fitness = fitness(best_solution, sorted_items, knapsack_width, knapsack_height, True, packing_method)
            tm1 = time.time()
            tmsum = tmsum + tm1-tm0
            fitness_history.append(best_fitness)
            amount=0
            for i in range(len(best_solution)):
                if best_solution[i]==1:
                    amount+=1
            if best_fitness==knapsack_height*knapsack_width or amount==len(items):
                bar.title = 'Solution Found - Early Stop'
                break
            mutation_rate *= 0.95   # Decrease mutation rate
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


    name, items, knapsack_width, knapsack_height = load_json("dataset.json")


    population_size =100
    generations = 50
    initial_mutation_rate = 0.1
    packing_method = 'top_search'


    # A Dynamic Programming based Python
    # Program for 0-1 Knapsack problem
    # Returns the maximum value that can
    # be put in a knapsack of capacity W
    #
    # Loosely based on https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/
    #
    def dynamic_programming(items, knapsack_width, knapsack_height, packing_method):
    # def knapSack(W, wt, val, n):
        max_weight = knapsack_height*knapsack_width
        K = [[[0 for _ in range(knapsack_height + 1)] for _ in range(knapsack_width + 1)] for _ in range(len(items) + 1)]

        # Build table K[][][] in bottom up manner
        with ab.alive_bar(len(items) + 1, force_tty=True) as bar:
            for item_idx in range(len(items) + 1):             # currently tested item form i
                bar()
                sorted_items = sorted(items, key=lambda x: x.area, reverse=True)  
                for curr_max_w in range(knapsack_width + 1):
                    for curr_max_h in range(knapsack_height + 1):
                        if item_idx == 0 or curr_max_w == 0 or curr_max_h == 0:
                            K[item_idx][curr_max_w][curr_max_h] = 0
                        elif items[item_idx-1].width <= curr_max_w and items[item_idx-1].height <= curr_max_h:
                            # Get current item list from table K\
                            back_max_w = curr_max_w
                            back_max_h =curr_max_h
                            packed_list = [0 for _ in items]
                            sack_empty = 1
                            for back_item in reversed(range(0,item_idx-1)):
                                # print(back_item)
                                if K[back_item][back_max_w][back_max_h] != K[back_item-1][back_max_w][back_max_h]:
                                    # item is packed
                                    print(back_item)
                                    back_max_w -= items[back_item].width
                                    back_max_h -= items[back_item].height
                                    packed_list[back_item] = 1
                                    sack_empty = 0
                            if sack_empty == 1:
                                packed_list[item_idx-1] = 1
                            # /\/\/\/\
                            K[item_idx][curr_max_w][curr_max_h] = max(fitness(packed_list, sorted_items, knapsack_width, knapsack_height, True, packing_method),
                                        K[item_idx-1][curr_max_w][curr_max_h])
                        else:
                            K[item_idx][curr_max_w][curr_max_h] = K[item_idx-1][curr_max_w][curr_max_h]
        print(K)
        return K[len(items)][knapsack_width][knapsack_height]



    # best_solution, best_fitness, fitness_history = genetic_algorithm(items, knapsack_width, knapsack_height, population_size, generations, initial_mutation_rate, packing_method)

    print(dynamic_programming(items, knapsack_width, knapsack_height, packing_method))
    # print("Best Solution:", best_solution)
    # print("Best Fitness:", best_fitness)
    
    # # visualize_solution(items, knapsack_width, knapsack_height, best_solution, packing_method)

    # plt.plot(fitness_history)
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness')
    # plt.title('Fitness Changes over Generations')
    # plt.show()