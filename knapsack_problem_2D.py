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

    # return not (item1.top_right[0] <= item2.bottom_left[0] or
    #             item1.bottom_left[0] >= item2.top_right[0] or
    #             item1.top_right[1] <= item2.bottom_left[1] or
    #             item1.bottom_left[1] >= item2.top_right[1])




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
                        fit += items[i].width * items[i].height
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

                # for j in range(0, len(packed_items)): #sprawdzam na prawo od kazdego elementu na razie, to byloby super, ale na start wersja z jazda po calym plecaku ^
                #     item_to_pack = PackedItem(items[i].name, [packed_items[j].bottom_left[1],packed_items[j].top_right[0]], [packed_items[j].bottom_left[1]+items[i].width, packed_items[j].top_right[0]+items[i].height])
                    # if item_to_pack.top_right[0]<knapsack_width and item_to_pack.top_right[1]<knapsack_height: #wiem, że się mieści, sprawdzam czy koliduje
                    #     for k in range(0, len(packed_items)):
                    #         collides = does_collide(item_to_pack, packed_items[k])
                    #     if collides == False:
                    #         fit+=items[i].width * items[i].height
                    #         break

    # if return_fit==True:
    #     return fit
    # else:
    #     return packed_items_ret


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
    # packed_items = next_fit(items, knapsack_width, knapsack_height, solution)
    packed_items = fitness(solution, items, knapsack_width, knapsack_height, False)
    # print(packed_items)
    fig, ax = plt.subplots()
    ax.set_xlim(0, knapsack_width)
    ax.set_ylim(0, knapsack_height)
    ax.set_aspect('equal', 'box')

    for item, x, y in packed_items:
        rect = patches.Rectangle((x, y), item.width, item.height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Items in Knapsack')
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
        best_solution = max(population, key=lambda x: fitness(x, items, knapsack_width, knapsack_height, True))
        best_fitness = fitness(best_solution, items, knapsack_width, knapsack_height, True)
        fitness_history.append(best_fitness)
        mutation_rate *= 0.95   # Decrease mutation rate
    return best_solution, best_fitness, fitness_history

# # Packs the items into the knapsack using the Next Fit heuristic and returns the packed items' coordinates for visualization
# def next_fit(items, knapsack_width, knapsack_height, solution):
#     packed_items = []
#     bin_widths = []
#     bin_heights = []
#     bin_widths.append(0)
#     bin_heights.append(0)

#     bin_index = 0
#     for i in range(len(items)):
#         if solution[i] == 1:
#             item = items[i]
#             if bin_widths[bin_index] + item.width <= knapsack_width:
#                 bin_widths[bin_index] += item.width
#                 bin_heights[bin_index] = max(bin_heights[bin_index], item.height)
#             else:
#                 bin_index += 1
#                 bin_widths.append(item.width)
#                 bin_heights.append(item.height)

#     x = 0
#     y = 0
#     max_height = 0
#     for i in range(len(items)):
#         if solution[i] == 1:
#             item = items[i]
#             if x + item.width > knapsack_width:
#                 x = 0
#                 y += max_height
#                 max_height = 0
#             # Adjust y-coordinate to stack items on lower surface if possible
#             if max_height > 0 and y + item.height > knapsack_height:
#                 # Try to find the lowest y-coordinate where the item can fit
#                 min_y = knapsack_height
#                 for packed_item in packed_items:
#                     if packed_item[1] + item.width <= knapsack_width:
#                         if packed_item[2] + packed_item[0].height <= min_y:
#                             min_y = packed_item[2] + packed_item[0].height
#                 if min_y < knapsack_height:
#                     y = min_y
#                 else:
#                     y = 0  # Reset y-coordinate to start a new column
#                     x += max_width  # Move to the next column
#                     max_height = 0  # Reset max_height for the new column
#             packed_items.append((item, x, y))
#             x += item.width
#             max_height = max(max_height, item.height)

#     return packed_items



# Main function of the program
if __name__ == "__main__":
    items = [
        Item("item1", 2, 3),
        Item("item2", 3, 4),
        Item("item3", 4, 5),
        Item("item4", 5, 6),
        Item("item5", 6, 7),
        Item("item6", 7, 8),
        Item("item7", 20, 29),
        Item("item8", 9, 10),
        Item("item9", 10, 11),
        Item("item10", 11, 12),
        Item("item11", 12, 13),
        Item("item12", 13, 14),
        Item("item13", 14, 15),
        Item("item14", 15, 16),
        Item("item15", 16, 17),
        Item("item16", 17, 18),
        Item("item17", 18, 19),
        Item("item18", 19, 20),
        Item("item19", 20, 21),
        Item("item20", 21, 22)
    ]
    knapsack_width = 40
    knapsack_height = 60
    population_size = 100
    generations = 100
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