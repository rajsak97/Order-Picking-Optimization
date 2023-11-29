import networkx as nx
import matplotlib.pyplot as plt
import random
from deap import creator
from deap import base
from deap import tools

# create graph of approximated nodes for the S.MART store
G = nx.Graph()

# add nodes
G.add_node(0.0, pos=(0, 0))  # starting position
G.add_node(4.1, pos=(-1, 0))
G.add_node(4.2, pos=(-3, 0))
G.add_node(4.3, pos=(-5, 0))
G.add_node(2.1, pos=(-6, 1))
G.add_node(2.2, pos=(-6, 2))
G.add_node(2.3, pos=(-4, 4))
G.add_node(1.1, pos=(0, 4))
G.add_node(3.1, pos=(-5, 2))
G.add_node(3.2, pos=(-3, 2.5))
G.add_node(3.3, pos=(-1, 2))
G.add_node(3.4, pos=(-3, 1.5))
G.add_node(0.1, pos=(0, 2))  # ending position

# add edges with weight equivalent to walking distance between nodes (actual walkpaths through the store)
G.add_weighted_edges_from([
    (0.0, 4.1, 1),
    (4.1, 4.2, 2),
    (4.1, 4.2, 2),
    (4.1, 3.3, 2),
    (4.2, 4.3, 1.5),
    (4.2, 3.4, 1.5),
    (4.3, 2.1, 1.5),
    (2.1, 2.2, 1),
    (2.2, 2.3, 4),
    (2.2, 3.1, 1),
    (2.3, 1.1, 4.5),
    (2.3, 3.2, 2),
    (1.1, 3.3, 2),
    (3.1, 3.2, 2.5),
    (3.1, 3.4, 2.5),
    (3.2, 3.3, 2.5),
    (3.3, 3.4, 2.5),
    (3.3, 0.1, 1),
])

pos = nx.get_node_attributes(G, "pos")
G_initial = G.__class__()
G_initial.add_nodes_from(G)
G_initial.add_edges_from(G.edges)

start_node = 0.0
end_node = 0.1

# add remaining edges to make a complete graph (every node is connected to every node)
for i in G.nodes:
    for j in G.nodes:
        if i < j:
            if G.has_edge(i, j) is False:
                weight = nx.dijkstra_path_length(G, i, j)
                G.add_weighted_edges_from([(i, j, weight)])
        elif i == j:
            if G.has_edge(i, j) is False:
                weight = 0
                G.add_weighted_edges_from([(i, j, weight)])

'''
input order with product IDs(EAN), product name, location node, weight in g/ml, packaging type, and 
temperature zone (product ID, product name, fixture number, weight, packaging type, temp zone)

packaging is categorized as: 
0(glass bottle/container), 
1(hard plastic case/box/bottle), 
2(paper/plastic cartons), and 
3(loose/no packaging)

temp zones are defined as:
0(open items), 
1(milk/beverages), 
2(frozen food), and 
3(warm/bakery items)
'''
order_attr = [[4260135129403, "Sea Salt Kartoffelchips", 16, 150, 3, 0],
              [4260572920106, "Bio Jasmin Reis", 5, 500, 2, 0],
              [7610900256891, "Dessert Tiramisu", 23, 90, 2, 1],
              [42428251, "Classic Milchalternative", 25, 1000, 2, 1],
              [4270002422711, "Olivenöl", 14, 1000, 0, 0],
              [42345893, "Limonade Ingwer", 4, 500, 0, 0],
              [9120083540024, "Active Body Wash", 10, 300, 1, 0],
              [8710573426467, "Nuss Nougat BrotSpreads", 11, 350, 0, 0],
              [5060120281616, "Haselnut Drink", 26, 1000, 1, 1],
              [4260159950519, "Basilikum Pesto", 12, 180, 0, 0],
              [4018722343127, "Bio Hartweizen Fusilli", 6, 500, 3, 0],
              [4260519490013, "Frühsport Freunde Müsli", 9, 280, 2, 0],
              [000000, "Butter Croissant", "Bakery", 70, 3, 3]
              ]
print("\nInput order is: ", *order_attr, sep="\n")

# assign the approximate node to each product by checking for the "fixture" number
for i in order_attr:
    if type(i[2]) is int:
        if 1 <= i[2] <= 3:
            i.insert(3, 1.1)
        elif 4 <= i[2] <= 6:
            i.insert(3, 2.3)
        elif 7 <= i[2] <= 8:
            i.insert(3, 2.2)
        elif i[2] == 9:
            i.insert(3, 2.1)
        elif i[2] == 10:
            i.insert(3, 3.1)
        elif 11 <= i[2] <= 13:
            i.insert(3, 3.4)
        elif i[2] == 14:
            i.insert(3, 3.3)
        elif 15 <= i[2] <= 17:
            i.insert(3, 3.2)
        elif 18 <= i[2] <= 19:
            i.insert(3, 4.3)
        elif 20 <= i[2] <= 22:
            i.insert(3, 4.2)
        elif 23 <= i[2] <= 26:
            i.insert(3, 4.1)
    elif type(i[2]) is str and i[2] == "Bakery":
        i.insert(3, 4.3)

# create the walkpath (node sequence) according to input order
sequence: list = [i[0] for i in order_attr]


# DEAP One Max Problem adapted for our problem statement

# define the function for creating a random node sequence from the input order node sequence
# to pass to the genetic algorithm
def randOrder():
    randsequence = random.sample(sequence, len(sequence))
    return randsequence


# create the classes FitnessMin and Individual for the genetic algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
# define 'attr_node' to be an attribute ('gene') which corresponds to node sequence
# sampled from the collection of nodes from "sequence"
toolbox.register("attr_node", randOrder)

# Structure initializers
# define 'individual' to be an individual consisting of 'attr_node' element ('gene')
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_node)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# evaluate fitness based on location
def location_fitness(individual):
    ind_route = []
    for i in individual:
        for j in order_attr:
            if i == j[0]:
                ind_route.append(j[3])
    ind_route.insert(0, start_node)
    ind_route.append(end_node)
    path_length = nx.path_weight(G, ind_route, "weight")
    return path_length


# evaluate fitness based on weight
def weight_fitness(individual):
    ind_weights = []
    for i in individual:
        for j in order_attr:
            if i == j[0]:
                ind_weights.append(j[4])

    weight_fit = 0
    weight_penalty = 0
    for i in range(len(ind_weights)):
        for j in range(len(ind_weights)):
            if i < j:
                if ind_weights[i] < ind_weights[j]:
                    weight_penalty = (ind_weights[j] - ind_weights[i])*(j - i)*0.01
                    weight_fit += weight_penalty
    return weight_fit


# evaluate fitness based on packaging
def packaging_fitness(individual):
    ind_packaging = []
    for i in individual:
        for j in order_attr:
            if i == j[0]:
                ind_packaging.append(j[5])

    packaging_fit = 0
    packaging_penalty = 0
    for i in range(len(ind_packaging)):
        for j in range(len(ind_packaging)):
            if i < j:
                if ind_packaging[i] > ind_packaging[j]:
                    packaging_penalty = (ind_packaging[i] - ind_packaging[j])*(j - i)
                    packaging_fit += packaging_penalty
    return packaging_fit


# evaluate fitness based on temperature zone
def temp_zone_fitness(individual):
    ind_temp_zone = []
    for i in individual:
        for j in order_attr:
            if i == j[0]:
                ind_temp_zone.append(j[6])

    temp_zone_fit = 0
    temp_zone_penalty = 0
    for i in range(len(ind_temp_zone)):
        for j in range(len(ind_temp_zone)):
            if i < j:
                if ind_temp_zone[i] > ind_temp_zone[j]:
                    temp_zone_penalty = (ind_temp_zone[i] - ind_temp_zone[j])*(j - i)
                    temp_zone_fit += temp_zone_penalty
    return temp_zone_penalty


# the goal ('fitness') function to be maximized
def evalOneMin(individual):
    overall_fitness = location_fitness(individual) + 2*(weight_fitness(individual)) \
                      + 5*(packaging_fitness(individual)) + 5*(temp_zone_fitness(individual))
    if len(individual) == len(set(individual)):
        return (overall_fitness,)
    else:
        return (1000,)


# ----------
# Operator registration
# ----------
# register the goal / fitness function

toolbox.register("evaluate", evalOneMin)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# shuffle each attribute/gene of 0.05
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------


def main():

    # create an initial population of 500 individuals (where
    # each individual is a list of nodes)
    pop = toolbox.population(n=500)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    print("\nStart of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of the population
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while min(fits) > 1 and g < 100:
        # A new generation
        g = g + 1
        print("\n-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("\nBest individual is %s, with overall fitness = %s" % (best_ind, best_ind.fitness.values[0]))

    # plot subgraph showing best route
    print("\nOptimized order sequence is:")
    best_sequence = []
    best_order =[]
    for i in best_ind:
        for j in order_attr:
            if i == j[0]:
                best_sequence.append(j[3])
                best_order.append(j)
    print(*best_order, sep="\n")
    best_sequence.insert(0, start_node)
    best_sequence.append(end_node)
    H = G_initial.subgraph(best_sequence)
    nx.draw(G_initial, pos, with_labels=True, node_size=500)
    nx.draw(H, pos, with_labels=True, node_size=500, node_color="r")
    nx.draw_networkx_edges(H, pos, H.edges, width=2, edge_color="r")
    plt.show()


if __name__ == "__main__":
    main()
