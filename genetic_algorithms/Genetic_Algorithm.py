import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation



NUMBER_OF_CITIES = 30
MAX_X_COORDINATE = 100
MAX_Y_COORDINATE = 100
NUMBER_OF_INDIVIDUALS = 100
NUMBER_OF_PARENTS = 50
NUMBER_OF_CHILDREN = 78
NUMBER_OF_GENERATIONS = 500
MUTATION_VAR = 0.1



def plot_path(cities, path):

    x = [cities[0][0]] + [cities[i][0] for i in path] + [cities[0][0]]
    y = [cities[0][1]] + [cities[i][1] for i in path] + [cities[0][1]]  
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o') 
    plt.title("Traveling Salesman Problem Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    

    for i, city in enumerate(cities):
        plt.annotate(str(i), (city[0], city[1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.grid()
    plt.show()






def generate_cities():
    
    global NUMBER_OF_CITIES

    cities = []

    for i in range(0, NUMBER_OF_CITIES):

        repeat = True
        while repeat:
            coord_x = random.randint(0, MAX_X_COORDINATE)
            coord_y = random.randint(0, MAX_Y_COORDINATE)

            city = (coord_x, coord_y)

            if city not in cities:
                cities.append(city)
                repeat = False

    return cities



def generate_individual():
    
    city_pool = list(range(1, NUMBER_OF_CITIES))
    individual = []

    for i in range(0, NUMBER_OF_CITIES - 1):
        chosen_city = random.choice(city_pool)
        city_pool.remove(chosen_city)
        individual.append(chosen_city)

    return individual

    
def generate_population():

    global NUMBER_OF_CITIES
    global NUMBER_OF_INDIVIDUALS

    population = []

    for i in range(NUMBER_OF_INDIVIDUALS):
        
        individual = generate_individual()   
        population.append(individual)
    
    return population



def calc_distance(origin, destination):
    return math.sqrt(pow(origin[0] - destination[0], 2) + pow(origin[1] - destination[1], 2))


def fitness_func(individual, cities):

    fitness = 0
    origin_index = 0
    
    for destination_index in individual:

        fitness += calc_distance(cities[origin_index], cities[destination_index])

        origin_index = destination_index

    fitness += calc_distance(cities[origin_index], cities[0])

    fitness = 1 / fitness

    return fitness



def tournament_selection(population, cities, tournament_size=6):
    selected = []

    for _ in range(NUMBER_OF_PARENTS):
        tournament = random.sample(population, tournament_size)
        
        best_individual = max(tournament, key=lambda individual: fitness_func(individual, cities))
        selected.append(best_individual)

    return selected



def roulette_selection(population, cities):
    selected = []

    total_sum = 0
    for individual in population:
        sum += fitness_func(individual, cities)


    for _ in range(NUMBER_OF_PARENTS):

        roulette_choice = random.uniform(0, total_sum)

        counter = 0
        i = -1
        while counter < roulette_choice:
            i += 1
            counter += fitness_func(population[i], cities)

        selected.append(population[i])

    return selected



def combine_genes(parent1, parent2):
    
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(start, len(parent1) - 1)

    new_individual = [None] * (NUMBER_OF_CITIES - 1)
    new_individual[start:end] = parent1[start:end]

    for gene in parent2:
        for i in range(len(new_individual)):
            if new_individual[i] is None:
                if gene not in new_individual: 
                    new_individual[i] = gene
                    break

    return new_individual



def create_new_generation(parent_pool, population, cities):

    new_generation = []

    for i in range(1, NUMBER_OF_CHILDREN // 2):

        parent1 = random.choice(parent_pool)
        parent2 = random.choice(parent_pool)

        new_generation.append(combine_genes(parent1, parent2))
        new_generation.append(combine_genes(parent2, parent1))

    for i in range(len(new_generation), math.ceil((NUMBER_OF_INDIVIDUALS * 0.95) - 1)):
        new_generation.append(random.choice(population)[:])
        

    mutate_population(new_generation)

    while len(new_generation) < NUMBER_OF_INDIVIDUALS:
        best = max(population, key=lambda x: fitness_func(x, cities))
        population.remove(best)
        new_generation.append(best[:])


    return new_generation



def mutate_population(population):

    for individual in population:
        mutate(individual)


def mutate(individual):

    rand = random.random()
    if rand < MUTATION_VAR * 3:
        inversion_mutation(individual)

    rand = random.random() 
    if rand < MUTATION_VAR * 2:
        scramble_mutation(individual)


    rand = random.random()

    if rand < MUTATION_VAR:
        i = random.randint(0, NUMBER_OF_CITIES - 3)
        individual[i], individual[i + 1] = individual[i + 1], individual[i]



def scramble_mutation(individual):

    start = random.randint(0, len(individual) - 1)
    end = random.randint(start + 1, len(individual)) 

    subset = individual[start:end]

    random.shuffle(subset)

    individual[start:end] = subset



def inversion_mutation(individual):

    start = random.randint(0, len(individual) - 1)
    end = random.randint(start + 1, len(individual))

    individual[start:end] = individual[start:end][::-1]



def main():

    global MUTATION_VAR

    MUTATION_VAR_OR = MUTATION_VAR

    previous_average = 0

    cities = generate_cities()

    population = generate_population()

    bestie_fitness = 0
    bestie_genes = []
    evolution = []

    for i in range(NUMBER_OF_GENERATIONS):

        parent_pool = tournament_selection(population, cities, 3)
        #parent_pool = roulette_selection(population, cities)
        
        best_local_genes = max(population, key=lambda x: fitness_func(x, cities))
        best_local_fitness = fitness_func(best_local_genes, cities)

        sum = 0
        for individual in population:
            sum += fitness_func(individual, cities)

        average = sum / NUMBER_OF_INDIVIDUALS

        if bestie_fitness < best_local_fitness:

            MUTATION_VAR = MUTATION_VAR_OR

            bestie_genes = best_local_genes
            bestie_fitness = best_local_fitness
            evolution.append(best_local_genes)

        elif average < previous_average:
            if MUTATION_VAR < 0.3:
                MUTATION_VAR = MUTATION_VAR * 1.1
        elif not MUTATION_VAR / 1.1 < MUTATION_VAR_OR:
            MUTATION_VAR = MUTATION_VAR / 1.1

        previous_average = average

        population = create_new_generation(parent_pool, population, cities)



    fig, ax = plt.subplots()

    ax.set_xlim(0, MAX_X_COORDINATE)
    ax.set_ylim(0, MAX_Y_COORDINATE)

    scat = ax.scatter([city[0] for city in cities], [city[1] for city in cities], color='blue')

    line, = ax.plot([], [], color='red')

    def update(frame):
        if frame < len(evolution):
            individual = evolution[frame] 
        else:
            individual = evolution[-1]  

        x = [cities[0][0]] + [cities[i][0] for i in individual] + [cities[0][0]] 
        y = [cities[0][1]] + [cities[i][1] for i in individual] + [cities[0][1]]  

        line.set_data(x, y)
        return line, scat


    pause_frames = 70  
    total_frames = len(evolution) + pause_frames

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True, interval=100)

    #ani.save("genetic_algorithm_evolution.gif", writer="pillow")
    

    plot_path(cities, bestie_genes)






if __name__ == "__main__":
    main()