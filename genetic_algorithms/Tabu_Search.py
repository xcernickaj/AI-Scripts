import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation



NUMBER_OF_CITIES = 30
NUMBER_OF_GENERATIONS = 500
NUMBER_OF_VARIATIONS = 200
TABU_LIST_LENGTH = 4
MAX_X_COORDINATE = 100
MAX_Y_COORDINATE = 100



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





def generate_individual():
    
    city_pool = list(range(1, NUMBER_OF_CITIES))
    individual = []

    for i in range(0, NUMBER_OF_CITIES - 1):
            chosen_city = random.choice(city_pool)
            city_pool.remove(chosen_city)

            individual.append(chosen_city)

    return individual




def generate_variations(current_best):
     
    variations = [current_best]

    while len(variations) < NUMBER_OF_VARIATIONS:
        new_variation = current_best.copy()

        number_of_swaps = 1
        rand = random.random()

        if rand < 0.1:
            scramble_mutation(new_variation)
        elif rand < 0.3:
            inversion_mutation(new_variation)
        elif rand < 0.7:
            number_of_swaps = random.randint(2, 10)

        for _ in range(number_of_swaps):
            i = random.randint(0, len(new_variation) - 2)

            new_variation[i], new_variation[i + 1] = new_variation[i + 1], new_variation[i]

        if new_variation not in variations:
            variations.append(new_variation)
    
    return variations    



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






def main():

    cities = generate_cities()
    current_best = generate_individual()

    failure_count = 0
    tabu_list = []
    bestie_fitness = 0
    bestie_genes = []
    descending = False
    evolution = [current_best]

    for _ in range(NUMBER_OF_GENERATIONS):
        variation_space = generate_variations(current_best)
        
        while True:
            new_best = max(variation_space, key=lambda individual: fitness_func(individual, cities))

            if fitness_func(new_best, cities) <= fitness_func(current_best, cities):
                failure_count += 1
            elif new_best in tabu_list:
                variation_space.remove(new_best)
                continue
            else:
                current_best = new_best
                failure_count = 0
                descending = False
                break


            if failure_count > 2 or descending:
                tabu_list.append(current_best)
                if len(tabu_list) > TABU_LIST_LENGTH:
                    tabu_list.pop(0)
                
                failure_count = 0
                descending = True

                while True:
                    variation_space.remove(current_best)
                    current_best = max(variation_space, key=lambda individual: fitness_func(individual, cities))

                    if current_best in tabu_list:
                        continue
                    else:
                        break
                break
            else:
                break
            
        if bestie_fitness < fitness_func(current_best, cities):
            bestie_fitness = fitness_func(current_best, cities)
            bestie_genes = current_best
            evolution.append(current_best)




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

    #ani.save("tabu_search_evolution.gif", writer="pillow")


    plot_path(cities, bestie_genes)






if __name__ == "__main__":
    main()
