import random
import math
import matplotlib.pyplot as plt
import time
import copy


TOTAL_NUMBER_OF_POINTS = 4020

#Funkcia na pocitanie dlzky danej funkcie

def func_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.2f} seconds to complete.")
        return result
    return wrapper



class Cluster: # Reprezentuje jeden klaster

    def __init__(self, first_member, first_index, centre_is_centroid):
        self.members = [first_member]
        self.indices = [first_index] # Indexy bodov v matici vzdialenosti priamo mapujuce na body v poli members
        self.distances = [0] # Sucet vzdialenosti jednotlivych bodov od vsetkych ostatnych bodov v klastri, mapuje sa priamo na body v poli members (na jednoduchsie pocitanie meodoidov)
        self.centre = first_member
        self.centre_is_centroid = centre_is_centroid # Boolova premenna urcujuca, ci je centrom klastra medoid alebo centroid


    def calc_avrg_dist_from_centre(self): # Na vypocet toho, ci klaster nema privelku priemernu vzdialenost bodov od stredu

        avrg = 0
        for member in self.members:
            avrg += Cluster_Divisor.calc_dist(member, self.centre)

        return avrg / len(self.members)


    def update_centroid(self): # Na updatovanie centroidov po zmeneni klastra

        avrg = [0, 0]
        for member in self.members:
            avrg[0] += member[0]
            avrg[1] += member[1]
        
        avrg[0] /= len(self.members)
        avrg[1] /= len(self.members)
        
        self.centre = avrg


    def update_medoid(self): # Na updatovanie medoidov po zmeneni klastra

        i = self.distances.index(min(self.distances))
        self.centre = self.members[i]


    def update_distances(self, new_indices, dist_matrix): # Na updatovanie vzdialenosti bodov od vsetkych ostatnych bodov v klastri

        for i, member_index in enumerate(self.indices):
            for new_index in new_indices:
                self.distances[i] += dist_matrix[new_index][member_index]


    def add_dist(self, new_indices, dist_matrix): # Na pridanie suctu vzdialenosti novo pridanych bodov od vsetkych ostatnych bodov v klastri

        for i in new_indices:
            new_dist = 0
            for member_index in self.indices:
                new_dist += dist_matrix[i][member_index]

            for new_index in new_indices:
                new_dist += dist_matrix[i][new_index]

            self.distances.append(new_dist)
        

    def add(self, new_cluster, dist_matrix): # Na pridanie bodov z noveho klastra do aktualneho klastra (pouziva sa pri mergovani)

        self.update_distances(new_cluster.indices, dist_matrix)
        self.add_dist(new_cluster.indices, dist_matrix)
        self.members.extend(new_cluster.members)
        self.indices.extend(new_cluster.indices)

        if self.centre_is_centroid:
            self.update_centroid()
        else:
            self.update_medoid()



class Cluster_Map: # Na generaciu, reprezentaciu a vizualizaciu vsetkych bodov v grafe

    @func_timer
    def __init__(self, coordinate_bounds, offset_bounds, init_points_count, total_points_count, centre_is_centroid):
        
        self.min_coordinate = coordinate_bounds[0] # Urcuju maximalne suradnice pre bod (su teda hranicou grafu) 
        self.max_coordinate = coordinate_bounds[1]

        self.min_offset = offset_bounds[0] # Urcuju maximalny offset pri generacii bodov
        self.max_offset = offset_bounds[1]
        
        self.init_points_count = init_points_count # Urcuju pocet pociatocnych nahodne vygenerovanych bodov, od ktorych sa generuju zvysne body
        self.total_points_count = total_points_count # Urcuje pocet vsetkych bodov (vcetne inicializacnych)

        self.init_map()
        self.init_clus_map(centre_is_centroid)

        self.clus_divisor = Cluster_Divisor()
        self.point_dist_matrix = self.clus_divisor.calc_dist_matrix(self.map) # Matica vzdialenosti bodov (pouziva sa na rychlejsi vypocet medoidov)
        self.clus_dist_matrix = self.clus_divisor.calc_dist_matrix(self.map) # Matica vzialenosti klastrov (pouziva sa na aglomerativny algoritmus)


    @func_timer # Funkcia na spustenie aglomerativneho algoritmu
    def divide(self, dist_limit):
        self.clus_divisor.divide(self.clus_map, self.clus_dist_matrix, self.point_dist_matrix, dist_limit)


    def rand_point(self): # Generacia nahodneho bodu

        coordinate_x = random.randint(self.min_coordinate, self.max_coordinate)
        coordinate_y = random.randint(self.min_coordinate, self.max_coordinate)
        new_point = (coordinate_x, coordinate_y)

        if new_point in self.map:
            return self.rand_point()
        
        return new_point


    def rand_offset(self): # Generacia nahodneho offsetu
        return (random.randint(self.min_offset, self.max_offset), random.randint(self.min_offset, self.max_offset))


    def new_neighbor(self): # Generacia noveho susedneho bodu (teda bodu generovaneho cez offset)
        
        rand_point = random.choice(self.map)
        offset = self.rand_offset()
        new_point = (rand_point[0] + offset[0], rand_point[1] + offset[1])

        if (new_point[0] < self.min_coordinate or new_point[0] > self.max_coordinate 
            or new_point[1] < self.min_coordinate or new_point[1] > self.max_coordinate
            or new_point in self.map):
            return self.new_neighbor()
        
        return new_point
    

    def init_map(self): # Inicializuje graf bodov
        
        self.map = []
        for _ in range(self.init_points_count):
            self.map.append(self.rand_point())
        
        for _ in range(self.init_points_count, self.total_points_count):
            self.map.append(self.new_neighbor())


    def init_clus_map(self, centre_is_centroid): # Inicializuje pole klastrov

        self.clus_map = []
        for i, member in enumerate(self.map):
            self.clus_map.append(Cluster(member[:], i, centre_is_centroid))
    
    
    def plot(self): # Vykresli graf

        plt.figure(figsize=(8, 8))

        for i, clus in enumerate(self.clus_map):
            x_coords = [point[0] for point in clus.members]
            y_coords = [point[1] for point in clus.members]
            plt.scatter(x_coords, y_coords, s=10, alpha=0.2, label=f'Group {i + 1}')
            plt.scatter(clus.centre[0], clus.centre[1], facecolors="red", edgecolors="black", s=10, label=f"Group {i + 1}")


        plt.title("Cluster Map")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.xlim(self.min_coordinate, self.max_coordinate)
        plt.ylim(self.min_coordinate, self.max_coordinate)
        plt.grid(True)

        plt.show()
        


class Cluster_Divisor: # Na vykonanie aglomerativneho algoritmu

    def __init__(self):
        return


    @staticmethod # Pocitanie vzdialenosti dvoch bodov
    def calc_dist(src, dest):
        return math.sqrt((src[0] - dest[0]) ** 2 + (src[1] - dest[1]) ** 2)


    @func_timer
    def calc_dist_matrix(self, points): # Pocitanie matice vzdialenosti

        dist_matrix = [[0 for _ in range(len(points))] for _ in range(len(points))]

        for i in range(len(points)):
            src = points[i]

            for j in range(i + 1, len(points)):
                dest = points[j]
                dist = Cluster_Divisor.calc_dist(src, dest)
                
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist

        return dist_matrix
    

    def find_nearest_clusters(self, dist_matrix): # Najde najblizsie dva klastre (v matici vzdialenosti klastrov)

        min_dist = float("inf")
        min_dist_indices = ()

        for x in range(len(dist_matrix)):
            for y in range(x + 1, len(dist_matrix)):
                if dist_matrix[x][y] < min_dist:
                    min_dist = dist_matrix[x][y]
                    min_dist_indices = (x, y)

        return  min_dist_indices


    def rm_cluster_in_dist_matrix(self, dist_matrix, index): # Vymaze klaster z matice vzdialenosti (pouziva sa pri mergovani klastrov)
        
        dist_matrix.pop(index)
        for y in range(len(dist_matrix)):
            dist_matrix[y].pop(index)


    def update_cluster_in_dist_matrix(self, clus_map, dist_matrix, index): # Updatne vzdialenosti v matici vzdialenosti (pouziva sa pri mergovani klastrov)

        for x in range(len(dist_matrix)):
            dist = Cluster_Divisor.calc_dist(clus_map[index].centre, clus_map[x].centre)
            dist_matrix[x][index] = dist

        for y in range(len(dist_matrix)):
            dist = Cluster_Divisor.calc_dist(clus_map[index].centre, clus_map[y].centre)
            dist_matrix[index][y] = dist

    
    def merge_clusters(self, clus_map, cluster_indices, clus_dist_matrix, point_dist_matrix): # Mergovanie klastrov
        
        clus_map[cluster_indices[0]].add(clus_map[cluster_indices[1]], point_dist_matrix) # Klaster B sa prida do klastra A
        
        self.update_cluster_in_dist_matrix(clus_map, clus_dist_matrix, cluster_indices[0]) # Hodnoty pre klaster A sa updatnu
        
        self.rm_cluster_in_dist_matrix(clus_dist_matrix, cluster_indices[1]) # Klaster B sa v matici vzdialenosti vymaze

        clus_map.pop(cluster_indices[1]) # Klaster B sa vymaze v liste klastrov


    def check_cluster_limit(self, cluster_1, cluster_2, dist_matrix, dist_limit): # Skontroluje, ci mozeme mergnut dva klastre a nepresiahnut limit
        
        cluster_1 = copy.deepcopy(cluster_1)
        cluster_2 = copy.deepcopy(cluster_2)

        cluster_1.add(cluster_2, dist_matrix)
        if cluster_1.calc_avrg_dist_from_centre() > dist_limit:
            return True
        
        return False


    def divide(self, clus_map, clus_dist_matrix, point_dist_matrix, dist_limit): # Na vykonanie aglomerativneho algoritmu

        while len(clus_map) > 1: # Kym existuje viac ako jeden klaster (pre extremne pripady kedy by vsetky body boli blizko seba)
            cluster_indices = self.find_nearest_clusters(clus_dist_matrix) # Najde najblizsie klastre

            clus_1 = clus_map[cluster_indices[0]]
            clus_2 = clus_map[cluster_indices[1]]

            if self.check_cluster_limit(clus_1, clus_2, point_dist_matrix, dist_limit): # Skontroluje, ci ich moze mergnut bez presiahnutia limitu
                break

            self.merge_clusters(clus_map, cluster_indices, clus_dist_matrix, point_dist_matrix) # Mergne klastre






def main():

    centre_is_centroid = ""
    print("Should centre be centroid or medoid? (input '1' for centroid and '2' for medoid)")
    centre_is_centroid = input()

    if centre_is_centroid == "1":
        centre_is_centroid = True
    elif centre_is_centroid == "2":
        centre_is_centroid = False
    else:
        print("Unkown input.")
        return 1

    clusters = Cluster_Map((-5000, 5000), (-100, 100), 20, TOTAL_NUMBER_OF_POINTS, centre_is_centroid)

    clusters.divide(500)

    clusters.plot()





if __name__ == "__main__":
    main()