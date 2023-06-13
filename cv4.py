from random import uniform
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import math
import sys
import copy
import time

class TravellingSalesman:
    def __init__(self, nr_of_cities):
        self.cities = []
        self.nr_of_cities = 0
        self.distances = []
        self.GenerateCities(nr_of_cities)
        self.visibility_matrix = []
        self.pheromone_matrix = []
        self.alpha = 1
        self.beta = 2
        self.Q = 1
        self.p = 0.5

    def GenerateCities(self, nr_of_cities):
        for i in range(0, nr_of_cities):
            x = random.uniform(0, 200)
            y = random.uniform(0, 200)
            self.AddCity(x, y)

    def AddCity(self, x, y):
        self.cities.append([len(self.cities), x, y])
        self.nr_of_cities += 1

    def CalculateDistance(self):
        for cityA in self.cities:
            for cityB in self.cities:
                distance = math.sqrt(pow((cityB[2] - cityA[2]), 2) + pow((cityB[1] - cityA[1]), 2))
                self.distances.append(distance)
                if cityA[0] == cityB[0]:
                    self.visibility_matrix.append(0)
                else:   
                    self.visibility_matrix.append(1/distance)
                self.pheromone_matrix.append(1.0)
        

    def GetDistance(self, cityA, cityB):
        return self.distances[cityA * self.nr_of_cities + cityB]
    
    def GetDistanceV(self, _visibility_matrix, cityA, cityB):
        return _visibility_matrix[cityA * self.nr_of_cities + cityB]

    def GetPheromone(self, cityA, cityB):
        return self.pheromone_matrix[cityA * self.nr_of_cities + cityB]
    
    def EvaluateRoute(self, route):
        distance = 0.0
        for i in range(1, len(route)):
            distance += self.GetDistance(route[i][0], route[i-1][0])
        return distance
    
    def GenerateRandomRoute(self, _starting_city):
        route = self.cities.copy()
        starting_city = self.cities[_starting_city]
        route.remove(starting_city)
        random.shuffle(route)
        route = self.InsertStartingAndEndingPoint(route, starting_city)
        distance = self.EvaluateRoute(route)
        return [distance, route]

    def GetBestFromPopulation(self, population):
        best = min(population, key = lambda x: x[0])
        return best

    def InsertStartingAndEndingPoint(self, route, starting_city):
        route.append(starting_city)
        route.insert(0, starting_city)
        return route


    def CombineParents(self, parentA, parentB):
        offspring = []
        split = random.randint(1, len(parentA) -2)
        for i in range(0, split):
            offspring.append(parentA[i])
        for i in range(0, len(parentB)):
            if parentB[i] in offspring:
                continue
            offspring.append(parentB[i])
        offspring.append(parentA[0])
        return offspring

    def Mutate(self, offspring):
        index1 = random.randint(1, len(offspring)- 2)
        index2 = index1
        while index1 == index2:
            index2 = random.randint(1, len(offspring) - 2)
        offspring[index1], offspring[index2] = offspring[index2], offspring[index1]
        return offspring
        


    def CalculateShortestRoute(self, starting_city, nr_of_generations, nr_of_seeds):
        population = []
        best_routes = []
        mutation_chance = 0.5
        for i in range(0, nr_of_seeds):
            population.append(self.GenerateRandomRoute(starting_city))
        best_routes.append(self.GetBestFromPopulation(population))
        for i in range(0, nr_of_generations):
            new_population = population.copy()
            for j in range(0, nr_of_seeds):
                _parentA = population[j]
                _parentB = _parentA
                while _parentA is _parentB:
                    _parentB = random.choice(population)
                parentA = _parentA[1]
                parentB = _parentB[1]
                offspringAB = self.CombineParents(parentA, parentB)
                if np.random.uniform() < mutation_chance:
                    offspringAB = self.Mutate(offspringAB)
                distance = self.EvaluateRoute(offspringAB)
                if distance < new_population[j][0]:
                    new_population[j] = [distance, offspringAB]
            best_routes.append(self.GetBestFromPopulation(new_population))
            population = new_population
        return best_routes

    def NullClmn(self, _visibility_matrix, city):
        for i in range(0, self.nr_of_cities):
            _visibility_matrix[i * self.nr_of_cities + city] = 0
            #_visibility_matrix[city * self.nr_of_cities + i] = 0
        return _visibility_matrix

    def GenerateAntsRoute(self, starting_city):
        _visibility_matrix = copy.deepcopy(self.visibility_matrix)
        route = []
        route.append(self.cities[starting_city])
        _visibility_matrix = self.NullClmn(_visibility_matrix, starting_city)
        for i in range(0, self.nr_of_cities - 1):
            probabilities = []
            _cities = []
            for j in range(0, self.nr_of_cities):
                if self.GetDistanceV(_visibility_matrix, route[-1][0], j) == 0:
                    continue
                p = (self.GetPheromone(route[-1][0], j)**self.alpha) * (self.GetDistanceV(_visibility_matrix ,route[-1][0], j))**self.beta
                probabilities.append(p)
                _cities.append(j)
            n = sum(probabilities)
            probabilities = [i/n for i in probabilities]
            for j in range(1, len(probabilities)):
                probabilities[j] += probabilities[j-1]
            rn = np.random.uniform()
            for j in range(0, len(probabilities)):
                if rn < probabilities[j]:
                    route.append(self.cities[_cities[j]])
                    _visibility_matrix = self.NullClmn(_visibility_matrix, _cities[j])
                    break
            if len(probabilities) == 1: 
                route.append(self.cities[_cities[0]])
                _visibility_matrix = self.NullClmn(_visibility_matrix, _cities[0])
               
        route.append(self.cities[starting_city])
        distance = self.EvaluateRoute(route)
        return [distance, route]

    def AdjustPheromoneMatrix(self, population, g):
        for i in range(0, len(self.pheromone_matrix)):
            self.pheromone_matrix[i] = self.pheromone_matrix[i] * self.p
        for i in range(0, len(population)):
            for j in range(1, len(population[i][1])):
                self.pheromone_matrix[population[i][1][j-1][0] * self.nr_of_cities + population[i][1][j][0]] = self.pheromone_matrix[population[i][1][j-1][0] * self.nr_of_cities + population[i][1][j][0]] + (1/population[i][0])

    def AntColonyOptimization(self, nr_of_generations):
        best_routes = []
        for i in range(0, nr_of_generations):
            population = []
            for j in range(0, self.nr_of_cities):
                route = self.GenerateAntsRoute(self.cities[j][0])
                population.append(route)
            best_routes.append(self.GetBestFromPopulation(population))
            self.AdjustPheromoneMatrix(population, i)
        return best_routes
                

def init_graph(cities):

    x = np.linspace(200, 200, 50)
    y = x

    
    fig, ax = plt.subplots()
    line, = ax.plot(x, y)

    for c in cities:
        plt.plot(c[1], c[2], 'bo')
    return fig, ax
        

nr_of_generations = 300
nr_of_seeds = 600

tsp = TravellingSalesman(20)

tsp.CalculateDistance()
start = time.time()
best = tsp.CalculateShortestRoute(0, nr_of_generations, nr_of_seeds) #genetic algo
#best = tsp.AntColonyOptimization(nr_of_generations) #ant colony optimization
end = time.time()
print(end - start)

fig, ax = init_graph(tsp.cities)
    
def visualize(i, *fargs): 
    lines = []
    #t = "G = " + str(i) + "\n" + "Best = " + str(fargs[0][i][0])
    #ax.set_title(t)
    for j in range(1, len(fargs[0][i][1])):
        _lines = plt.plot([fargs[0][i][1][j][1], fargs[0][i][1][j-1][1]], [fargs[0][i][1][j][2], fargs[0][i][1][j-1][2]])
        for line in _lines:
            lines.append(line)
    t = "G: " + str(i) + "     distance: " + str(fargs[0][i][0])
    print(t)
    return lines
                
anim = animation.FuncAnimation(fig, visualize,
                               frames=nr_of_generations, interval=50, blit=True, fargs=(best, fig))

plt.show()
