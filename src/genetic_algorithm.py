import json
import random
from dataclasses import replace
from operator import sub
from time import perf_counter

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from filetools import *
from fitutils import *
from imtools import *
from utils import *

DIRECT_RESULT = True

def order_by_fitness(population, fitness):
    ordered_indexes = np.argsort(fitness) # ascending order
    fitness = fitness[ordered_indexes]
    population = population[ordered_indexes]
    return (population, fitness)

def get_result(image: np.ndarray, s2: float, p: float) -> float:
    if DIRECT_RESULT:
        params, _ = model_params()
        params[2] = s2
        params[3] = p
        y_hat = model(params)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(image, y_hat))
        loss = tf.reduce_mean(loss)
        return loss.numpy()
    else:
        _, losses = fit(image, max_epoch=1000, initial_params=(s2, p))
        return float(np.mean(losses))

def get_fitness(image: np.ndarray, population: np.ndarray, size: int) -> np.ndarray:
    fitness = [0]*size
    for i in range(size):
        # print(f'\nINDIVIDUAL {i}')
        result = get_result(image, population[i][0], population[i][1])
        if np.isnan(result):
            result = 1000
        # print(f'Obtained loss: {result}')
        fitness[i] = result
    return np.array(fitness)

def get_parents(population: np.ndarray, fitness: np.ndarray, n: int) -> tuple:
    population, fitness = order_by_fitness(population, fitness)
    return (population[: n].copy(), fitness[: n].copy())

def crossover(parents: np.ndarray, dims: tuple) -> np.ndarray:
    children = np.zeros(shape=dims,dtype=float)
    for i in range(dims[0]):
        p1, p2 = np.random.choice(parents.shape[0], 2, replace=False)
        k = random.randint(0,dims[1])
        children[i][:k] = parents[p1][:k]
        children[i][k:] = parents[p2][k:]
    return children

def mutation(children: np.ndarray, tax: float, limits: tuple = (0.01, 1)) -> np.ndarray:
    m = int(round(children.shape[0]*tax))
    coord0 = np.random.choice(children.shape[0], m, replace=False)
    coord1 = np.random.choice(children.shape[1], m, replace=True)
    
    for i, j in zip(coord0, coord1):
        k = np.random.uniform(limits[0],limits[1])
        children[i][j] = k
        
    return children

def new_population(image: np.ndarray, parents: np.ndarray, fitness: np.ndarray, children: np.ndarray, dims: tuple) -> tuple:    
    new_pop = np.zeros(shape=dims, dtype=float)
    new_fit = np.zeros(shape=dims[0], dtype=float)
    
    proportion = 0.3
    idx = int(proportion*dims[0])
    
    new_pop[: idx] = parents[: idx]
    new_fit[: idx] = fitness[: idx]
    
    children_fit = get_fitness(image, children, children.shape[0])
    children, children_fit = order_by_fitness(children, children_fit)    
    
    new_pop[idx:] = children[:-idx] 
    new_fit[idx:] = children_fit[:-idx] 
    
    return (new_pop, new_fit)

EXPERIMENTS = [
    '10-microns particles-60X',
    'Isolada 3--3',
    'Isolada 3--2',
    'Isolada-2-10 um',
    'Calibration2_Single Cell',
    'Calibration1_Single Cell',
    'Four-mixing particles together',
    'Several 10-micron-particles together',
    'Calibration 10-microns',
    '30 microns-beads-60X-measuring 2',
    'Calibration-1-4 Cells',
    '3 particles_10 um',
]

experiment_index = 0

RESULT_PATH = 'genetic_all_experiments'
CONFIG_PATH = 'config'
EXPERIMENT = EXPERIMENTS[experiment_index]
CONFIG_FILENAME = os.path.join(CONFIG_PATH, EXPERIMENT + '.json')
with open(CONFIG_FILENAME, 'r') as fp:
    config = json.load(fp)
    
N_IMAGES = config['N_IMAGES']
    
if not os.path.isdir(os.path.join(RESULT_PATH, EXPERIMENT)):
    os.mkdir(os.path.join(RESULT_PATH, EXPERIMENT))

low = 0.01
high = 1
individuals = 100
genes = 2

for i in range(N_IMAGES):
    print(f'Start proccess for image {i}')    
    
    if not os.path.isdir(os.path.join(RESULT_PATH, EXPERIMENT, f'image{i}')):
        os.mkdir(os.path.join(RESULT_PATH, EXPERIMENT, f'image{i}'))

    IMAGE_INDEX = i

    diameters = []

    # Read image
    imfile = os.path.join(config['OUTPUT_PATH'], 'crop_centered', 'crop_centered_' + str(IMAGE_INDEX) + '.tif')
    image = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)

    #% Resize and pre-process image
    image = preprocess_image(image, new_shape=NEW_SHAPE)

    populationSize = (individuals,genes)


    # initial_p = np.random.uniform(low = -1, high = high, size = (individuals, 1))
    # initial_s2 = np.random.uniform(low = low, high = high, size = (individuals, 1))
    # population = np.hstack([initial_p, initial_s2])
    population = np.random.uniform(low = low, high = high, size = populationSize)

    generations = 1000

    fitness = get_fitness(image, population, individuals)

    data_per_generation = []

    init_time = perf_counter()
    for g in tqdm(range(generations)):
        # print(f'Starting generation {g}')
        # print(f'Parent fitness: {fitness}')
        n = int(0.3*individuals)
        parents, fitness = get_parents(population,fitness,n)
        children = crossover(parents, populationSize)
        
        mutation_tax = 0.01
        children = mutation(children,mutation_tax)

        population, fitness = new_population(image, parents, fitness, children, populationSize)
        
        data_array = np.hstack([population, fitness.reshape((fitness.shape[0], 1))])
        
        data_per_generation.append(
            data_array
        )
        
        # print(f'Generation {g} finished\n')
        
    print(f'Executed in {int(round( (perf_counter() - init_time)/60 ))} minutes')
    print('Saving results to JSON')

    population, fitness = order_by_fitness(population, fitness)
    
    data_array = np.hstack([population, fitness.reshape((fitness.shape[0], 1))])
        
    data_per_generation.append(
        data_array
    )
    
    data_per_generation = np.array(data_per_generation)
    np.save(os.path.join(RESULT_PATH, EXPERIMENT, f'image{IMAGE_INDEX}',f'data{IMAGE_INDEX}.npy'), data_per_generation)

    with open(os.path.join(RESULT_PATH, EXPERIMENT, f'image{IMAGE_INDEX}',f'result{IMAGE_INDEX}.json'), 'w') as fp:
        json.dump(
            {
                's2': population[0, 0],
                'p': population[0, 1],
                'loss': fitness[0]
            },
            fp,
            indent=4
        )
    
print('Finished')
