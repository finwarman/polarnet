#! /usr/bin/env python3
'''
Trains a simple neural network to convert from polar coordinates to Cartesian coordinates.

Adapted from Chapter 36 of the A. K. Dewdney's `New Turing Omnibus`,
describing an implementation of POLARNET (Rietman-Frye)

Further neural net info: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
'''

import numpy as np
import math
from random import random, seed
from statistics import mean

import termplotlib as tpl
# - for plotting graphs in terminal (requires `gnuplot`)
import colorama
from colorama import Fore, Back, Style
# - for colourising output
colorama.init()  # allows escaping of ANSI codes on windows

# ============ PROGRAM PARAMETERS ============

# The number of iterations to train the network (i.e. predict, then back propagate from errors)
TRAINING_EPOCHS = 100000

# The number of neurons per single medial layer (Dewdney: "requires only 30 to 40 neurons")
MEDIAL_NEURONS = 30

# Hyperparameter determining the amount that weights are updated during an iteration of training (back prop.)
# Helps to prevent overshoot' (e.g. missing a local optima, by being too heavily skewed by a singly large error)
#   smaller = more training epochs required, may get stuck looking for a solution
#   larger = rapid changes and fewer epochs required, may converge prematurely on suboptimal solution
# (Dewdney: around 0.1, for 10^4 iterations)
# See: https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
LEARNING_RATE = 0.01

# An additional input neuron alongside the two polar coordinate inputs.
# Shifts the activation function (e.g. left or right), to enable successful learning for near-zero values.
# See: https://www.pico.net/kb/the-role-of-bias-in-neural-networks
BIAS = 2

# The number of input and output neurons
# Input: Polar Coords (radius, angle) + a bias node
NEURONS_IN = 3

# Output: Cartesian Coords (x, y)
NEURONS_OUT = 2

# =============================================

"""
compute_target -
    determine the ground truth values for cartesian coordinates, from the polar input, by formula.
    this is used to determine the error of the model, and therefore train it during backpropagation
"""


def compute_target(distance, angle, *args):
    return [distance * np.cos(angle), distance * np.sin(angle)]


"""
hyperbolic_tanget -
    the hyperbolic tangent function, tanh(x), to nonlinearly 'squeeze' the signal between -1 and 1
    (= sinh(x) / cosh(x) = (e^x - e^-x)/(e^-x - e^x))
    sigmoidal function to allow network to respond nonlinearly to its environment,
    & keeps values of intermediate neurons bounded (i.e. between 0 and 1)
    [other sigmoidal functions: tanh, arctan, fermi, etc.]
"""


def hyberbolic_tangent(x):
    return np.tanh(x)


"""
convert_coords -
    generate the output neurons value from the given input, and determine error.
        i.e. - estimate output cartesian coordinates from given polar coords
    by default, back propagates to train network based on error by calling `back_propagation`
    (See: Dewdney 246. 'PART ONE: COORDINATE CONVERSION')
"""


def convert_coords(input_neurons, synone, syntwo, back_propagate=True):
    medin = [0 for _ in range(MEDIAL_NEURONS)]
    medout = [0 for _ in range(MEDIAL_NEURONS)]

    for i in range(MEDIAL_NEURONS):
        for j in range(NEURONS_IN):
            # todo? - flip synone to match book with [j][i]
            medin[i] += synone[i][j] * input_neurons[j]
        medout[i] = hyberbolic_tangent(medin[i])

    output = [0 for _ in range(NEURONS_OUT)]
    error = [0 for _ in range(NEURONS_OUT)]

    target = compute_target(*input_neurons)

    for i in range(NEURONS_OUT):
        for j in range(MEDIAL_NEURONS):
            output[i] = output[i] + syntwo[j][i] * medout[j]
        error[i] = target[i] - output[i]

    # limit values in list to range(-1, 1) to prevent overflow in sigmoid
    medin = np.clip(medin, -1, 1)
    medout = np.clip(medout, -1, 1)

    if back_propagate:
        # train network from error, default True
        back_propagation(input_neurons, synone, syntwo, medin, medout, error)

    # return the average error of both outputs (standard error, E = sqrt(e1^2 + e2^2) )
    avg_error = math.sqrt(error[0]**2 + error[1]**2)
    return avg_error


"""
back_propagation -
    adjust the synaptic weights, from back (output) to front (input) in the network.
    the adjustment of these weights depends on the error calculated during the coordinate conversion (output estimation),
    alongside the 'rate' parameter to adjust how much this error can affect the weight adjustment in a single iteration.
    (See: Dewdney 247. 'PART TWO: BACK PROPAGATION ALGORITHM')
"""


def back_propagation(input_neurons, synone, syntwo, medin, medout, error):
    # adjust second synaptic layer
    for i in range(NEURONS_OUT):
        for j in range(MEDIAL_NEURONS):
            syntwo[j][i] += LEARNING_RATE * medout[j] * error[i]

    # derive the sigmoidal signal
    sigma = [0 for _ in range(MEDIAL_NEURONS)]
    sigmoid = [0 for _ in range(MEDIAL_NEURONS)]

    for i in range(MEDIAL_NEURONS):
        for j in range(NEURONS_OUT):
            sigma[i] += error[j] * syntwo[i][j]
        # the derivative of the sigmoidal function (i.e. tanh(x)
        sigmoid[i] = 1 - (medin[i] ** 2)

    # adjust the first synaptic layer
    for i in range(NEURONS_IN):
        for j in range(MEDIAL_NEURONS):
            delta = LEARNING_RATE * sigmoid[j] * sigma[j] * input_neurons[i]
            synone[j][i] = synone[j][i] + delta

# ===== HELPER FUNCTIONS =======


def create_synapses(ports, med_neurons):
    synapses = []
    # initialise each between 0 and 1
    for i in range(med_neurons):
        synapses.append([])
        for _ in range(ports):
            synapses[i].append(0.1 * random())
    return synapses


def random_polar_coord():
    radius = random()  # within unit circle
    angle = random() * 2 * np.pi  # angle in radians (0 to 2PI)
    return (radius, angle)

# ===============================


# seed(10)  # Set seed for deterministic pseudo-random numbers - useful for testing

synone = create_synapses(NEURONS_IN, MEDIAL_NEURONS)
syntwo = create_synapses(NEURONS_OUT, MEDIAL_NEURONS)

error_over_time = []
last_error = None

ROLLING_AVG_WINDOW = TRAINING_EPOCHS // 1000
avg_queue = []  # rolling average of last n epochs

print("Training for {} epochs...".format(TRAINING_EPOCHS))
print("Learning Rate: {}".format(LEARNING_RATE))

print("\n(Average error taken from last {} samples)".format(ROLLING_AVG_WINDOW))

for count in range(1, TRAINING_EPOCHS + 1):
    # 2.1 - Select Random Polar Coordinate (in unit circle)
    radius, angle = random_polar_coord()
    input_neurons = [radius, angle, BIAS]

    # 2.2, 2.3 - Convert Coords, and Back Propagate
    avg_error = convert_coords(input_neurons, synone, syntwo)
    # (avg_error refers to the average error of both output neurons in the last single run)

    # Update rolling average
    avg_queue.append(avg_error)
    if len(avg_queue) > ROLLING_AVG_WINDOW:
        avg_queue.pop(0)  # dequeue
    rolling_avg_error = mean(avg_queue)

    # 2.4 - Output Error, E
    if count % (TRAINING_EPOCHS // 10) == 0:
        print("Iteration: {}".format(count).ljust(18), end=" | ")
        print("Avg Error: {:.6f}".format(rolling_avg_error), end=" ")

        # Show improvement since last recorded error
        if last_error is not None:
            diff = last_error - rolling_avg_error
            sign = "+" if diff > 0 else "-"
            difference = "({}{:.2f})".format(sign, abs(diff))
            difference = (Fore.GREEN if diff > 0 else Fore.RED) + difference
            print(difference + Style.RESET_ALL)
        else:
            print()

        last_error = rolling_avg_error

    error_over_time.append([count, avg_error])

print()

# Display error over time (epochs) as graph
x, y = zip(*error_over_time)
print("Training Results".center(60))
fig = tpl.figure()
fig.plot(x, y, label="Avg. Error", width=60, height=20)
fig.show()
print("Epoch No.".center(60))


# TODO: allow user input to test conversion on trained network

colorama.deinit()  # continue standard windows output after exit
