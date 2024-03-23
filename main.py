import tensorflow as tf
import matplotlib.pyplot as plt
import random as random

def activation_function(linear_combination):
 return tf.sigmoid(linear_combination)
def error_function(actual_output, expected_output):
 return 0.5 * pow(expected_output - actual_output, 2)
def randomize(weights):
 for i in range(0, 3):
  weights[i] = random.uniform(0, 1)

learning_coeff = 0.5
max_error = 0.01
epochs = 0
learn = True
training_set = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

neuron1_weights = [0, 0, 0]
neuron2_weights = [0, 0, 0]
neuron3_weights = [0, 0, 0]
randomize(neuron1_weights)
randomize(neuron2_weights)
randomize(neuron3_weights)
print(f"Wagi neuronu1: {neuron1_weights}")
print(f"Wagi neuronu2: {neuron2_weights}")
print(f"Wagi neuronu3: {neuron3_weights}")

while learn:
 epochs += 1
 error_sum = 0
 for i in range(4):
  x1 = training_set[i][0]
  x2 = training_set[i][1]
  expected_output = training_set[i][2]

  y1 = activation_function(neuron1_weights[0]*x1 + neuron1_weights[1]*x2 - neuron1_weights[2])
  y2 = activation_function(neuron2_weights[0]*x1 + neuron2_weights[1]*x2 - neuron2_weights[2])
  z = activation_function(neuron3_weights[0]*y1 + neuron3_weights[1]*y2 - neuron3_weights[2])
  error_sum += error_function(z, expected_output)

  neuron1_weights[0] += learning_coeff*y1*(1-y1)*(expected_output - z)*z*(1-z)*neuron3_weights[0]*x1
  neuron1_weights[1] += learning_coeff*y1*(1-y1)*(expected_output - z)*z*(1-z)*neuron3_weights[0]*x2
  neuron1_weights[2] -= learning_coeff*y1*(1-y1)*(expected_output - z)*z*(1-z)*neuron3_weights[0]

  neuron2_weights[0] += learning_coeff*y2*(1-y2)*(expected_output - z)*z*(1-z)*neuron3_weights[1]*x1
  neuron2_weights[1] += learning_coeff*y2*(1-y2)*(expected_output - z)*z*(1-z)*neuron3_weights[1]*x2
  neuron2_weights[2] -= learning_coeff*y2*(1-y2)*(expected_output - z)*z*(1-z)*neuron3_weights[1]

  neuron3_weights[0] += learning_coeff*(expected_output - z)*z*(1-z)*y1
  neuron3_weights[1] += learning_coeff*(expected_output - z)*z*(1-z)*y2
  neuron3_weights[2] -= learning_coeff*(expected_output - z)*z*(1-z)
 print(error_sum.numpy())
 if error_sum <= max_error:
  learn = False

print("Epochs: ", epochs)




for i in range(4):
 x1 = training_set[i][0]
 x2 = training_set[i][1]
 expected_output = training_set[i][2]
 y1 = activation_function(neuron1_weights[0]*x1 + neuron1_weights[1]*x2 - neuron1_weights[2])
 y2 = activation_function(neuron2_weights[0]*x1 + neuron2_weights[1]*x2 - neuron2_weights[2])
 z = activation_function(neuron3_weights[0]*y1 + neuron3_weights[1]*y2 - neuron3_weights[2])
 print(f"For x1 = {x1} and x2 = {x2} Output: {z.numpy()}")
