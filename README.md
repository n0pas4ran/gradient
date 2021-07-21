# gradient
# my code for grad 

import random
from main import grad_step, linear_grad
from typing import  List, Tuple, Iterator, TypeVar
from scratch.linear_algebra import scalar_multiply, add, vector_mean, Vector

def gradient_step(v: Vector, grad: Vector, step_size: float)->Vector:
   """Step in gradient way"""
   step = scalar_multiply(grad, step_size)
   return add(v, step)

def linear_grad(x: float, y: float, theta: Vector)->Vector:
   """Gradient calculating"""
   slope, intercept = theta
   predict = slope * x + intercept
   error = predict - y
   grad = [2*error*x, 2*error]
   return grad

T = TypeVar('T')

def minibatches(dataset: List[T], bathc_size: int, shuffle: bool)->Iterator[List[T]]:
   
   batch_start = [start for start in range(0, len(dataset), bathc_size)]

   if shuffle: random.shuffle(batch_start)

   for start in batch_start:
      end = start + bathc_size
      yield dataset[start:end]

inputs = [(x, 20*x + 5) for x in range(-50, 50)]
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
learning_rate = 0.001

for epoch in range(500):
   for batch in minibatches(inputs, 20, True):
      grad = vector_mean([linear_grad(x, y, theta) for x, y in batch])
      theta = grad_step(theta, grad, -learning_rate)

   
slope, intercept = theta

assert 19.9<slope<20.1, 'Slope is about 20'
assert 4.9<intercept<5.1, 'Intercept is about 5'
print('Success!!!')

