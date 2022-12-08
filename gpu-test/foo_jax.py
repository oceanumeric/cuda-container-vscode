import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import os
import sys


# check executable path
print(sys.executable)
# set memory allocation 
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'


key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)