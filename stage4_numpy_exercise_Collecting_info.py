import numpy as np

def collect_info(array):
    shape = array.shape
    dimensions = array.ndim
    size = array.size
    return f'Shape: {shape}; dimensions: {dimensions}; size: {size}'