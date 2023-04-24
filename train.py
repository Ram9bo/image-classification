"""
Data loading and preparation.
"""

import numpy as np
import network
import dataloader

def train_network():
    model = network.compile_model()

    data = dataloader.load_data()
