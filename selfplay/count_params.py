#!/usr/bin/env python3
"""
Parameter counting utility for the Neural Network model.
This script creates a network with default parameters and prints a breakdown
of parameter counts for each component (backbone and heads).
"""

import torch
from network import Network
from param_counter import print_parameter_counts


def main():
    """
    Create a network with default parameters and print parameter counts.
    """
    # Create a network with commonly used parameters
    print("Creating network with default parameters...")
    network = Network(
        channel_sequence=[192, 224, 256, 256],
        repeats=[2, 2, 1, 1],
        batch_size=1
    )
    
    # Initialize the network to ensure all parameters are created
    network.reset()
    
    # Call our utility to print detailed parameter breakdown
    print_parameter_counts(network)
    
    # Additionally, print counts with and without gradients
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in network.parameters())
    
    print("\nSUMMARY:")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Print some extra model information
    print("\nEXTRA INFO:")
    print(f"Input channels: {network.n_channels}")
    print(f"History size: {network.history_size}")
    

if __name__ == "__main__":
    main() 