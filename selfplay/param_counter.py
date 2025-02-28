import torch
from network import Network

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_counts(model):
    """Print the number of parameters in each component of the model."""
    total_params = count_parameters(model)
    backbone_params = count_parameters(model.backbone)
    square_head_params = count_parameters(model.square_head)
    direction_head_params = count_parameters(model.direction_head)
    value_head_params = count_parameters(model.value_head)
    
    # Calculate percentage of total for each component
    backbone_percent = backbone_params / total_params * 100
    square_head_percent = square_head_params / total_params * 100
    direction_head_percent = direction_head_params / total_params * 100
    value_head_percent = value_head_params / total_params * 100
    
    print(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    print(f"MODEL PARAMETER BREAKDOWN")
    print(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    print(f"Total parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,} ({backbone_percent:.2f}%)")
    print(f"Square head parameters: {square_head_params:,} ({square_head_percent:.2f}%)")
    print(f"Direction head parameters: {direction_head_params:,} ({direction_head_percent:.2f}%)")
    print(f"Value head parameters: {value_head_params:,} ({value_head_percent:.2f}%)")
    print(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    
    # Also count non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if non_trainable_params > 0:
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='='")

def main():
    """Initialize a network and print parameter counts."""
    # Create a network with default parameters
    network = Network(
        channel_sequence=[192, 224, 256, 256], 
        repeats=[2, 2, 1, 1],
        batch_size=1
    )
    
    # Print parameter counts
    print_parameter_counts(network)

if __name__ == "__main__":
    main() 