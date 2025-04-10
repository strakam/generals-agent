import torch
import os


def clean_checkpoint(checkpoint_path: str, output_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    keys = list(checkpoint.keys())
    print(keys)
    if "state_dict" in keys:
        checkpoint["model"] = checkpoint["state_dict"]
    # Remove all keys except for "model"
    for key in keys:
        if key != "model":
            del checkpoint[key]
    # Save the cleaned checkpoint
    torch.save(checkpoint, output_path)
    print(f"Checkpoint cleaned and saved to {output_path}")


def save_checkpoint(fabric, network, optimizer, checkpoint_dir: str, iteration: int, win_rate_threshold: float):
    """Save a checkpoint in Fabric format for the current self-play iteration."""
    if fabric.is_global_zero:  # Only save on main process
        checkpoint_name = f"cp_{iteration}_threshold_{int(win_rate_threshold*100)}.ckpt"
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create state dictionary with everything needed to resume training
        state = {
            "model": network,
            "optimizer": optimizer,
        }

        # Let Fabric handle the saving
        fabric.save(checkpoint_path, state)
        print(f"Saved Fabric checkpoint for self-play iteration {iteration} to {checkpoint_path}")


def check_and_save_checkpoints(
    fabric,
    network,
    optimizer,
    checkpoint_dir: str,
    thresholds: list,
    saved_thresholds: set,
    win_rate: float,
    iteration: int,
):
    """Check if we've crossed any win rate thresholds and save checkpoints if needed."""
    newly_saved = set()
    for threshold in thresholds:
        if win_rate >= threshold and threshold not in saved_thresholds:
            # Save checkpoint
            save_checkpoint(fabric, network, optimizer, checkpoint_dir, iteration, threshold)
            newly_saved.add(threshold)

    return newly_saved


def count_parameters(model, component_name=None):
    """Count parameters in the model or a specific component."""
    if component_name is not None:
        component = getattr(model, component_name, None)
        if component is None:
            return 0
        return sum(p.numel() for p in component.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameter_breakdown(model):
    """Print detailed breakdown of model parameters by component."""
    total_params = count_parameters(model)
    backbone_params = count_parameters(model, "backbone")
    square_head_params = count_parameters(model, "square_head")
    direction_head_params = count_parameters(model, "direction_head")
    value_head_params = count_parameters(model, "value_head")
    
    # Calculate percentage of total for each component
    backbone_percent = backbone_params / total_params * 100 if total_params > 0 else 0
    square_head_percent = square_head_params / total_params * 100 if total_params > 0 else 0
    direction_head_percent = direction_head_params / total_params * 100 if total_params > 0 else 0
    value_head_percent = value_head_params / total_params * 100 if total_params > 0 else 0
    
    print("\n" + "="*50)
    print("MODEL PARAMETER BREAKDOWN")
    print("="*50)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,} ({backbone_percent:.2f}%)")
    print(f"Square head parameters: {square_head_params:,} ({square_head_percent:.2f}%)")
    print(f"Direction head parameters: {direction_head_params:,} ({direction_head_percent:.2f}%)")
    print(f"Value head parameters: {value_head_params:,} ({value_head_percent:.2f}%)")
    
    # Also count non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {total_params + non_trainable_params:,}")
    print("="*50 + "\n")
