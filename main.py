import torch

def clean_checkpoint(checkpoint_path, output_path=None):
    """
    Loads a checkpoint, removes the config object, and re-saves it.
    
    Args:
        checkpoint_path (str): Path to the original checkpoint file.
        output_path (str, optional): Path to save the cleaned checkpoint.
                                     If not provided, will overwrite the original.
    """
    # Load the checkpoint from the specified file.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Remove the problematic config if it exists.
    if "config" in checkpoint:
        print("Removing 'config' from checkpoint.")
        del checkpoint["config"]
    else:
        print("No 'config' key found in checkpoint. Nothing to remove.")
    
    # Determine the output path: either overwrite or save to a new file.
    if output_path is None:
        output_path = checkpoint_path
    
    # Save the cleaned checkpoint.
    torch.save(checkpoint, output_path)
    print(f"Checkpoint successfully saved to '{output_path}' without the config.")

def main():
    # Update these paths as appropriate.
    original_checkpoint = "path/to/your/original_checkpoint.ckpt"
    cleaned_checkpoint = "path/to/your/cleaned_checkpoint.ckpt"  # You can also use original_checkpoint to overwrite.
    clean_checkpoint(original_checkpoint, cleaned_checkpoint)

if __name__ == "__main__":
    main() 