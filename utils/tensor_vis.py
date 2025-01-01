import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Function to visualize the tensor
def visualize_tensor(tensor):
    num_slices, height, width = tensor.shape

    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Add space for widgets

    # Display the first slice
    img = ax.imshow(tensor[0], cmap="viridis")
    ax.set_title("Slice 1")

    # Add text annotations for the values
    annotations = []
    for i in range(height):
        row = []
        for j in range(width):
            text = ax.text(
                j, i, f"{int(tensor[0, i, j])}", ha="center", va="center", color="white"
            )
            row.append(text)
        annotations.append(row)

    # Slider for navigating slices
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Slice", 0, num_slices, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        slice_idx = int(slider.val) - 1
        img.set_data(tensor[slice_idx])
        ax.set_title(f"Slice {slice_idx + 1}")

        # Update text annotations
        for i in range(height):
            for j in range(width):
                annotations[i][j].set_text(f"{int(tensor[slice_idx, i, j])}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Function to handle keyboard events
    def on_key(event):
        current_val = slider.val
        if event.key == "left":
            new_val = max(current_val - 1, 1)
        elif event.key == "right":
            new_val = min(current_val + 1, num_slices)
        else:
            return
        slider.set_val(new_val)  # Move the slider programmatically

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Button for quitting the visualization
    ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
    button = Button(ax_button, "Quit", color="red", hovercolor="pink")

    def quit(event):
        plt.close(fig)

    button.on_clicked(quit)

    plt.show()
