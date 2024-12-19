# Use the official Python 3.12.7 Bookworm image as the base
FROM python:3.12.7-bookworm

# Set environment variables to ensure consistent behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install required Python packages
RUN pip install --no-cache-dir \
    torch \
    lightning \
    pettingzoo \
    gymnasium \
    wandb \
    numpy \
    matplotlib \
    neptune \
    networkx \
    scipy

# Set the default command to open bash
CMD ["/bin/bash"]

