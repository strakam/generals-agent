#!/bin/bash

export TMPDIR=$SCRATCHDIR

HOME=/storage/praha1/home/strakam3/

# Move payload from the storage
echo "Moving data.."
cp $HOME/generals-agent/ $SCRATCHDIR -r

# Install the package
echo "Installing generals-bots.."
cd $SCRATCHDIR
git clone https://github.com/strakam/generals-bots.git
cd generals-bots/
pip3 install -e .

echo "Installing packages.."
cd $SCRATCHDIR/generals-agent/
pip3 install -e .

echo "Extracting replays..."
cd $SCRATCHDIR/generals-agent/supervised/datasets/
tar -xf above70.tar

# Run the agent
echo "Running main.py.."
cd $SCRATCHDIR/generals-agent/supervised/
python3 main.py
