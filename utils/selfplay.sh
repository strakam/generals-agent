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

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOTcxZmFjYy0wOGFlLTQxZWItOTZjOC1mMzAwOTgwZGJkZDEifQ==" 

# Run the agent
echo "Running main.py.."
cd $SCRATCHDIR/generals-agent/selfplay/
python3 main.py
