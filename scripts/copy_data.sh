#!/bin/bash

# Create directories only if they don't exist
[ ! -d "data/datasets" ] && mkdir -p data/datasets
[ ! -d "data/memory" ] && mkdir -p data/memory
[ ! -d "data/hssd-hab/metadata" ] && mkdir -p data/hssd-hab/metadata

# Copy datasets
echo "Copying datasets..."
cp -r memento_data/datasets/* data/datasets/

# Copy memory
echo "Copying memory..."
cp -r memento_data/memory/* data/memory/

# Copy HSSD-HAB metadata
echo "Copying HSSD-HAB metadata..."
cp -r memento_data/hssd-hab/metadata/*.csv data/hssd-hab/metadata/

echo "Data copy completed!" 