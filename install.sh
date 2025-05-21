#!/bin/bash
conda env create -f environment.yaml
conda activate dpcnet
python -m pip install -r requirements.txt
