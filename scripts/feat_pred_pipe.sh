#!/bin/bash

set -e


jupyter nbconvert --to notebook --execute 03_feature_pipeline.ipynb
jupyter nbconvert --to notebook --execute 05_batch_inf_pipeline.ipynb