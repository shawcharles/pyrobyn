# Robyn Quick Guide

## Overview

This guide provides a quick demonstration of the advanced marketing mix modeling using the Meta Open Source project Robyn.

## Step 0: Setup Environment

- **Installation**: 
  - Install the latest stable version from GitHub.
  - Import the required libraries.

## Step 1: Load Data

- **Simulated Dataset**: Check the simulated dataset or load your own dataset.
- **Holidays**: Check holidays from Prophet. You can add any events manually.

## Step 2: Set Model Parameters

- **Hyperparameters**: Define hyperparameters for the model.
- **Calibration**: Use experimental inputs like Facebook Lift for calibration.

## Step 3: Run Model

- **Execution**: Run the model with defined parameters.
- **Validation**: Validate the model output against business expectations.

## Step 4: Export Results

- **CSV Files**: Four CSV files are exported for further usage:
  - `pareto_hyperparameters.csv`: Hyperparameters per Pareto output model.
  - `pareto_aggregated.csv`: Aggregated decomposition per independent variable of all Pareto output.
  - `pareto_media_transform_matrix.csv`: All media transformation vectors.
  - `pareto_alldecomp_matrix.csv`: All decomposition vectors of independent variables.

## Note

For detailed information on each step, refer to the corresponding sections in the Robyn documentation.
