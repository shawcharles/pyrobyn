# Robyn: Marketing Mix Modeling in Python

A Python implementation of Meta's automated Marketing Mix Modeling (MMM) package for marketing optimization. This is a Python port of Meta's Robyn R package (v3.11.1).

## Overview
The package provides automated marketing mix modeling capabilities including:
- Ridge regression modeling with automated hyperparameter tuning
- Pareto optimization for model selection 
- Response curve analysis and ROI calculations 
- Budget allocation optimization
- One-pager model diagnostics and visualizations

## Quick Start in Colab
```python
# Install pyrobyn
#token = 'ghp_XXXX' # your access token 
#!pip install -q -U git+https://{token}@github.com/shawcharles/pyrobyn.git
!pip install -q -U git+https://@github.com/shawcharles/pyrobyn.git

# Import Required Libraries
import pandas as pd
from robyn.robyn import Robyn
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
```

## Key Components

### MMM Data Configuration
- `dep_var`: Target metric (e.g., "revenue", "conversions")
- `dep_var_type`: Type of dependent variable ("revenue" for ROI or "conversion" for CPA)
- `date_var`: Column name containing dates
- `window_start/end`: Analysis time period

#### Variable Types:
- `paid_media_spends`: Media spend columns (e.g., TV, Facebook, Search)
- `paid_media_vars`: Media exposure metrics (impressions, clicks) 
- `context_vars`: External factors (competitor activities, events)
- `organic_vars`: Non-paid marketing activities (email, social)

### Holiday Data Configuration
Components for modeling seasonality and events:
- `dt_holidays`: Holiday/events data
- `prophet_vars`: Time components ("trend", "season", "holiday")
- `prophet_country`: Country code for holidays
- `prophet_signs`: Effect direction ("default", "positive", "negative")

### Hyperparameter Configuration
Media channel parameters:
- `alphas` [0.5, 3]: Controls saturation curve shape
  - Lower (0.5-1): More diminishing returns
  - Higher (2-3): More S-shaped response
  
- `gammas` [0.3, 1]: Controls saturation curve inflection
  - Lower: Earlier diminishing returns
  - Higher: Later diminishing returns

- `thetas` [0, 0.8]: Controls adstock decay rate
  - Low (0-0.2): Fast decay (Search, Social)
  - Medium (0.1-0.4): Medium decay (Print, OOH)
  - High (0.3-0.8): Slow decay (TV)

### Budget Allocation Scenarios

The package supports multiple optimization scenarios:

1. **Maximum Response**
   - Optimizes for maximum return within spend constraints
   - Parameters:
     - `total_budget`: Total spend constraint
     - `channel_constr_low`: Minimum spend multiplier
     - `channel_constr_up`: Maximum spend multiplier 
     - `date_range`: Optimization period

2. **Target Efficiency**
   - Optimizes spend to hit specific ROAS/CPA targets
   
3. **Minimum Spend**
   - Finds minimum spend needed for target response
   
4. **Maximum Expected Response**  
   - Maximizes response within confidence intervals

## Credits 
Based on Meta's [Robyn](https://github.com/facebookexperimental/Robyn) R package.
