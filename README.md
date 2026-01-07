# dfloader

[![PyPI - Version](https://img.shields.io/pypi/v/dfloader.svg)](https://pypi.org/project/dfloader)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dfloader.svg)](https://pypi.org/project/dfloader)

A data loader for pandas DataFrames that creates sliding window batches—ideal for time-series data and sequence modeling.

## Installation

```console
pip install dfloader
```

## Quick Start

```python
import pandas as pd
from dfloader import Dataset

# Create sample time-series data
df = pd.DataFrame({
    'price': [100.0, 101.5, 99.2, 102.3, 103.1],
    'volume': [1000, 1200, 800, 1500, 1100]
})

# Create a dataset with context windows of length 3
dataset = Dataset(df, batch_size=2, context_length=3)

# Iterate through batches
for batch in dataset:
    int_data, float_data, extra_data = batch
    print(float_data.shape)  # (batch_size, context_length, num_float_columns)
```

## Core Concepts

### The Dataset Class

`Dataset` wraps a DataFrame and produces batches of consecutive rows (context windows). Each output has shape `(batch_size, context_length, num_columns)`.

```python
Dataset(
    df,                          # pandas DataFrame or numpy array
    batch_size=1,                # Number of sequences per batch
    context_length=1,            # Length of each sequence window
    stride=1,                    # Step size between consecutive windows
    start_idx=0,                 # Starting index in the DataFrame
    use_entire_df=True,          # Include partial batches at the end
    shuffle_seed=None,           # Seed for reproducible shuffling
    return_type='numpy',         # 'numpy' or 'dict'
    columns=None,                # Column names (auto-detected for DataFrames)
    force_numeric=True,          # Convert to numeric types
    integer_columns=None,        # Columns to keep as int32
)
```

### How Windowing Works

Each window is a slice of `context_length` consecutive rows. The `start_idx` parameter controls where the first window ends.

**Default behavior** (`start_idx=0`, `context_length=3`, `stride=1`):

The first window ends at row 0, so it needs padding for earlier positions:

```python
# DataFrame rows: [0, 1, 2, 3, 4]
```

- Window 0: `[pad, pad, 0]` — padded with row 0's values
- Window 1: `[pad, 0, 1]`
- Window 2: `[0, 1, 2]`
- Window 3: `[1, 2, 3]`
- Window 4: `[2, 3, 4]`

**With `start_idx=2`:**

The first window ends at row 2, so no initial padding is needed:

```python
# DataFrame rows: [0, 1, 2, 3, 4], context_length=3, start_idx=2
```

- Window 0: `[0, 1, 2]`
- Window 1: `[1, 2, 3]`
- Window 2: `[2, 3, 4]`

**With `stride=2`:**

Windows skip by 2 positions instead of 1:

```python
# DataFrame rows: [0, 1, 2, 3, 4], context_length=3, stride=2, start_idx=2
```

- Window 0: `[0, 1, 2]`
- Window 1: `[2, 3, 4]`

**With `use_entire_df`:**

Controls whether to include windows that extend past the end of the DataFrame.

`use_entire_df=True` (default): Includes all windows, padding with the last row's values when needed:

```python
# DataFrame rows: [0, 1, 2, 3, 4], context_length=3, stride=2, start_idx=2
```

- Window 0: `[0, 1, 2]`
- Window 1: `[2, 3, 4]`
- Window 2: `[4, pad, pad]` — padded with row 4's values

`use_entire_df=False`: Only includes windows that fit entirely within the DataFrame:

```python
# Same settings but use_entire_df=False
```

- Window 0: `[0, 1, 2]`
- Window 1: `[2, 3, 4]`
- *(Window 2 excluded because it would require padding)*

Use `__valid_data__` in extra columns to identify which positions are real vs padded data.

### Return Types

**NumPy mode** (`return_type='numpy'`, default):

Returns a tuple of three arrays: `(int_data, float_data, extra_data)`

```python
dataset = Dataset(df, batch_size=4, context_length=10, return_type='numpy')
int_data, float_data, extra_data = dataset[0]
# int_data.shape: (4, 10, num_integer_cols)
# float_data.shape: (4, 10, num_float_cols)
# extra_data.shape: (4, 10, 3)  # the 3 extra columns
```

**Dict mode** (`return_type='dict'`):

Returns a tuple of three dictionaries mapping column names to arrays:

```python
dataset = Dataset(df, batch_size=4, context_length=10, return_type='dict')
int_dict, float_dict, extra_dict = dataset[0]
# Access by column name: float_dict['price'].shape is (4, 10)
```

### Integer vs Float Columns

By default, all columns are treated as float32. Use `integer_columns` to preserve integer precision:

```python
dataset = Dataset(
    df,
    integer_columns=['user_id', 'category_id'],  # Keep as int32
    # All other columns become float32
)
```

## Extra Columns

Each batch includes three automatically-generated metadata columns:

| Column | Description |
|--------|-------------|
| `__valid_data__` | 1 if the row corresponds to valid data in the original DataFrame, 0 otherwise (for padding at boundaries) |
| `__repeat_count__` | How many times this row appears across all windows (useful when stride < context_length) |
| `__seen_count__` | Which occurrence of this row this is (1 = first time, up to repeat_count) |

### Example Values

Consider a DataFrame with rows `[0, 1, 2, 3, 4]`, `context_length=3`, `stride=1`, `start_idx=0`:

```
Window 0: data=[pad, pad, 0]  valid=[0, 0, 1]  repeat=[0, 0, 3]  seen=[0, 0, 1]
Window 1: data=[pad, 0, 1]    valid=[0, 1, 1]  repeat=[0, 3, 3]  seen=[0, 2, 1]
Window 2: data=[0, 1, 2]      valid=[1, 1, 1]  repeat=[3, 3, 3]  seen=[3, 2, 1]
Window 3: data=[1, 2, 3]      valid=[1, 1, 1]  repeat=[3, 3, 3]  seen=[3, 2, 1]
Window 4: data=[2, 3, 4]      valid=[1, 1, 1]  repeat=[3, 3, 3]  seen=[3, 2, 1]
Window 5: data=[3, 4, pad]    valid=[1, 1, 0]  repeat=[3, 3, 0]  seen=[3, 2, 0]
Window 6: data=[4, pad, pad]  valid=[1, 0, 0]  repeat=[3, 0, 0]  seen=[3, 0, 0]
```

**`__valid_data__`**: 0 for padded positions, 1 for real data. Use this to mask loss calculations.

**`__repeat_count__`**: Each row appears in 3 windows (e.g., row 2 appears in windows 2, 3, and 4), so repeat count is 3. Padded positions have repeat count 0.

**`__seen_count__`**: Counts how many times this row has appeared *so far* when iterating through windows in order. In window 1, position 1 contains row 0—this is the 2nd time row 0 has appeared (it first appeared in window 0), so seen=2. The rightmost position (t=2) always has seen=1 since that row is appearing for the first time.

### Example with batch_size=2

With `batch_size=2`, each `dataset[idx]` returns 2 windows. The seen_count tracks across the entire iteration, so the same row can appear multiple times *within a single batch*:

```python
# DataFrame rows: [0, 1, 2, 3, 4, 5], context_length=3, stride=1, start_idx=2, batch_size=2
```

```
dataset[0] returns:
  Window 0: data=[0, 1, 2]  seen=[1, 1, 1]   <- row 2 seen 1st time
  Window 1: data=[1, 2, 3]  seen=[2, 2, 1]   <- row 2 seen 2nd time (same batch!)

dataset[1] returns:
  Window 2: data=[2, 3, 4]  seen=[3, 2, 1]   <- row 2 seen 3rd time
  Window 3: data=[3, 4, 5]  seen=[3, 2, 1]
```

Notice that **row 2 appears twice in `dataset[0]`**: at position t=2 in window 0 (seen=1) and at position t=1 in window 1 (seen=2). This is important for avoiding double-counting when computing losses or metrics:

```python
int_data, float_data, extra_data = dataset[idx]
valid_mask = extra_data[:, :, 0]      # __valid_data__
repeat_counts = extra_data[:, :, 1]   # __repeat_count__
seen_counts = extra_data[:, :, 2]     # __seen_count__

# Only compute loss on first occurrence of each row
first_occurrence_mask = (seen_counts == 1)

# Or weight by inverse repeat count to avoid overcounting
weights = 1.0 / repeat_counts.clip(min=1)
```

## Shuffling

Enable shuffling with a seed for reproducibility:

```python
dataset = Dataset(df, batch_size=32, shuffle_seed=42)

# Reshuffle between epochs
dataset.reshuffle()
```

## Working with Multiple DataFrames

Use `get_shuffled_batched_dataset` to combine multiple DataFrames:

```python
from dfloader import get_shuffled_batched_dataset

# Multiple time series (e.g., different stocks, sensors, etc.)
dfs = [df1, df2, df3]

dataset = get_shuffled_batched_dataset(
    dfs,
    batch_size=32,
    context_length=50,
    shuffle_chunk_size=1000,  # Shuffle in chunks for better mixing
)
```

## Full Example: Training Loop

```python
import pandas as pd
from dfloader import Dataset

# Load your data
df = pd.read_csv('timeseries.csv')

# Create dataset
dataset = Dataset(
    df,
    batch_size=64,
    context_length=100,
    stride=10,
    shuffle_seed=42,
    integer_columns=['label'],
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx in range(len(dataset)):
        int_data, float_data, extra_data = dataset[batch_idx]
        
        # float_data: your features
        # int_data: integer columns like labels
        # extra_data[:,:,0]: mask for valid data points
        
        # ... your training code ...
    
    dataset.reshuffle()  # Shuffle for next epoch
```

## License

`dfloader` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
