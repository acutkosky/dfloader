# SPDX-FileCopyrightText: 2024-present Ashok Cutkosky <ashok@cutkosky.com>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from typing import Sequence, Optional, Union, List
import dask.dataframe as dd
import collections
import loadit


def drop_non_numeric_columns(df: Union[pd.DataFrame, dd.DataFrame]):
    # mostly copied from chatgpt :)
    def is_numeric_series(series):
        # Attempt to convert series to numeric, non-convertible entries will be NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # If all original NaNs are still NaN and all non-NaNs are converted to numeric, then it's a numeric column
        return numeric_series.notna().equals(series.notna())

    # Apply this function to each column and filter out non-numeric columns
    numeric_columns = [col for col in df.columns if is_numeric_series(df[col])]
    if len(numeric_columns) == len(df.columns):
        return df
    return df[numeric_columns]

def same_lists(l1, l2):
    if len(l1)!=len(l2):
        return False
    for a,b in zip(l1,l2):
        if a!=b:
            return False
    return True

def is_dataframe(df):
    return isinstance(df, pd.DataFrame) or isinstance(df, dd.DataFrame)

class Dataset(collections.abc.Sequence):
    def __init__(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        batch_size: int = 1,
        context_length: int = 1,
        stride: int = 1,
        start_idx: int = 0,
        use_entire_df: bool = True,
        shuffle_seed: Optional[int] = None,
        skip_index_check: bool = False,
        return_type: str = 'numpy',
        columns: List[str] = [],
    ):


        if return_type not  in ['numpy', 'dict']:
            raise ValueError(f"Unknown return type: {return_type}")
        self.return_type = return_type

        self.columns = columns + ['__mask__']
        if is_dataframe(df):
            if columns != []:
                if not same_lists(columns, list(df.columns)):
                    raise ValueError("provided columns argument is not the same as df.columns")
            self.columns  = list(df.columns) + ['__mask__']

        if isinstance(df, np.ndarray) and return_type == 'dict':
            if  len(columns) != df.shape[-1]:
                raise ValueError("number of column names does not match number of data columns")


        if is_dataframe(df):
            self.mask_dtype = bool
        else:
            self.mask_dtype = df.dtype

        self.df = df

        self.context_length = context_length
        self.batch_size = batch_size
        self.stride = stride
        self.start_idx = start_idx
        self.use_entire_df = use_entire_df

        # we should think of the input df as an array of shape [L, C].
        # Each output of this loader will have shape:
        # [B, C + K, T] where B is batch_size, T is context_length and K represents
        # some extra  columns that will be described shortly.
        # That is, this loader can be viewed as an [N, B, C + K, T] dimensional array

        
        # To gain intuition about what this loader outputs, we do some calculations
        # regarding loader[:, :C, T] (that is, ignoring the extra K columns for a moment)
        
        # First, assuming start_idx = 0 and B=1, we have:
        # loader[n, 0, c, t] = df[n*stride+t - T +1 , c]
        # Where we fill df[-x, c] = df[0, x] for x>0
        # That is, loader[n, 0, c, :] is the T consecutive elements of df[:, c] ending
        # at index n*stride.
        # When start_idx = S, we instead set:
        # loader[n, 0, c, t] = df[S + n*stride+t - T +1 , c]

        # In general, when B>1:
        # loader[n, b, c, t] = df[S + (n*B + b)*stride + t - T + 1, c]
        # so the maximum value is:
        # loader[N-1, B-1, C-1, T-1] = df[S + (N*B -1)*stride + T-1, C-1]

        # This maximum value might fall outside the maximum indices of df,
        # the flag use_entire_df specifies  how to deal with this.
        # if use_entire_df is False, then we restrict N to the maximum value:
        # N = floor(  ((L + 1  - T  -S) / stride + 1) / B )

        # Alternatively, if use_entire_df is True, then we allow N to be:
        # N = ceil(  ((L + 1  - T  -S) / stride + 1) / B )
        # in this case, we set df[i,j] = df[L-1, j] when i >= L.


        # Now, let us discuss the extra K columns.
        # Currently  K=1 (maybe it will be bigger in later version...)
        
        # The first extra column is the "valid_data" mask column.
        # We set
        # loader[n, b, C, t] = 1 if [n,b, :, t] corresponds to valid indices in df
        # (these indices are df[S + (n*B + b)*stride + t - T + 1, :] ).
        # Otherwise, loader[n ,b, C, t] = 0

        
        if isinstance(df, np.ndarray):
            L, C = df.shape
        else:
            L = len(df)
            C = len(df.columns)

        ideal_length = ((L +  1 - self.context_length - self.start_idx) / self.stride + 1)/self.batch_size

        if self.use_entire_df:
            self.length = int(np.ceil(ideal_length))
        else:
            self.length = int(np.floor(ideal_length))
            
        self.shuffle_seed = shuffle_seed
        if self.shuffle_seed is not None:
            self.rng  = np.random.default_rng(self.shuffle_seed)
            self.reshuffle()

        self.df_has_nonconsecutive_index = False
        if isinstance(self.df, pd.DataFrame) or isinstance(self.df, dd.DataFrame) and not skip_index_check:
            self.df_has_nonconsecutive_index = np.any(np.arange(len(self.df)) != np.array(self.df.index))
            
    def __len__(self):
        return self.length

    def reshuffle(self):
        self.shuffled_indices = self.rng.permutation(self.length * self.batch_size)

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError

       
        # This next line is unfortunately a bit "clever".
        # Index [b, c, t] of the output corresponds to df[virtual_df_start + b*stride + t, c]
        # We'd like to  make a [B, T] shape array A where A[b,t] = virtual_df_start + b * stride + t
        # So, we make two range arrays: a "context_indices" arrau that is (0, ..., context_length - 1)
        # and a "batch_indices" array that is (virtual_df_start, virtual_df_start + stride, ..., virtual_df_start + stride*(batch_size -1)).
        # Then we reshape these to  [1, context_length] and [batch_size, 1] and add. Numpy broadcasting will produce
        # the  correct array of  shape [batch_size, context_length]
        # Unfortunately, this is even more involved  when we are shuffling the data.
        # In this case, the  batch_indices are need to be passed through  a  permutation of the original indices.
        #
        context_indices  = np.arange(0, self.context_length)
        batch_indices = np.arange(idx*self.batch_size, idx*self.batch_size + self.batch_size)
        if self.shuffle_seed is not None:
            # print("batch indices: ",batch_indices, "suffled indices: ",self.shuffled_indices)
            batch_indices = self.shuffled_indices[batch_indices]
        virtual_df_indices = self.start_idx  - self.context_length + 1 + (context_indices.reshape(1,-1) + self.stride * batch_indices.reshape(-1,1))

        # mask_column[b, t] = 0 whenever start_idx + (idx * batch_size + b -1) * stride + t - context_length + 1
        # is not in >= 0 and < len(df).
        mask_column = (virtual_df_indices >= 0) * (virtual_df_indices < len(self.df))
        mask_column = mask_column.astype(self.mask_dtype).reshape((self.batch_size, 1, self.context_length))


        logical_df_indices = np.clip(virtual_df_indices, a_min=0, a_max=len(self.df)-1)

        if self.df_has_nonconsecutive_index:
            logical_df_indices = np.array(self.df.index)[logical_df_indices]



        if isinstance(self.df, np.ndarray):
            data  = self.df[logical_df_indices]
        elif isinstance(self.df, pd.DataFrame):
            data = self.df.loc[logical_df_indices.flatten()]
            data = data.to_numpy().reshape(list(logical_df_indices.shape)+[-1])

        elif isinstance(self.df, dd.DataFrame):
            data = self.df.loc[logical_df_indices.flatten()]
            data = data.compute().to_numpy().reshape(list(logical_df_indices.shape)+[-1])

        data = data.transpose((0,2,1))
            

        data = np.concatenate((data, mask_column), axis=1)

        if self.return_type == 'numpy':
            return data
        else:
            return {self.columns[k]: data[:,k,:] for k in range(len(self.columns))}
            


# a bit like pytorch's default collate, but doesn't yell at you
# if you don't have tensors.
def default_collate_fn(batch: Sequence):
    """
    batch is a list of examples
    each example is a dictionary whose values are list-like objects
    """

    keys = batch[0].keys()
    return {key: [example[key] for example in batch] for key in keys}


class BatchedSequence(collections.abc.Sequence):
    def __init__(
        self, seq: Sequence, batch_size: int, collate_fn: callable = default_collate_fn
    ):
        self.batch_size = batch_size
        self.seq = seq
        self.collate_fn = collate_fn

        self.length = (len(self.seq) + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if idx > len(self):
            raise IndexError

        start = idx * self.batch_size
        end = min(len(self.seq), start + self.batch_size)

        return self.collate_fn([self.seq[i] for i in range(start, end)])


def get_shuffled_batched_dataset(
    dfs: Sequence[Union[pd.DataFrame, dd.DataFrame]],
    *ds_args,
    batch_size: int,
    shuffle_chunk_size: int = 0,
    **ds_kwargs,
):
    datasets = [Dataset(df=df, *ds_args, **ds_kwargs) for df in dfs]

    dataset = loadit.util.ConcatableSequence(*datasets)

    if shuffle_chunk_size > 0:
        dataset = loadit.util.chunk_shuffle(dataset, chunk_size=shuffle_chunk_size)

    dataset = BatchedSequence(dataset, batch_size=batch_size)

    return dataset
