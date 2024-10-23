# SPDX-FileCopyrightText: 2024-present Ashok Cutkosky <ashok@cutkosky.com>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from typing import Sequence, Optional, Union, List
import collections
import loadit

try:
    import dask.dataframe as dd
    dask_available = True
    df_type = Union[pd.DataFrame, dd.DataFrame]
except ImportError:
    dask_available = False
    df_type = pd.DataFrame

def drop_non_numeric_columns(df: df_type):
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
    return isinstance(df, df_type)

class Dataset(collections.abc.Sequence):
    def __init__(
        self,
        df: df_type,
        batch_size: int = 1,
        context_length: int = 1,
        stride: int = 1,
        start_idx: int = 0,
        use_entire_df: bool = True,
        shuffle_seed: Optional[int] = None,
        skip_index_check: bool = False,
        return_type: str = 'numpy',
        columns: Optional[List[str]] = None,
    ):


        if return_type not  in ['numpy', 'dict']:
            raise ValueError(f"Unknown return type: {return_type}")
        self.return_type = return_type

        #  validate columns argument/generate defaults when needed.
        extra_columns = ['__valid_data__', '__repeat_count__', '__seen_count__']

        if is_dataframe(df):
            if columns is not None:
                if not same_lists(columns, list(df.columns)):
                    raise ValueError("provided columns argument is not the same as df.columns")
            columns  = list(df.columns)
        else:
            if columns is None:
                columns = list(range(df.shape[1]))

        for c in extra_columns:
            if c in columns:
                raise ValueError(f"original columns cannot already contain {c}")

        self.columns = columns + extra_columns

        if isinstance(df, np.ndarray) and return_type == 'dict':
            if  len(columns) != df.shape[-1]:
                raise ValueError("number of column names does not match number of data columns")


        if is_dataframe(df):
            self.valid_data_dtype = bool
        else:
            self.valid_data_dtype = df.dtype

        self.df = df

        self.context_length = context_length
        self.batch_size = batch_size
        self.stride = stride
        self.start_idx = start_idx
        self.use_entire_df = use_entire_df

        # we should think of the input df as an array of shape [L, C].
        # Each output of this loader will have shape:
        # [B, T, C + K] where B is batch_size, T is context_length and K represents
        # some extra  columns that will be described shortly.
        # That is, this loader can be viewed as an [N, B, T, C + K,] dimensional array

        
        # To gain intuition about what this loader outputs, we do some calculations
        # regarding loader[:, :, :C] (that is, ignoring the extra K columns for a moment)
        
        # First, assuming start_idx = 0 and B=1, we have:
        # loader[n, 0, t, c] = df[n*stride+t - T +1 , c]
        # Where we fill df[-x, c] = df[0, x] for x>0
        # That is, loader[n, 0, :, c] is the T consecutive elements of df[:, c] ending
        # at index n*stride.
        # When start_idx = S, we instead set:
        # loader[n, 0, t, c] = df[S + n*stride+t - T +1 , c]

        # In general, when B>1:
        # loader[n, b, t, c] = df[S + (n*B + b)*stride + t - T + 1, c]
        # so the maximum value is:
        # loader[N-1, B-1, T-1, C-1] = df[S + (N*B -1)*stride + T-1, C-1]

        # This maximum value might fall outside the maximum indices of df,
        # the flag use_entire_df specifies  how to deal with this.
        # if use_entire_df is False, then we restrict N to the maximum value:
        # N = floor(  ((L + 1  - T  -S) / stride + 1) / B )

        # Alternatively, if use_entire_df is True, then we allow N to be:
        # N = ceil(  ((L + 1  - T  -S) / stride + 1) / B )
        # in this case, we set df[i,j] = df[L-1, j] when i >= L.


        # Now, let us discuss the extra K columns.
        # Currently  K=3 (maybe it will be bigger in later version...)


        ### EXTRA COLUMN 1: __valid_data__ ###
        # The first extra column is the "valid_data" mask column.
        # We set
        # loader[n, b, t, C] = 1 if [n,b, t, :] corresponds to valid indices in df
        # (these indices are df[S + (n*B + b)*stride + t - T + 1, :] ).
        # Otherwise, loader[n ,b, t, C] = 0
        # That is, this  column just checks if S + (n*B + b)*stride + t - T +1 is
        # (1) non-negative, and (2) less than len(df)


        ### EXTRA COLUMN 2: __repeat_count__ ###
        # The second extra column is the "repeat_count" column. Since the stride
        # may not equal the context length, some rows may repeat. We will generate
        # a new column that tells how many times each row repeats. We set the  value to zero
        # when the data is out-of-range in the original df (i.e.  when __valid_data__ is False).
        # Let's do the calculution carefully.
        # Recall that loader[n, b, t, :] corresponds to
        # df[S +(n*B +b)*stride +  t -T + 1, :].
        # Let us write 
        # F := S + (n*B + b)*stride  + t - T +1
        #
        # Therefore loader[n, b, t, C+1] should just count the number of possible
        # values of n', b', t'  such that
        # F = S + (n' * B + b') *stride  + t' - T + 1
        #
        # To do this count, first observe that for a given n',b', there is at most
        # one possible value for t'. 
        # Further, the values of (n'*B + b') that result in "valid" data from df 
        # (that is, for which S + (n' * B + b') *stride + t' - T +1 \in [0, L-1] for some t')
        # take on all integers from max(0, -floor(S/stride) ) to  floor((L - S + T - 1)/stride) exactly once each,
        # where we recall that df has shape [L, C]
        
        # Next for a given value of p = n'*B+b', in order for there to be a possible value for t', we must have:
        # S + p*stride - T + 1 <= F
        # AND
        # S +  p *  stride >= F
        # Equivalently:
        # p * stride \in [F - S, F - S + T - 1]
        # So, p must satisfy:
        # p \in [ceil( (F-S)/stride ), floor( (F -S + T  -1)/stride ) ]
        #     = [n*B+b + ceil((t-T+1)/stride), n*B+b + floor(t/stride)]
        # Combining both restrictions on p:
        # p \in [max(n*B+b + ceil((t-T+1)/stride), 0, -floor(S/stride)),  min(n*B+b+floor(t/stride), floor((L-S+T-1)/stride))]
        # So finally, the number  of  possible p values (and therefore the number of duplicates) is:
        # min(n*B+b + floor(t/stride), floor((L-S+T-1)/stride)) - max(n*B+b + ceil((t-T+1)/stride), 0, -floor(S/stride)) + 1


        ### EXTRA COLUMN 3: __seen_count__ ###
        # The third extra column counts how many times this row has appeared so far when iterating through the
        # data in order.
        # That is, if we iterate through  loader[n, b, t, :] by first cycling through b from 0 to B-1 and then incrementing
        # n (which will go through the data in the same order as iterating through df), then
        # loader[n, b, t, C+2] will indicate the number of times this row has been produced before (counting the current iteration).
        # So, loader[n, b, t, C+2] will have minimum value 1 and maximum value equal to loader[n, b, C+1,  t].
        # We set the  value to zero when the data is out-of-range in the original df (i.e.  when __valid_data__ is False).

        # This can be used to identify specific occurances of each row. For example, we might wish to compute a loss only on labels for
        # for which  this is the first time we will see the label.

        # To calculate this value, we can reuse much of the algebra from computing extra column 2 (__repeat_count__):
        # the minimum value of n'*B+b' that will generate the current row is:
        # p_min = max(n*B+b + ceil((t-T+1)/stride), 0, -floor(S/stride))
        # All values of  n'*B+b' that are between the current value and p_min must have appeared exactly once each previously, so:
        # loader[n, b, t, C+2] = n*B+b - p_min + 1

        
        

        
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

        self.set_shuffle_seed(shuffle_seed)

        self.df_has_nonconsecutive_index = False
        if is_dataframe(self.df) and not skip_index_check:
            self.df_has_nonconsecutive_index = np.any(np.arange(len(self.df)) != np.array(self.df.index))
            
    def __len__(self):
        return self.length

    def set_shuffle_seed(self, shuffle_seed):
        self.shuffle_seed = shuffle_seed
        if self.shuffle_seed is not None:
            self.rng  = np.random.default_rng(self.shuffle_seed)
            self.reshuffle()
        

    def reshuffle(self):
        self.shuffled_indices = self.rng.permutation(self.length * self.batch_size)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        
        if idx >= len(self) or idx < 0:
            raise IndexError
        

       
        # This next line is unfortunately a bit "clever".
        # Index [b, t, c] of the output corresponds to df[virtual_df_start + b*stride + t, c]
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
            batch_indices = self.shuffled_indices[batch_indices]

        context_indices = context_indices.reshape(1, -1)
        batch_indices = batch_indices.reshape(-1, 1)
        virtual_df_indices = self.start_idx  - self.context_length + 1 + (context_indices + self.stride * batch_indices)


        ### Computing extra columns ###

        ### __valid__data__ ###
        # valid_data[b, t] = 0 whenever start_idx + (idx * batch_size + b -1) * stride + t - context_length + 1
        # is not in >= 0 and < len(df).
        valid_data = (virtual_df_indices >= 0) * (virtual_df_indices < len(self.df))
        valid_data = valid_data.astype(self.valid_data_dtype).reshape((self.batch_size, self.context_length, 1))

        ### __repeat_count__ ###
        p_max = np.minimum(batch_indices + np.floor(context_indices/self.stride), np.floor((len(self.df)-self.start_idx + self.context_length - 1)/self.stride))

        p_min = np.maximum(np.maximum(batch_indices + np.ceil((context_indices-self.context_length+1)/self.stride), -np.floor(self.start_idx/self.stride)), 0.0)

        repeat_count = (p_max - p_min + 1).astype(int)
        repeat_count = repeat_count.reshape((self.batch_size, self.context_length, 1)) * valid_data

        ### __seen_count__ ###
        seen_count = (batch_indices - p_min + 1).astype(int)
        seen_count = seen_count.reshape((self.batch_size, self.context_length, 1)) * valid_data


        


        logical_df_indices = np.clip(virtual_df_indices, a_min=0, a_max=len(self.df)-1)

        if self.df_has_nonconsecutive_index:
            logical_df_indices = np.array(self.df.index)[logical_df_indices]



        if isinstance(self.df, np.ndarray):
            data  = self.df[logical_df_indices]
        elif isinstance(self.df, pd.DataFrame):
            data = self.df.loc[logical_df_indices.flatten()]
            data = data.to_numpy().reshape((self.batch_size, self.context_length, -1)) #list(logical_df_indices.shape)+[-1])

        elif dask_available and isinstance(self.df, dd.DataFrame):
            data = self.df.loc[logical_df_indices.flatten()]
            data = data.compute().to_numpy().reshape((self.batch_size, self.context_length, -1)) #list(logical_df_indices.shape)+[-1])

        # data = data.transpose((0,2,1))
            

        data = np.concatenate((data, valid_data, repeat_count, seen_count), axis=2)

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
    dfs: Sequence[df_type],
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
