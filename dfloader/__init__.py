# SPDX-FileCopyrightText: 2024-present Ashok Cutkosky <ashok@cutkosky.com>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from typing import Sequence, Optional, Union
import dask.dataframe as dd
import collections
import loadit


class Dataset(collections.abc.Sequence):
    def __init__(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        context_length: int = 1,
        stride: Optional[int] = 1,
        label_length: Optional[int] = None,
        start_idx: Optional[int] = 0,
        allow_incomplete_context: bool = False,
        columns: Optional[Sequence[str]] = None,
        drop_non_numeric: bool=True
    ):

        self.using_dask = isinstance(df, dd.DataFrame)

        if self.using_dask:
            self.df_module = dd
        else:
            self.df_module = pd

        if columns is None:
            columns = df.columns
        self.df = df[columns]
        self.context_length = context_length
        self.columns = columns
        self.stride = stride
        if label_length is None:
            label_length = stride

        self.label_length = label_length
        if start_idx is None:
            start_idx = self.context_length-1
        self.start_idx = start_idx
        self.drop_non_numeric = drop_non_numeric

        self.allow_incomplete_context = allow_incomplete_context

        # self.original_length = len(self.df)
        # self.front_padding = 0

        self.length = int((len(self.df)-1 - self.start_idx) / self.stride)  + 1
        # append extra rows to the df if needed:
        if self.allow_incomplete_context:
            leftover_rows = (len(self.df)-1 - self.start_idx) % self.stride
            if leftover_rows != 0:
                self.length += 1
                
            #     num_extra_rows = self.stride - leftover_rows
            #     print("num extra rows: ",num_extra_rows)

            #     extra_rows = self.df.loc[[len(self.df)-1]] * num_extra_rows
            #     #     pd.DataFrame(
            #     #     np.nan, index=range(num_extra_rows), columns=self.df.columns
            #     # )
            #     # if self.using_dask:
            #     #     extra_rows = dd.from_pandas(extra_rows, npartitions=self.df.npartitions)

            #     # print('self.df: ',self.df)
            #     # print("extra row: ",extra_rows)
            #     self.df = self.df_module.concat([self.df, extra_rows], ignore_index=True)

            # front_padding = self.context_length - self.start_idx - 1

            # if front_padding > 0:
            #     front_rows = self.df.loc[[0]] * front_padding
            #     print("front padding: ",front_padding)
            #     self.df = self.df_module.concat([front_rows, self.df], ignore_index=True)

            #     self.start_idx = self.context_length
            #     self.front_padding = front_padding

        # print("now self.df: ",self.df)

        # self.length = int((len(self.df)-1 - self.start_idx) / self.stride)  + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError

        # df.loc appears to be inclusive of the start and end index...
        virtual_df_end = idx * self.stride + self.start_idx
        virtual_df_start = virtual_df_end - self.context_length + 1

        logical_df_end = min(virtual_df_end, len(self.df)-1)
        logical_df_start = max(virtual_df_start, 0)

        if self.allow_incomplete_context:
            end_padding = virtual_df_end - logical_df_end
            start_padding = logical_df_start - virtual_df_start
        else:
            end_padding = 0
            start_padding = 0

        # print("end padding: ",end_padding)
        # print("start_padding: ",start_padding)
        
        
        # max(df_end - self.context_length + 1,0)

        # assert df_start < self.original_length

        # end_padded_indices = max(0, df_end - (self.original_length - 1))
        # front_padded_indices = max(df_start, self.front_padding) - df_start
        # print("start: ",df_start)
        # print('end: ',df_end)
        # print('padded_indices: ', end_padded_indices)
        # print('front padding indices: ',front_padded_indices)

        results = self.df.loc[logical_df_start : logical_df_end]
        if self.using_dask:
            results = results.compute()
        results = results.copy()
        results['__contextmask__'] = results.notna().all(axis=1)
        # results['__labelmask__'] = results['__contextmask__']

        if end_padding > 0:
            end_rows = pd.concat([results.iloc[[-1]]] * end_padding)
            results = pd.concat([results, end_rows], ignore_index=True)
            results.iloc[-end_padding:, results.columns.get_loc('__contextmask__')] = False
        # print(results)
        if start_padding > 0:
            start_rows = pd.concat([results.iloc[[0]]] * start_padding)
            results = pd.concat([start_rows, results], ignore_index=True)
            results.iloc[:start_padding, results.columns.get_loc('__contextmask__')] = False

        results['__labelmask__'] = results['__contextmask__']
        label_start = max(0, len(results['__labelmask__'])-self.label_length)
        results.iloc[:label_start, results.columns.get_loc('__labelmask__')] = False
        

        if self.drop_non_numeric:
            results = results.apply(pd.to_numeric, errors='coerce')  # This will convert non-numerics to NaN in numeric columns
            results = results.dropna()

        results = results.to_dict(orient="list")


        return results


# like pytorch's default collate, but doesn't yell at you
# if you don't have tensors.
def default_collate_fn(batch: Sequence):
    '''
    batch is a list of examples
    each example is a dictionary whose values are list-like objects
    '''

    # print("about to tree map batch: ")
    # print(batch)
    keys = batch[0].keys()
    return {
        key: [example[key] for example in batch] for key in keys
    }



class BatchedSequence(collections.abc.Sequence):

    def __init__(self, seq: Sequence, batch_size: int, collate_fn: callable = default_collate_fn):

        self.batch_size = batch_size
        self.seq = seq
        self.collate_fn = collate_fn

        self.length = (len(self.seq) + self.batch_size - 1)//self.batch_size


    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int):

        if idx > len(self):
            raise IndexError

        start = idx *  self.batch_size
        end = min(len(self.seq), start + self.batch_size)


        return self.collate_fn(
            [self.seq[i] for i in range(start, end)]
        )



def get_shuffled_batched_dataset(
    dfs:Sequence[Union[pd.DataFrame, dd.DataFrame]],
    *ds_args,
    batch_size: int,
    shuffle_chunk_size:int =0,
    **ds_kwargs,
    ):

    datasets = [
        Dataset(
            df=df,
            *ds_args,
            **ds_kwargs
        ) for df in dfs
    ]

    dataset = loadit.util.ConcatableSequence(*datasets)

    if shuffle_chunk_size > 0:
        dataset = chunk_shuffle(dataset, chunk_size=shuffle_chunk_size)

    dataset = BatchedSequence(dataset, batch_size=batch_size)

    return dataset


def chunk_shuffle_idx(chunk_size: int, length: int, seed: Optional = None):
    num_chunks = length // chunk_size

    rng = np.random.default_rng(seed)
    last_chunk_size = length - num_chunks * chunk_size
    if last_chunk_size > 0:
        last_chunk = [num_chunks * chunk_size  + rng.permutation(last_chunk_size)]
    else:
        last_chunk = []

    permutations = np.concatenate(
        [i * chunk_size + rng.permutation(chunk_size) for i in range(num_chunks)] + last_chunk
    )
    return permutations


def chunk_shuffle(
    seq: Sequence,
    chunk_size: Optional[int],
    length: Optional[int] = None,
    seed: Optional = None,
):
    if length is None:
        length = len(seq)

    if chunk_size is None:
        chunk_size = len(seq)

    shuffle_idx = chunk_shuffle_idx(chunk_size, length, seed)

    if isinstance(seq, loadit.util.SequenceView):
        return seq[shuffle_idx]
    else:
        return loadit.util.SequenceView(seq, shuffle_idx)


