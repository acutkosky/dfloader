# SPDX-FileCopyrightText: 2024-present Ashok Cutkosky <ashok@cutkosky.com>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from typing import Sequence, Optional, Union
import dask.dataframe as dd
import collections
import loadit


def drop_non_numeric_columns(df):
    # mostly copied from chatgpt :)
    def is_numeric_series(series):
        # Attempt to convert series to numeric, non-convertible entries will be NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # If all original NaNs are still NaN and all non-NaNs are converted to numeric, then it's a numeric column
        return numeric_series.notna().equals(series.notna())

    # Apply this function to each column and filter out non-numeric columns
    numeric_columns = [col for col in df.columns if is_numeric_series(df[col])]
    return df[numeric_columns]

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
        drop_non_numeric: bool = False,
        drop_na: bool = True,
    ):
        self.using_dask = isinstance(df, dd.DataFrame)

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
            start_idx = self.context_length - 1
        self.start_idx = start_idx
        self.drop_non_numeric = drop_non_numeric
        self.drop_na = drop_na

        self.allow_incomplete_context = allow_incomplete_context


        self.length = int((len(self.df) - 1 - self.start_idx) / self.stride) + 1
        if self.allow_incomplete_context:
            leftover_rows = (len(self.df) - 1 - self.start_idx) % self.stride
            if leftover_rows != 0:
                self.length += 1


    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError

        # df.loc appears to be inclusive of the start and end index...
        virtual_df_end = idx * self.stride + self.start_idx
        virtual_df_start = virtual_df_end - self.context_length + 1

        logical_df_end = min(virtual_df_end, len(self.df) - 1)
        logical_df_start = max(virtual_df_start, 0)

        if self.allow_incomplete_context:
            end_padding = virtual_df_end - logical_df_end
            start_padding = logical_df_start - virtual_df_start
        else:
            end_padding = 0
            start_padding = 0


        results = self.df.loc[logical_df_start:logical_df_end]
        if self.using_dask:
            results = results.compute()
        results = results.copy()
        results["__contextmask__"] = results.notna().all(axis=1)

        if end_padding > 0:
            end_rows = pd.concat([results.iloc[[-1]]] * end_padding)
            results = pd.concat([results, end_rows], ignore_index=True)
            results.iloc[
                -end_padding:, results.columns.get_loc("__contextmask__")
            ] = False
        # print(results)
        if start_padding > 0:
            start_rows = pd.concat([results.iloc[[0]]] * start_padding)
            results = pd.concat([start_rows, results], ignore_index=True)
            results.iloc[
                :start_padding, results.columns.get_loc("__contextmask__")
            ] = False

        results["__labelmask__"] = results["__contextmask__"]
        label_start = max(0, len(results["__labelmask__"]) - self.label_length)
        results.iloc[:label_start, results.columns.get_loc("__labelmask__")] = False

        if self.drop_non_numeric:
            results = drop_non_numeric_columns(results)
            # results = results.apply(
            #     pd.to_numeric, errors="coerce"
            # )  # This will convert non-numerics to NaN in numeric columns
            # results = results.dropna()
        if self.drop_na:
            results = results.dropna()

        results = results.to_dict(orient="list")

        return results


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
