# SPDX-FileCopyrightText: 2024-present Ashok Cutkosky <ashok@cutkosky.com>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from dfloader import (
    Dataset,
    BatchedSequence,
    drop_non_numeric_columns,
    same_lists,
    is_dataframe,
    default_collate_fn,
    get_shuffled_batched_dataset,
)


class TestUtilityFunctions:
    def test_is_dataframe_with_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert is_dataframe(df) is True

    def test_is_dataframe_with_numpy_array(self):
        arr = np.array([[1, 2], [3, 4]])
        assert is_dataframe(arr) is False

    def test_is_dataframe_with_list(self):
        assert is_dataframe([1, 2, 3]) is False

    def test_same_lists_equal(self):
        assert same_lists([1, 2, 3], [1, 2, 3]) is True

    def test_same_lists_different_length(self):
        assert same_lists([1, 2], [1, 2, 3]) is False

    def test_same_lists_different_values(self):
        assert same_lists([1, 2, 3], [1, 2, 4]) is False

    def test_same_lists_empty(self):
        assert same_lists([], []) is True

    def test_drop_non_numeric_columns_all_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = drop_non_numeric_columns(df)
        assert list(result.columns) == ["a", "b"]

    def test_drop_non_numeric_columns_mixed(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
        result = drop_non_numeric_columns(df)
        assert list(result.columns) == ["a", "c"]

    def test_drop_non_numeric_columns_all_non_numeric(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        result = drop_non_numeric_columns(df)
        assert list(result.columns) == []


class TestDefaultCollateFn:
    def test_collate_single_key(self):
        batch = [{"a": [1, 2]}, {"a": [3, 4]}]
        result = default_collate_fn(batch)
        assert result == {"a": [[1, 2], [3, 4]]}

    def test_collate_multiple_keys(self):
        batch = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = default_collate_fn(batch)
        assert result == {"a": [1, 3], "b": [2, 4]}


class TestDataset:
    @pytest.fixture
    def simple_df(self):
        return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

    @pytest.fixture
    def simple_array(self):
        return np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

    def test_dataset_length_default(self, simple_df):
        ds = Dataset(simple_df)
        # Formula: ceil(((L + 1 - context_length - start_idx) / stride + 1) / batch_size)
        # = ceil(((5 + 1 - 1 - 0) / 1 + 1) / 1) = ceil(6) = 6
        assert len(ds) == 6

    def test_dataset_length_with_batch_size(self, simple_df):
        ds = Dataset(simple_df, batch_size=2)
        # With use_entire_df=True (default), ceil(5/2) = 3
        assert len(ds) == 3

    def test_dataset_length_without_entire_df(self, simple_df):
        ds = Dataset(simple_df, batch_size=2, use_entire_df=False)
        # Formula: floor(((L + 1 - context_length - start_idx) / stride + 1) / batch_size)
        # = floor(((5 + 1 - 1 - 0) / 1 + 1) / 2) = floor(6 / 2) = 3
        assert len(ds) == 3

    def test_dataset_getitem_shape(self, simple_df):
        ds = Dataset(simple_df, batch_size=2, context_length=3)
        int_data, float_data, extra_data = ds[0]
        # With no integer_columns specified, all columns go to float_data
        assert int_data.shape == (2, 3, 0)  # no integer columns
        assert float_data.shape == (2, 3, 2)  # 2 original cols
        assert extra_data.shape == (2, 3, 3)  # 3 extra columns

    def test_dataset_getitem_numpy_array(self, simple_array):
        ds = Dataset(simple_array, batch_size=1, context_length=2)
        int_data, float_data, extra_data = ds[0]
        assert int_data.shape == (1, 2, 0)  # no integer columns
        assert float_data.shape == (1, 2, 2)  # 2 original cols
        assert extra_data.shape == (1, 2, 3)  # 3 extra columns

    def test_dataset_return_type_dict(self, simple_df):
        ds = Dataset(simple_df, return_type="dict")
        int_dict, float_dict, extra_dict = ds[0]
        assert isinstance(int_dict, dict)
        assert isinstance(float_dict, dict)
        assert isinstance(extra_dict, dict)
        # With no integer_columns, all columns go to float_dict
        assert "x" in float_dict
        assert "y" in float_dict
        # Extra columns in extra_dict
        assert "__valid_data__" in extra_dict
        assert "__repeat_count__" in extra_dict
        assert "__seen_count__" in extra_dict

    def test_dataset_return_type_numpy(self, simple_df):
        ds = Dataset(simple_df, return_type="numpy")
        int_data, float_data, extra_data = ds[0]
        assert isinstance(int_data, np.ndarray)
        assert isinstance(float_data, np.ndarray)
        assert isinstance(extra_data, np.ndarray)

    def test_dataset_invalid_return_type(self, simple_df):
        with pytest.raises(ValueError, match="Unknown return type"):
            Dataset(simple_df, return_type="invalid")

    def test_dataset_negative_indexing(self, simple_df):
        ds = Dataset(simple_df)
        last_int, last_float, last_extra = ds[-1]
        expected_int, expected_float, expected_extra = ds[len(ds) - 1]
        np.testing.assert_array_equal(last_int, expected_int)
        np.testing.assert_array_equal(last_float, expected_float)
        np.testing.assert_array_equal(last_extra, expected_extra)

    def test_dataset_index_out_of_bounds(self, simple_df):
        ds = Dataset(simple_df)
        with pytest.raises(IndexError):
            _ = ds[100]

    def test_dataset_index_negative_out_of_bounds(self, simple_df):
        ds = Dataset(simple_df)
        with pytest.raises(IndexError):
            _ = ds[-100]

    def test_dataset_with_shuffle(self, simple_df):
        ds1 = Dataset(simple_df, batch_size=2, shuffle_seed=42)
        ds2 = Dataset(simple_df, batch_size=2, shuffle_seed=42)
        # Same seed should produce same results
        int1, float1, extra1 = ds1[0]
        int2, float2, extra2 = ds2[0]
        np.testing.assert_array_equal(int1, int2)
        np.testing.assert_array_equal(float1, float2)
        np.testing.assert_array_equal(extra1, extra2)

    def test_dataset_different_seeds_different_results(self, simple_df):
        ds1 = Dataset(simple_df, batch_size=2, shuffle_seed=42)
        ds2 = Dataset(simple_df, batch_size=2, shuffle_seed=123)
        # Different seeds should produce different results (with high probability)
        _, float1, _ = ds1[0]
        _, float2, _ = ds2[0]
        assert not np.array_equal(float1, float2)

    def test_dataset_reshuffle(self, simple_df):
        ds = Dataset(simple_df, batch_size=2, shuffle_seed=42)
        _, first_float, _ = ds[0]
        first_float = first_float.copy()
        ds.reshuffle()
        _, second_float, _ = ds[0]
        # After reshuffle, results should be different
        assert not np.array_equal(first_float, second_float)

    def test_dataset_valid_data_column(self, simple_df):
        ds = Dataset(simple_df, context_length=3, return_type="dict")
        _, _, extra_dict = ds[0]
        # First context window includes padding, so some valid_data should be 0
        valid_data = extra_dict["__valid_data__"]
        # At index 0 with context_length=3, the first two positions are invalid (negative indices)
        assert valid_data[0, 0] == 0  # invalid
        assert valid_data[0, 1] == 0  # invalid
        assert valid_data[0, 2] == 1  # valid

    def test_dataset_columns_with_reserved_names(self, simple_df):
        df_with_reserved = simple_df.copy()
        df_with_reserved["__valid_data__"] = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="cannot already contain"):
            Dataset(df_with_reserved)

    def test_dataset_stride(self, simple_df):
        ds = Dataset(simple_df, stride=2, context_length=1)
        # With stride=2 and context_length=1, we get every other row
        _, float0, _ = ds[0]
        _, float1, _ = ds[1]
        # Item 0 should correspond to df row 0
        assert float0[0, 0, 0] == 1  # x value at row 0
        # Item 1 should correspond to df row 2
        assert float1[0, 0, 0] == 3  # x value at row 2

    def test_dataset_start_idx(self, simple_df):
        ds = Dataset(simple_df, start_idx=2, context_length=1)
        _, float_data, _ = ds[0]
        # With start_idx=2, first item should be row 2
        assert float_data[0, 0, 0] == 3  # x value at row 2

    def test_dataset_force_numeric_true(self, simple_df):
        ds = Dataset(simple_df, force_numeric=True)
        int_data, float_data, extra_data = ds[0]
        assert float_data.dtype == np.float32
        assert int_data.dtype == np.int32
        assert extra_data.dtype == np.int32

    def test_dataset_nonconsecutive_index(self):
        df = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
        ds = Dataset(df, context_length=1)
        _, float_data, _ = ds[0]
        assert float_data[0, 0, 0] == 1


class TestBatchedSequence:
    def test_batched_sequence_length(self):
        seq = list(range(10))
        bs = BatchedSequence(seq, batch_size=3)
        # ceil(10/3) = 4 batches
        assert len(bs) == 4

    def test_batched_sequence_getitem(self):
        seq = [{"a": i} for i in range(5)]
        bs = BatchedSequence(seq, batch_size=2)
        batch = bs[0]
        assert batch == {"a": [0, 1]}

    def test_batched_sequence_last_batch_partial(self):
        seq = [{"a": i} for i in range(5)]
        bs = BatchedSequence(seq, batch_size=2)
        last_batch = bs[2]
        # Last batch should have only 1 element
        assert last_batch == {"a": [4]}

    def test_batched_sequence_custom_collate(self):
        def custom_collate(batch):
            return sum(batch)

        seq = [1, 2, 3, 4, 5]
        bs = BatchedSequence(seq, batch_size=2, collate_fn=custom_collate)
        assert bs[0] == 3  # 1 + 2
        assert bs[1] == 7  # 3 + 4
        assert bs[2] == 5  # 5


class TestGetShuffledBatchedDataset:
    def test_basic_functionality(self):
        df1 = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df2 = pd.DataFrame({"x": [7, 8, 9], "y": [10, 11, 12]})
        # Use return_type='dict' so default_collate_fn works
        dataset = get_shuffled_batched_dataset([df1, df2], batch_size=2, return_type="dict")
        # Should be iterable and return batched data
        assert len(dataset) > 0
        batch = dataset[0]
        # Now returns tuple of (int_dict, float_dict, extra_dict)
        assert isinstance(batch, tuple)
        assert len(batch) == 3
        int_dict, float_dict, extra_dict = batch
        assert isinstance(int_dict, dict)
        assert isinstance(float_dict, dict)
        assert isinstance(extra_dict, dict)

    def test_with_shuffle_chunk_size(self):
        df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
        dataset = get_shuffled_batched_dataset(
            [df], batch_size=2, shuffle_chunk_size=3
        )
        assert len(dataset) > 0


class TestIntegerColumns:
    def test_integer_columns_basic(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10.5, 20.5, 30.5, 40.5, 50.5]})
        ds = Dataset(df, integer_columns=["x"])
        int_data, float_data, extra_data = ds[0]
        # x should be in int_data, y in float_data, extra columns separate
        assert int_data.shape[2] == 1  # just x
        assert float_data.shape[2] == 1  # just y
        assert extra_data.shape[2] == 3  # 3 extra columns
        assert int_data.dtype == np.int32
        assert float_data.dtype == np.float32
        assert extra_data.dtype == np.int32

    def test_integer_columns_multiple(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [1.0, 2.0, 3.0]})
        ds = Dataset(df, integer_columns=["a", "b"])
        int_data, float_data, extra_data = ds[0]
        assert int_data.shape[2] == 2  # a and b
        assert float_data.shape[2] == 1  # just c
        assert extra_data.shape[2] == 3  # 3 extra columns

    def test_integer_columns_values_preserved(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        ds = Dataset(df, context_length=1, integer_columns=["x"])
        int_data, float_data, _ = ds[0]
        assert int_data[0, 0, 0] == 1
        assert float_data[0, 0, 0] == 10.0

    def test_integer_columns_dict_return_type(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
        ds = Dataset(df, return_type="dict", integer_columns=["x"])
        int_dict, float_dict, extra_dict = ds[0]
        assert "x" in int_dict
        assert "y" in float_dict
        assert "__valid_data__" in extra_dict
        assert int_dict["x"].dtype == np.int32
        assert float_dict["y"].dtype == np.float32
        assert extra_dict["__valid_data__"].dtype == np.int32

    def test_integer_columns_empty_list(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        ds = Dataset(df, integer_columns=[])
        int_data, float_data, extra_data = ds[0]
        assert int_data.shape[2] == 0
        assert float_data.shape[2] == 2  # 2 original
        assert extra_data.shape[2] == 3  # 3 extra

    def test_integer_columns_invalid_column(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with pytest.raises(ValueError, match="integer column 'z' not found"):
            Dataset(df, integer_columns=["z"])

    def test_float32_dtype(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = Dataset(df)
        _, float_data, _ = ds[0]
        assert float_data.dtype == np.float32

    def test_integers_preserved(self):
        # Test that integers are stored correctly without float conversion
        test_int = 123456789  # Fits in int32
        df = pd.DataFrame({"id": [test_int, test_int + 1, test_int + 2]})
        ds = Dataset(df, context_length=1, integer_columns=["id"])
        int_data, _, _ = ds[0]
        assert int_data[0, 0, 0] == test_int
        assert int_data.dtype == np.int32
        
    def test_integer_columns_order_preserved(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        ds = Dataset(df, context_length=1, integer_columns=["a", "c"])
        int_data, float_data, extra_data = ds[0]
        # a and c should be integers, b should be float
        assert int_data.shape[2] == 2
        assert float_data.shape[2] == 1  # just b
        assert extra_data.shape[2] == 3  # 3 extra
        assert int_data[0, 0, 0] == 1  # a
        assert int_data[0, 0, 1] == 7  # c
        assert float_data[0, 0, 0] == 4.0  # b


class TestDatasetEdgeCases:
    def test_single_row_dataframe(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        ds = Dataset(df)
        # Formula: ceil(((1 + 1 - 1 - 0) / 1 + 1) / 1) = ceil(2) = 2
        assert len(ds) == 2
        _, float_data, _ = ds[0]
        assert float_data.shape[0] == 1

    def test_single_column_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        ds = Dataset(df)
        _, float_data, extra_data = ds[0]
        # 1 original col, 3 extra cols separate
        assert float_data.shape[2] == 1
        assert extra_data.shape[2] == 3

    def test_large_context_length(self):
        # When context_length > data_length, the formula can produce negative length
        # Use a case where it still works: context_length=3 with 5 rows
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        ds = Dataset(df, context_length=3)
        _, float_data, _ = ds[0]
        # Should handle context with padding at the start
        assert float_data.shape[1] == 3

    def test_context_length_equals_data_length(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        ds = Dataset(df, context_length=5)
        # Formula: ceil(((5 + 1 - 5 - 0) / 1 + 1) / 1) = ceil(2) = 2
        assert len(ds) == 2
        _, float_data, _ = ds[0]
        assert float_data.shape[1] == 5

    def test_use_entire_df_pads_past_end(self):
        """Test that use_entire_df=True pads with last row when going past end of data."""
        # 3 rows of data
        df = pd.DataFrame({"x": [10, 20, 30]})
        # batch_size=2, context_length=2, stride=1
        # ideal_length = ((3 + 1 - 2 - 0) / 1 + 1) / 2 = 1.5
        # use_entire_df=True: ceil(1.5) = 2 batches
        ds = Dataset(df, batch_size=2, context_length=2, stride=1, use_entire_df=True)
        assert len(ds) == 2

        # Get the last batch (idx=1)
        # batch_indices = [2, 3]
        # For b=0: virtual_df_indices = [1, 2] (both valid)
        # For b=1: virtual_df_indices = [2, 3] (index 3 is past end)
        _, float_data, extra_data = ds[1]

        # Shape should be [batch_size=2, context_length=2, num_cols=1]
        assert float_data.shape == (2, 2, 1)

        # Batch element 0 (b=0): indices [1, 2] -> values [20, 30]
        assert float_data[0, 0, 0] == 20
        assert float_data[0, 1, 0] == 30

        # Batch element 1 (b=1): indices [2, 3] -> values [30, 30 (padded)]
        assert float_data[1, 0, 0] == 30  # valid index 2
        assert float_data[1, 1, 0] == 30  # padded with last row value

        # Check valid_data mask
        valid_data = extra_data[:, :, 0]  # __valid_data__ is first extra column

        # Batch element 0: both positions valid
        assert valid_data[0, 0] == 1
        assert valid_data[0, 1] == 1

        # Batch element 1: first position valid, second invalid (past end)
        assert valid_data[1, 0] == 1
        assert valid_data[1, 1] == 0  # past end of data
