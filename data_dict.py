# Data class
import json

from functools import reduce
import numpy as np
import pandas as pd
import datetime
import warnings

# for smoothing of values by date
from astropy.convolution import convolve, Gaussian2DKernel

def to_array(*args, date_format="%Y-%m-%d"):
    """
    generator to convert arguments to np.ndarray
    """

    for x in args:
        if isinstance(x, datetime.date):
            yield np.array([x.strftime(date_format)], dtype="datetime64[D]")
        # if already an array yield as is
        if isinstance(x, np.ndarray):
            yield x
        elif isinstance(x, (list, tuple)):
            yield np.array(x)
        # elif isinstance(x, (pd.Series, pd.core.indexes.base.Index, pd.core.series.Series)):
        #     yield x.values
        elif isinstance(x, (int, float, str, bool, np.bool_)):
            yield np.array([x], dtype=type(x))
        # np.int{#}
        elif isinstance(x, (np.int8, np.int16, np.int32, np.int64)):
            yield np.array([x], dtype=type(x))
        # np.float{#}
        elif isinstance(x, (np.float16, np.float32, np.float64)):
            yield np.array([x], dtype=type(x))
        # np.bool*
        elif isinstance(x, (np.bool, np.bool_, np.bool8)):
            yield np.array([x], dtype=type(x))
        # np.datetime64
        elif isinstance(x, np.datetime64):
            yield np.array([x], "datetime64[D]")
        elif x is None:
            yield np.array([])
        else:
            from warnings import warn
            warn(f"Data type {type(x)} is not configured in to_array.")
            yield np.array([x], dtype=object)


def match(x, y):
    """match elements in x to their location in y (taking first occurrence)"""
    # require x,y to be arrays
    x, y = to_array(x, y)
    # NOTE: this can require large amounts of memory if x and y are big
    mask = x[:, None] == y
    row_mask = mask.any(axis=1)
    assert row_mask.all(), \
        f"{(~row_mask).sum()} not found, uniquely : {np.unique(np.array(x)[~row_mask])}"
    return np.argmax(mask, axis=1)


class DataDict(dict):

    def __init__(self, vals, dims=None, name=None, is_flat=False):
        assert isinstance(vals, np.ndarray), f"vals expected to be np.ndarray, got {type(vals)}"
        assert isinstance(dims, (dict, type(None))), f"dims expected to be np.ndarray, got {type(dims)}"

        # if dims not provided, default to numbered idx values
        if dims is None:
            dims = {f"idx{i}": np.arange(s) for i, s in enumerate(vals.shape)}

        # make all the dims values be np.arrays
        for k in dims.keys():
            dims[k], = to_array(dims[k])

        self.vals = vals
        self.dims = dims
        self.name = name
        self.flat = is_flat

        # check inputs
        val_dims = vals.shape

        # if not flat (n-d array) check de
        if not self.flat:
            assert len(val_dims) == len(dims), f"dimensions did not match: " \
                                               f"len(val_dims)={len(val_dims)} " \
                                               f"len(dims)={len(dims)}"
            dim_keys = list(dims.keys())
            for i, vd in enumerate(val_dims):
                assert vd == len(dims[dim_keys[i]]), f"dim num: {i} with dim name: {dim_keys[i]} " \
                                                     f"did not match, got: \n" \
                                                     f"{vd} vs {len(dims[dim_keys[i]])}"

            # require dimension values are unique
            for k, v in dims.items():
                _, c = np.unique(v, return_counts=True)
                assert np.array_equal(np.unique(c), np.array([1])), \
                    f"dim name: {k} had duplicate entries for\n{_[c != 1]}"

        # otherwise is flat, so dims should contain arrays of equal length
        else:
            dim_keys = list(dims.keys())
            for i, vd in enumerate(val_dims):
                assert len(vals) == len(dims[dim_keys[i]]), \
                    f"flat={self.self}, expect dims to contain " \
                    f"arrays same length as vals\n" \
                    f"dim name: {dims[dim_keys[i]]} had length: " \
                    f"{len(dims[dim_keys[i]])}, expected: {len(vals)}"

    # def __str__(self):
    #     out = f"DataDict object:\n" + \
    #           f"name: {self.name}\n" + \
    #           f"vals.shape: {self.vals.shape}\n" + \
    #           f"vals.dtype: {self.vals.dtype}\n" + \
    #           f"dims names: {list(self.dims.keys())}\n" +\
    #           f"dims lens: {[len(v) for k,v in self.dims.items()]}\n" + \
    #           f"is flat: {self.flat}\n" + \
    #           f"DataDict: key, values:\n" + super().__repr__()
    #     return out

    def __repr__(self):

        out = f"DataDict object:\n" + \
              f"name: {self.name}\n" + \
              f"vals.shape: {self.vals.shape}\n" + \
              f"vals.dtype: {self.vals.dtype}\n" + \
              f"dims names: {list(self.dims.keys())}\n" + \
              f"dims lens: {[len(v) for k, v in self.dims.items()]}\n" + \
              f"is flat: {self.flat}\n" + \
              f"key, values:\n" + super().__repr__()

        # return "DataDict: key, values:\n" + super().__repr__()
        return out

    @staticmethod
    def _reshape_and_move(vals, dims, new_dims):
        """reshape axis and add """
        new_keys = list(new_dims.keys())
        cur_keys = list(dims.keys())

        assert np.in1d(cur_keys, new_keys).all(), f"all current dim keys must be in (new) dim keys"

        # destination for current keys in new_dims
        dst_keys = match(cur_keys, new_keys)

        # temp shape to move to - add 1 for each key not in new_keys
        temp_shape = list(vals.shape) + [1 for k in new_keys if k not in cur_keys]

        # move to temp shape (by adding dimension if needed)
        # add move axis
        return np.moveaxis(vals.reshape(temp_shape),
                           np.arange(len(cur_keys)),
                           dst_keys)

    def _union_dims(self, other):
        # output dims will match
        new_dims = self.dims.copy()
        # check or add dims
        for k, v in other.dims.items():
            if k in new_dims:
                # TODO: this could be too strict, all them to match up to some sorting?
                #  - or take an intersection / union of the two
                assert np.array_equal(v, new_dims[k]), f"dim: {k} had mismatching dim values"
            if k not in new_dims:
                new_dims[k] = v
        return new_dims

    def __add__(self, other):

        if isinstance(other, DataDict):
            new_dims = self._union_dims(other)

            vals = self._reshape_and_move(self.vals, self.dims, new_dims) + \
                   self._reshape_and_move(other.vals, other.dims, new_dims)

        else:
            new_dims = self.dims
            vals = self.vals + other

        # TODO: consider applying a name here?
        return DataDict(vals=vals, dims=new_dims)

    def __sub__(self, other):

        if isinstance(other, DataDict):
            new_dims = self._union_dims(other)

            vals = self._reshape_and_move(self.vals, self.dims, new_dims) - \
                   self._reshape_and_move(other.vals, other.dims, new_dims)
        else:
            new_dims = self.dims
            vals = self.vals - other

        return DataDict(vals=vals, dims=new_dims)

    def __mul__(self, other):
        if isinstance(other, DataDict):
            new_dims = self._union_dims(other)

            vals = self._reshape_and_move(self.vals, self.dims, new_dims) * \
                   self._reshape_and_move(other.vals, other.dims, new_dims)
        else:
            new_dims = self.dims
            vals = self.vals * other

        return DataDict(vals=vals, dims=new_dims)

    def __truediv__(self, other):
        if isinstance(other, DataDict):
            new_dims = self._union_dims(other)

            vals = self._reshape_and_move(self.vals, self.dims, new_dims) / \
                   self._reshape_and_move(other.vals, other.dims, new_dims)
        else:
            new_dims = self.dims
            vals = self.vals / other

        return DataDict(vals=vals, dims=new_dims)


    def _pretty_print_dims(self):
        # TODO: want to be able to trim output
        # print(json.dumps({k: v.tolist() for k, v in self.dims.items()}, indent=4))
        for k, v in self.dims.items():
            print(f"'{k}': {v}")

    def set_dim_idx(self, dim_idx, new_idx=None, dim_vals=None, inplace=True):

        new_idx = dim_idx if new_idx is None else new_idx
        assert dim_idx in self.dims, f"dim_idx: {dim_idx} is not in {self.dims.keys()}"
        assert len(self.dims[dim_idx]) == len(dim_vals), \
            f"len(self.dims[dim_idx]): {len(self.dims[dim_idx])} " \
            f"!= len(dim_vals) :  {len(dim_vals)}"

        dims = self.dims.copy()
        dims = {new_idx if k == dim_idx else k:
                    dim_vals if k == dim_idx else v
                for k, v in dims.items()}
        # dims.pop(dim_idx)
        # dims[new_idx] = dim_vals
        if inplace:
            self.dims = dims
        else:
            return DataDict(vals=self.vals, dims=dims, name=self.name, is_flat=self.flat)

    def fill_value(self, fill, select_array=None, select_dims=None):
        if select_array is not None:
            assert isinstance(select_array, np.ndarray), \
                f"select_array should be an array, got: {type(select_array)}"

            assert self.vals.shape == select_array.shape, \
                f"shape mismatch: {self.vals.shape} vs {select_array.shape}"
            assert select_array.dtype == bool, f"select_array dtype expected to be bool, got: {select_array.dtype}"

            self.vals[select_array] = fill
        elif select_dims is not None:
            idx, _ = self._index_loc_from_select_dims(select_dims)
            self.vals[idx] = fill
        else:
            print("select_array and select_dims both None, doing nothing")

    def _index_loc_from_select_dims(self, select_dims):
        """get object to select values from vals, as well as location dic"""
        locs = {}
        for k in self.dims.keys():
            # if key is not in select take entire dimension
            if k not in select_dims:
                locs[k] = np.arange(len(self.dims[k]))
            else:
                if self.flat:
                    locs[k] = np.in1d(self.dims[k], select_dims[k])
                else:
                    # TODO: is match needed here?
                    locs[k] = match(select_dims[k], self.dims[k])
        # if flat, just return bool array
        if self.flat:
            return reduce(lambda x, y: x & y, [v for k, v in locs.items()]).astype(bool), locs
        else:
            return np.ix_(*[locs[k] for k in locs.keys()]), locs

    def copy(self, new_name=None):
        """make a copy of object"""
        new_name = new_name if new_name is not None else self.name
        return DataDict(vals=self.vals, dims=self.dims, name=new_name, is_flat=self.flat)

    @staticmethod
    def dims_equal(dims1, dims2):
        # return dims1 == dims2

        equal_dims = True
        # TODO: could the below be done in a cleaner way?
        if not isinstance(dims1, dict):
            print(f"dims1 should be dict, got {type(dims1)}")
            equal_dims = False

        if not isinstance(dims2, dict):
            print(f"dims2 should be dict, got {type(dims1)}")
            equal_dims = False

        if not len(dims1) == len(dims2):
            print(f"dimsensions did not match")
            equal_dims = False

        if not set(dims1) == set(dims2):
            print(f"set(dims1) != set(dims2)")
            equal_dims = False

        for k, v in dims1.items():
            if not np.array_equal(v, dims2[k]):
                print(f"dims1[{k}] not equal to dims2[{k}]")
                equal_dims = False

        # assert isinstance(dims1, dict), f"dims1 shold be dict, got {type(dims1)}"
        # assert isinstance(dims2, dict), f"dims2 shold be dict, got {type(dims1)}"
        # assert len(dims1) == len(dims2), f"dimsensions did not match"
        # assert set(dims1) == set(dims2), f"set(dims1) != set(dims2)"
        #
        # for k, v in dims1.items():
        #     assert np.array_equal(v, dims2[k]), f"dims1[{k}] not equal to dims2[{k}]"
        #

        return equal_dims

    def equal(self, other, verbose=False):
        """compare DataDict object to see if equal"""
        # TODO: here should check class
        # if not d.name == self.name:
        if not self.flat == other.flat:
            if verbose:
                print(f"flat values: {self.flat} vs {other.flat}")
            return False

        # if not self.dims == d.dims:
        if not self.dims_equal(self.dims, other.dims):
            if verbose:
                print("dims not equal")
            return False

        # TODO: allow for almost equal?
        if not np.array_equal(self.vals, other.vals):
            if verbose:
                print("values not equal")
            return False

        return True

    def subset(self,
               select_dims=None,
               select_array=None,
               inplace=False,
               # strict=True,
               new_name=None,
               verbose=False):
        """select a subset of data"""
        # TODO: allow for non strict matching of dimension values

        new_name = new_name if new_name is not None else self.name

        if select_dims is not None:
            assert isinstance(select_dims, dict), f"select_dims must be dict, got: {type(select_dims)}"

            for k in select_dims.keys():
                assert k in self.dims, f"select contained key: {k} which is not in dims: {self.dims.keys()}"
                # make select dims values an array
                select_dims[k], = to_array(select_dims[k])

            # idx will allow for selecting values in an array
            # locs is a dict of dimensions being selected
            idx, locs = self._index_loc_from_select_dims(select_dims)

            # get subset of vals and dims
            if self.flat:
                vals = self.vals[idx]
                dims = {k: v[idx] for k, v in self.dims.items()}
            else:
                # select subset of vals based on locations
                vals = self.vals[idx]
                dims = {k: v[locs[k]] for k, v in self.dims.items()}

            if inplace:
                self.vals = vals
                self.dims = dims
            else:
                return DataDict(vals=vals, dims=dims, is_flat=self.flat, name=new_name)

        elif select_array is not None:
            assert isinstance(select_array, np.ndarray), \
                f"select_array should be ndarray, got: {type(select_array)}"
            assert select_array.shape == self.vals.shape, \
                f"select_array shape mismatch: {select_array.shape} != {self.vals.shape}"

            assert str(select_array.dtype) == 'bool', \
                f"select_array.dtype != 'bool', got: '{str(select_array.dtype)}'"

            true_loc = np.nonzero(select_array)
            vals = self.vals[true_loc]
            dims = self.dims.copy()
            for kidx, k in enumerate(dims.keys()):
                if self.flat:
                    dims[k] = dims[k][true_loc[0]]
                else:
                    dims[k] = dims[k][true_loc[kidx]]

            if inplace:
                # TODO: here store original dims
                self.vals = vals
                self.dims = dims
                self.flat = True
            else:
                return DataDict(vals=vals, dims=dims, is_flat=True, name=new_name)
        else:
            print(f"neither select_array or select_dims was provided, doing nothing")

    def flatten(self, inplace=False):
        """flatten data, if it's not already"""
        # flatten data if it's not already
        if not self.flat:
            # NOTE: this approach requires creating a bool array equal to the size of vals
            bool_array = np.ones(self.vals.shape, dtype=bool)
            # flatten the values
            vals = self.vals.flatten()
            # use bool array to get location of values in each dimension
            true_loc = np.nonzero(bool_array)
            dims = self.dims.copy()
            for kidx, k in enumerate(dims.keys()):
                dims[k] = dims[k][true_loc[kidx]]

            if inplace:
                # original dims
                self._org_dims = self.dims.copy()
                self.dims = dims
                self.vals = vals
                self.flat = True
            else:
                _ = DataDict(vals=vals, dims=dims, name=self.name, is_flat=True)
                # TODO: change how this attribute is set
                _._org_dims = self.dims.copy()
                return _
        # otherwise already flat
        else:
            print("already flat")
            # do nothing (return None) if inplace
            if inplace:
                return None
            # otherwise make a copy
            else:
                return self.copy()

    def unflatten(self, inplace=False, fill_val=np.nan, udims=None, verbose=False):

        if self.flat:
            # for each dimension get the unique values
            if udims is None:
                udims = {k: np.unique(v) for k, v in self.dims.items()}
                # if has a _org_dims attribute, re-order udims values
                if hasattr(self, "_org_dims"):
                    udims = {k:  v[np.argsort(match(v, self._org_dims[k]))]
                             for k, v in udims.items()}
            # if unique dims has been provided use those
            else:
                assert isinstance(udims, dict), f"udims provided by needs to be dict"
                for k in self.dims.keys():
                    assert k in udims, f"key: {k} in dims is not in provided udims"
                if verbose:
                    print("using provided unique dims (udims), with sizes:")
                    udims_shape = {k: len(v) for k,v in udims.items()}
                    print(udims_shape)
            # get the shape from num. of unique values
            shape = [len(v) for k, v in udims.items()]
            # create an nd-array to populate
            # NOTE: the fill_val and dtype must be compatible, e.g. np.nan and float
            vals = np.full(shape, fill_val, dtype=self.vals.dtype)
            # get the locations of the dim values in unique positions
            locs = [match(v, udims[k]) for k, v in self.dims.items()]
            # fill nd-array at correct locations
            vals[tuple(locs)] = self.vals
            if inplace:
                self.dims = udims
                self.vals = vals
                self.flat = False
            else:
                return DataDict(vals=vals, dims=udims, name=self.name, is_flat=False)


        # otherwise already flat
        else:
            print("already not flat")
            # do nothing (return None) if inplace
            if inplace:
                return None
            # otherwise make a copy
            else:
                return self.copy()

    def movedims(self, src, dst):
        """wrapper for move axis, use dimension values instead"""
        src, dst = to_array(src, dst)
        assert len(src) == len(dst)

        source = match(src, list(self.dims.keys()))
        self.moveaxis(source, dst)

    def moveaxis(self, source, destination):
        source, destination = to_array(source, destination)
        order = [n for n in range(self.vals.ndim) if n not in source]

        for dest, src in sorted(zip(destination, source)):
            order.insert(dest, src)
        new_key_ord = np.array(list(self.dims.keys()))[order]
        self.vals = np.moveaxis(self.vals, source, destination)
        self.dims = {k: self.dims[k] for k in new_key_ord}

    @classmethod
    def dims_intersection(cls, *dims):
        """get the intersection of the dimension valaues (for each dimension)
        return dims dict"""
        for d in dims:
            assert isinstance(d, dict), f"expect each element to be a dict, for one got: {type(d)}"
        if len(dims) == 1:
            print("only set of dims provided")
            return dims[0]
        # check all keys are the same
        for i in range(1, len(dims)):
            assert set(dims[i - 1]) == set(dims[i])

        int_dims = {k: None for k in dims[0].keys()}
        for k in int_dims.keys():
            int_dims[k] = reduce(lambda x, y: np.intersect1d(x, y),
                                 [dim[k] for dim in dims])

        return int_dims

    @classmethod
    def concatenate(cls, *obs, dim_name="new_dim", name="", verbose=True):

        # check all the dims match
        if len(obs) == 1:
            if verbose:
                print(f"only one object provided, will effectively just add dimension: {dim_name}")
            # return obs[0]

        # else:
        # check objects are all correct class
        for i, o in enumerate(obs):
            assert isinstance(o, cls), f"object: {i} was wrong class, got: {type(o)}"

        # check the dims match
        for i in range(1, len(obs)):
            assert cls.dims_equal(obs[i - 1].dims, obs[i].dims), f"obs[{i - 1}].dims != obs[{i}].dims"
        # require each object is not flat
        for ob in obs:
            assert not ob.flat
        # require all the names are unique
        dim_idx = np.array([ob.name for ob in obs])
        assert len(dim_idx) == len(np.unique(dim_idx))

        # add dimension vals and concatentate together
        vals = np.concatenate([ob.vals[..., None] for ob in obs], axis=-1)

        dims = obs[0].dims.copy()
        dims[dim_name] = dim_idx

        return DataDict(vals=vals, dims=dims, is_flat=False, name=name)

    def not_nan(self, inplace=False):
        return self.subset(select_array=~np.isnan(self.vals), inplace=inplace)

    def to_dataframe(self):
        """convert to pandas DataFrame"""

        if self.flat:
            midx = pd.MultiIndex.from_arrays([v for v in self.dims.values()],
                                           names=[k for k in self.dims.keys()])
            df = pd.DataFrame(self.vals, index=midx, columns=[self.name])
        else:
            midx = pd.MultiIndex.from_product([v for v in self.dims.values()],
                                              names=[k for k in self.dims.keys()])
            df = pd.DataFrame(self.vals.flatten(), index=midx, columns=[self.name])

        return df

    def from_dataframe(self):
        """make DataDict from a DataFrame"""
        warnings.warn("from_dataframe not implemented")

    def clip_smooth_by_date(self, nan_mask=None, vmin=None, vmax=None, std=1):
        """clip smooth a DataDict object that has date in dims
        - expect remaining dim to be 2d
        """
        # assert isinstance(obj, DataDict), f"obj needs to be DataDict, got: {type(obj)}"

        assert "date" in self.dims, f"date not in dims"
        assert not self.flat, "must not be flat"
        assert len(self.vals.shape) == 3, "can only handle 3-d data"

        if nan_mask is not None:
            assert nan_mask.shape == tuple([len(v) for k, v in self.dims.items() if k != "date"]), \
                f"nan_mask.shape = {nan_mask.shape}, not in align with obj shape"

        out = []
        for date in self.dims['date']:

            # the following was mostly just copied from smooth()
            # select data
            data_smth = self.subset(select_dims={"date": date}).vals.copy()
            data_smth = np.squeeze(data_smth)
            if data_smth.dtype != float:
                data_smth = data_smth.astype('float')

            # apply convolution on squeezed values
            data_smth[np.isinf(data_smth)] = np.nan
            if vmax is not None:
                data_smth[data_smth >= vmax] = vmax
            if vmin is not None:
                data_smth[data_smth <= vmin] = vmin
            # TODO: review this -
            data_smth = convolve(data_smth,
                                 Gaussian2DKernel(x_stddev=std, y_stddev=std))
            # TODO: this is in included for legacy reasons, requires review
            #  - i.e. when / where would it get populated with zeros?
            data_smth[data_smth == 0] = np.nanmean(data_smth)
            if nan_mask is not None:
                data_smth[nan_mask] = np.nan

            new_dims = {k: v for k, v in self.dims.items() if k != 'date'}
            out.append(DataDict(vals=data_smth, dims=new_dims, is_flat=False, name=date))

        # TODO: clip_smooth_each_date add smooth to name? keep name
        out = DataDict.concatenate(*out, dim_name="date", name=self.name)
        out.movedims('date', 2)
        # include key,values (?)
        for k, v in self.items():
            out[k] = v

        return out

    @staticmethod
    def full(shape=None, dims=None, fill_val=None, name=None, dtype=None):
        """create an array 'full' of fill_value"""

        if shape is None:
            assert dims is not None, f"either shape or dims must be provided to full"
            shape = [len(v) for v in dims.values()]
        assert fill_val is not None, "please pick a different fill_val than None"
        vals = np.full(shape, fill_val)

        if dtype is not None:
            vals = vals.astype(dtype)

        return DataDict(vals=vals, dims=dims, name=name, is_flat=len(shape)==1)


if __name__ == "__main__":


    # --
    # input parameters
    # --

    # data dimensions
    dims = {
        "x": np.arange(10),
        "y": np.arange(3, 9),
        "t": ["2020", "2021", "1900"]
    }
    # data shape
    shape = [len(v) for k, v in dims.items()]
    # data values
    vals = np.arange(np.prod(shape)).reshape(*shape)

    # ---
    # Create 'DataDict' object
    # ---

    d = DataDict(vals=vals, dims=dims, name="data")

    # --
    # full of fill_vals
    # --

    df = DataDict.full(shape=(2, 3, 4), fill_val=9)

    # --
    # copy object
    # --

    d2 = d.copy(new_name="new_data")

    print(f"copy data object is equal: {d.equal(d2, verbose=False)}")


    # --
    # move axis (and dimensions)
    # --

    dm = d.copy()
    dm.moveaxis(0, 2)
    assert np.array_equal(dm.vals, np.moveaxis(d.vals, 0, 2))

    # move dimensions
    dm = d.copy()
    dm.movedims('x', 2)

    # ---
    # flatten data
    # ---

    dflat = d.flatten(inplace=False)

    print(f"flatten data is equal: {d.equal(dflat, verbose=True)}")

    # unflatten
    duflat = dflat.unflatten()

    # unflattening a previously flattened DataDict should give the same result
    assert duflat.equal(d)

    # ---
    # convert to pd.DataFrame
    # ---

    df0 = d.to_dataframe()
    df1 = dflat.to_dataframe()

    assert df0.equals(df1), f"DataFrames from flat and not flat object expected to be equal"

    # ---
    # subset
    # ---

    # select subset with select_dims, vals stay as ndarray
    dsub = d.subset(select_dims={"x": [1, 2, 3]})

    # select with a bool mask on values - get flat
    select_array = d.vals > np.mean(d.vals)
    dsub2 = d.subset(select_array=select_array, inplace=False, new_name="subset")

    # select from flat data using select_dims
    _ = dsub2.subset(select_dims={"y": [4, 5]})
    # select from flat data using select_array
    _ = dsub2.subset(select_array=(dsub2.vals > np.mean(dsub2.vals)))

    # ---
    # combine
    # ----

    names = ["a", "b", "c"]
    _ = [d.copy(new_name=n) for n in names]

    dcon = d.concatenate(*_, name="combined")

    # ---
    # TODO: undo flat
    # ---

    # ---
    # TODO: add a to_pandas methods
    # ---

    # ---
    # add,
    # TODO: include - sub, mult, div
    # ___

    # add
    d3 = d2 + d2
    assert np.allclose(d3.vals, d2.vals * 2), f"adding DataDict did not work as expected"

    # add with scalar
    _ = DataDict.full(dims=d2.dims, fill_val=5.) + 3
    _ = np.unique(_.vals)
    assert (len(_) == 1) & (_ == 8.0)


    # when dims equal addition operation is symetric
    assert (d2 + d).equal(d + d2) & DataDict.dims_equal(d2.dims, d.dims)

    # add with more dimensions
    _ = {"g": np.arange(-3, 4), **d.dims}
    shape = [len(v) for v in _.values()]
    _ = DataDict(vals=np.arange(np.prod(shape)).reshape(shape),
                 dims=_)
    d4 = d + _

    # NOTE: this is not working as expected
    assert np.allclose(d4.vals,
                       np.moveaxis(_.vals, 0, -1) + d.vals[..., None]), f"adding none overlapping dim names didn't ork"

    # NOTE: order of addition matters when there is a mismatch in dimensions, e.g.: x+y != y+x
    d4a = _ + d
    assert not d4.equal(d4a)

    # add with fewer dimensions
    _ = DataDict(vals=np.arange(len(d.dims['x'])), dims={'x': d.dims['x']})
    d5 = d + _
    assert np.allclose(d5.vals, _.vals[:, None, None] + d.vals), f"adding none overlapping dim names didn't ork"
