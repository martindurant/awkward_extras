import zarr
import pickle
import base64
import pyarrow as pa
import awkward1 as ak

arr = ak.fromjson("/Users/mdurant/Downloads/bikeroutes.json")
parr = ak.to_arrow(arr)


def pa_to_zarr(arr, path):
    z = zarr.open_group(path, mode='w')
    z.attrs['none'] = [b is None for b in arr.buffers()]
    z.attrs['length'] = len(arr)
    for i, buf in enumerate(arr.buffers()):
        if buf is None:
            continue
        z.empty(name=i, dtype='uint8', shape=(len(buf), ))
        z[:] = buf
    z.attrs['type'] = base64.b64encode(pickle.dumps(arr.type)).decode()


def zarr_to_pa(path):
    z = zarr.open_group(path, mode='r')
    buffers = [None if n else pa.py_buffer(z[i][:])
               for i, n in enumerate(z.attrs['none'])]
    typ = pickle.loads(base64.b64decode(z.attrs['type'].encode()))
    return pa.Array.from_buffers(typ, length=z.attrs['length'], buffers=buffers)


def ak_to_zarr(arr, path):
    z = zarr.open_group(path, mode='w')
    schema, arr_dict, parts = ak.to_arrayset(arr)
    z.attrs['schema'] = schema.tojson()
    for key, data in arr_dict.items():
        z.create_dataset(key, dtype=data.dtype, data=data, shape=data.shape)


def ak_from_zarr(path):
    z = zarr.open_group(path, mode='r')
    form = ak.forms.Form.fromjson(z.attrs['schema'])
    arr = ak.from_arrayset(form, {k: v[:] for k, v in z.items()})
    return arr
