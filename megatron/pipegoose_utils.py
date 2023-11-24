import random
import socket
from functools import partial
from typing import Callable
from megatron.logger import Logger
import struct
import numpy as np
import os
import random
import torch

import torch.multiprocessing as mp


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                Logger()("Found free port: {}".format(port))
                return port
        except OSError as e:
            raise e


def spawn(func: Callable, world_size: int = 1, **kwargs):
    if kwargs.get("port") is None:
        port = find_free_port()
    else:
        port = kwargs["port"]
        kwargs.pop("port")

    wrapped_func = partial(func, world_size=world_size, port=port, **kwargs)

    mp.spawn(wrapped_func, nprocs=world_size)

import struct
import numpy as np
from functools import reduce

def write_bin(filename, array):
    # Force endianess: https://stackoverflow.com/questions/23831422/what-endianness-does-python-use-to-write-into-files
    dtype_to_format = {
        np.int8: 'i',
        np.int16: 'i',
        np.int32: 'i',
        np.int64: 'i',
        np.unsignedinteger: 'I',
        np.float16: 'f',
        np.float32: 'f',
        np.float64: 'f',
        np.double: 'd'
    }
    fmt = dtype_to_format[array.dtype.type]
    shapes = [shape for shape in array.shape]
    if len(shapes) == 0:
        shapes = [1]
    # n, c, h, w = array.shape
    with open(filename, "wb") as f:
        # number of dim
        f.write(struct.pack('I', len(shapes)))
        for shape in shapes:
            f.write(struct.pack('I', shape))
        f.write(struct.pack('c', bytes(fmt, 'utf-8')))
        f.write(struct.pack(f"{fmt}"*(reduce(lambda x, y: x * y, shapes)), *array.flatten(order="C").tolist()))

def read_bin(filename):
    # https://qiita.com/madaikiteruyo/items/dadc99aa29f7eae0cdd0
    format_to_byte = {
        'c': 1,
        'i': 4,
        'I': 4,
        'f': 4,
        'd': 8
    }

    data = []
    dims, fmt = None, None
    with open(filename, "rb") as f:
        # read row and col (np.int = 4 bytes)
        byte = f.read(format_to_byte['i'])

        if byte == b'':
            raise Exception("read_bin: Empty binary")
        else:
            nb_dim = struct.unpack('I', byte)
        
        # Read dims
        byte = f.read(nb_dim[0] * format_to_byte['I'])
        dims = struct.unpack('I'*nb_dim[0], byte)
        # Read character format
        byte = f.read(1)
        if byte == b'':
            raise Exception("read_bin: Empty binary")
        else:
            fmt = chr(struct.unpack('c', byte)[0][0])

        if len(fmt) != 1: raise Exception("read_bin: No format dumped in binary")
        
        while True:
            byte = f.read(format_to_byte[fmt])
            if byte == b'':
                break
            else:
                data.append(struct.unpack(fmt, byte)[0])

    return np.array(data).reshape(*dims)