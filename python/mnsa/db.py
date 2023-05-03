import sqlite3
import numpy as np


dtype2type = {np.int32: 'INTEGER',
              np.int64: 'INTEGER',
              np.float32: 'REAL',
              np.float64: 'REAL',
              np.str_: 'TEXT'}


def make_create_table(name=None, data=None,
                      response_names=['F', 'N', 'g', 'r', 'z', 'W1', 'W2',
                                      'W3', 'W4']):
    """Create a CREATE TABLE statement based on an ndarray

    Parameters
    ----------

    name : str
        name of table

    data : ndarray
        array to base on

    response_names : list of str
        list of names to append to flux-related columns

    Returns
    -------

    statement : str
        CREATE TABLE statement
"""
    statement = "CREATE TABLE {name} (\n".format(name=name)
    for n in data.dtype.names:
        t = data[n][0]
        if(type(t) == np.ndarray):
            print(n)
            print(t)
            for i, r in enumerate(response_names):
                col = n + '_' + r
                statement = statement + "{col} {tp},\n".format(col=col,
                                                               tp=dtype2type[type(t[i])])
        else:
            statement = statement + "{col} {tp},\n".format(col=n,
                                                           tp=dtype2type[type(t)])

    return(statement)
