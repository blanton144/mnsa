import os
import sqlite3
import numpy as np


class MNSAdb(object):
    def __init__(self, version='0.3.0'):
        self.connection = sqlite3.connect(os.path.join(os.getenv('MNSA_DATA'),
                                                       version,
                                                       'mnsa-{v}.db'.format(v=version)))
        return


class MNSADataTableSQLite(object):

    def __init__(self, name=None, data=None, primary_key=None, mnsadb=None,
                 mnsa_response_names=['F', 'N', 'g', 'r', 'z', 'W1', 'W2',
                                      'W3', 'W4'],
                 nsa_response_names=['F', 'N', 'u', 'g', 'r', 'i', 'z']):
        self.name = name
        self.data = data
        self.primary_key = primary_key
        self.mnsa_response_names = mnsa_response_names
        self.nsa_response_names = nsa_response_names
        self.dtype2type = {np.int32: 'INTEGER',
                           np.int64: 'INTEGER',
                           np.bool_: 'INTEGER',
                           np.float32: 'REAL',
                           np.float64: 'REAL',
                           np.str_: 'TEXT'}
        if(mnsadb is None):
            self.mnsadb = MNSAdb()
        else:
            self.mnsadb = mnsadb
        self.mnsadb_cursor = self.mnsadb.connection.cursor()
        return

    def make_create_table(self):
        """Create a CREATE TABLE statement based on an ndarray
        
        Returns
        -------

        statement : str
            CREATE TABLE statement
"""
        statement = "CREATE TABLE {name} (\n".format(name=self.name)
        for n in self.data.dtype.names:
            t = self.data[n][0]
            if(type(t) == np.ndarray):
                if(n[0:4] == 'nsa_'):
                    response_names = self.nsa_response_names
                else:
                    response_names = self.mnsa_response_names
                for i, r in enumerate(response_names):
                    col = n + '_' + r
                    statement = statement + "{col} {tp},\n".format(col=col,
                                                                   tp=self.dtype2type[type(t[i])])
            else:
                statement = statement + "{col} {tp},\n".format(col=n,
                                                               tp=self.dtype2type[type(t)])

        if(self.primary_key is None):
            self.primary_key = 'pk'
            statement = statement + "pk INTEGER,\n"
        else:
            statement = statement + "PRIMARY KEY ({pk} ASC) );".format(pk=self.primary_key)

        return(statement)

    def drop_table(self):
        self.mnsadb_cursor.execute("DROP TABLE {n}".format(n=self.name))
        return

    def create_table(self):
        """Create the data table"""
        statement = self.make_create_table()
        self.mnsadb_cursor.execute(statement)
        return

    def load_table(self):
        """Load the data table in sqllite3"""
        datalist = []
        for n in self.data.dtype.names:
            if(type(self.data[n][0]) == np.ndarray):
                for i in np.arange(len(self.data[n][0]), dtype=int):
                    col = list(self.data[n][:, i])
                    datalist.append(col)
            else:
                col = list(self.data[n])
                # SQLLite does not like numpy integer array input
                if(type(col[0]) in [np.int32, np.int64]):
                    col = [int(x) for x in col]
                datalist.append(col)
        statement = "INSERT INTO {n} VALUES (".format(n=self.name)
        for i in np.arange(len(datalist), dtype=int):
            statement = statement + "?,"
        statement = statement[0:-1]
        statement = statement + ")"
        datalist_transpose = list(map(list, zip(*datalist)))
        self.mnsadb_cursor.executemany(statement, datalist_transpose)
        self.mnsadb.connection.commit()
        return
