"""Minimal SQLAlchemy type for pgvector.

The extension remains required on PostgreSQL. A cast is explicit on bound values,
so psycopg string parameter typing cannot change vector operator resolution.
"""
import numpy as np
from sqlalchemy import cast
from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int):
        self.dimensions = int(dimensions)

    def get_col_spec(self, **kw):
        return f"vector({self.dimensions})"

    def bind_expression(self, bindvalue):
        return cast(bindvalue, self)

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            x = np.asarray(value, dtype=np.float32)
            if x.shape != (self.dimensions,) or not np.isfinite(x).all():
                raise ValueError("Invalid vector dimension or nonfinite value")
            return "[" + ",".join(format(float(v), ".9g") for v in x) + "]"
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                result = np.fromstring(value.strip("[]"), sep=",", dtype=np.float32)
            else:
                result = np.asarray(value, dtype=np.float32)
            if result.shape != (self.dimensions,):
                raise ValueError("Stored vector dimension mismatch")
            return result
        return process
