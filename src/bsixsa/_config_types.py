"""Number types for the configuration classes.

Integer options accept whole numbers written as floats (`num_live_points=1e4`)
or NumPy scalars, and refuse booleans, strings and non-whole values.
"""

from __future__ import annotations

import numbers
from typing import Annotated, Any

from pydantic import BeforeValidator, Field


def _whole_number_as_int(value: Any) -> Any:
    """Turn a whole number (`1e4`, `numpy.int64(5)`) into an `int`."""
    if isinstance(value, bool):
        return value  # refused by the strict check
    if isinstance(value, numbers.Integral) or (isinstance(value, numbers.Real) and float(value).is_integer()):
        return int(value)
    return value


Integer = Annotated[int, BeforeValidator(_whole_number_as_int), Field(strict=True)]
PositiveInt = Annotated[Integer, Field(gt=0)]
NonNegativeInt = Annotated[Integer, Field(ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0)]
