# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import json
import base64
import typing
from typing import Callable, List, Dict, Any, Optional
import pydantic
from pydantic import BaseModel
import numpy as np
from numpy.typing import NDArray
import bittensor as bt


class IsAlive(bt.Synapse):
    answer: typing.Optional[str] = None
    completion: str = pydantic.Field(
        ...,
        title="Completion",
        description="Completion status of the current request.",
    )


class CAsynapse(bt.Synapse):
    # Input data.
    steps: int = pydantic.Field(
        ...,
        title="Steps",
        description="Number of steps to run the cellular automata.",
    )
    rule_instance: str = pydantic.Field(
        ...,
        title="Rule Instance",
        description="The rule instance to use for the cellular automata.",
    )

    # Output data.
    ca_data: typing.Optional[str] = None

    required_hash_fields: typing.List[str] = pydantic.Field(
        [
            "steps",
            "rule_instance",
            "ca_data",
        ],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    def deserialize(self) -> Optional[np.ndarray]:
        if self.ca_data is None:
            return None
        self.ca_data_dict = json.loads(self.ca_data)
        self.ca_data = base64.b64decode(self.ca_data_dict["array"])
        self.ca_dtype = np.dtype(self.ca_data_dict["dtype"])
        self.ca_shape = self.ca_data_dict["shape"]
        self.ca = np.frombuffer(self.ca_data, dtype=self.ca_dtype).reshape(
            self.ca_shape
        )
        return self.ca

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: json.dumps(
                {
                    "array": base64.b64encode(v.tobytes()).decode("utf-8"),
                    "shape": v.shape,
                    "dtype": str(v.dtype),
                }
            )
        }
