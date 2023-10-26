from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[str]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[str]
    body: Optional[str]

class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
