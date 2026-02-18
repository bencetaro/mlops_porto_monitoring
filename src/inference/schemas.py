from pydantic import BaseModel, RootModel
from typing import List, Dict, Any

class Item(RootModel[Dict[str, Any]]):
    pass

class BatchRequest(RootModel[List[Dict[str, Any]]]):
    pass

class PredictionResponse(BaseModel):
    prediction: float
