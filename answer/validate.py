import json, orjson
from pydantic import BaseModel
from typing import List, Union

class Source(BaseModel):
    document: str | int
    page: int

AnswerType = Union[float, int, bool, str, List[str]]

class AnswerObj(BaseModel):
    question_id: str | int
    answer: AnswerType
    sources: List[Source]

def as_json_obj(obj: dict) -> dict:
    parsed = AnswerObj(**obj)
    return json.loads(parsed.model_dump_json())
