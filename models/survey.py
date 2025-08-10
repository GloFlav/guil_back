# models/survey.py

from pydantic import BaseModel
from typing import List, Optional

class ExpectedAnswer(BaseModel):
    answer_id: str
    answer_type: str
    next_question_id: Optional[str] = None

class Question(BaseModel):
    question_id: str
    question_type: str
    question_text: str
    expected_answers: List[ExpectedAnswer]
    predecessor_answer_id: Optional[str] = None

class Category(BaseModel):
    category: str
    questions: List[Question]

class SurveyResponse(BaseModel):
    introduction: str
    title: str
    survey_total_duration: str
    number_of_respondents: int
    number_of_investigators: int
    number_of_locations: int
    location_characteristics: str
    survey: List[Category]
