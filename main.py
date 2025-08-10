from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from services.gpt_service import generate_survey
from pydantic import BaseModel
import asyncio
import traceback

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SurveyPrompt(BaseModel):
    user_prompt: str

@app.post("/generate-survey")
async def generate_survey_endpoint(prompt: SurveyPrompt):
    try:
        # print(generate_survey.content)
        result = await asyncio.to_thread(generate_survey, prompt.user_prompt, log_callback=None)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Erreur inattendue"))

        return JSONResponse(
            content=jsonable_encoder(result["data"]))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
