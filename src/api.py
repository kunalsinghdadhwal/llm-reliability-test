"""FastAPI backend for the LLM Reliability Gate."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference

from src.config.loader import load_config
from src.orchestrator import build_response, run_workflow
from src.wrappers.elasticsearch_helper import index_doc

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Reliability Gate", docs_url=None)


@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="LLM Reliability Gate",
    )


class EvaluateRequest(BaseModel):
    config_path: str | None = None


class EvaluateResponse(BaseModel):
    run_id: str
    model_under_test: str
    total_claims: int
    supported: int
    weakly_supported: int
    unsupported: int
    hallucination_risk: float
    reliability_score: float
    decision: str
    claims: list[dict]


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    config = load_config(request.config_path)
    state = {"config": config}
    run_workflow(state)
    result = build_response(state)

    logger.info("run_id=%s Evaluation complete, decision=%s", result["run_id"], result["decision"])

    index_doc("runs", result["run_id"], result)

    return EvaluateResponse(**result)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
