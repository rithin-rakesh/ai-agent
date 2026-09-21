"""FastAPI Router for Candidate Answer Bank (Phase 5.8).

Provides management endpoints for inspecting, updating, and dry-run testing
candidate-approved answers for automated job application forms.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.automation.forms.answer_resolver import FormAnswerResolver
from app.automation.forms.models import ApplicationQuestion, QuestionInputType
from app.automation.forms.question_normalizer import QuestionNormalizer
from app.candidate_answers.bank import CandidateAnswerBank, get_candidate_answer_bank
from app.candidate_answers.models import (
    CandidateAnswer,
    CandidateAnswerTestRequest,
    CandidateAnswerTestResponse,
    CandidateAnswerUpdateRequest,
)
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/candidate/answers", tags=["Candidate Answers"])


def get_profile_service() -> ProfileService:
    """Dependency injection for ProfileService."""
    return ProfileService()


@router.get(
    "",
    summary="Get all approved candidate answers",
    description="Retrieve the complete list of candidate-approved answers from the persistent answer bank.",
)
def get_candidate_answers(
    bank: CandidateAnswerBank = Depends(get_candidate_answer_bank),
) -> Dict[str, Any]:
    """Return all approved answers in the bank."""
    answers = bank.list_answers()
    return {
        "status": "success",
        "total_answers": len(answers),
        "answers": [ans.model_dump() for ans in answers],
    }


@router.put(
    "",
    summary="Create or update candidate answer",
    description="Update or store an approved answer in the candidate answer bank with file persistence.",
)
def update_candidate_answer(
    update_req: CandidateAnswerUpdateRequest,
    bank: CandidateAnswerBank = Depends(get_candidate_answer_bank),
) -> Dict[str, Any]:
    """Add or update an answer in the bank."""
    answer = bank.set_answer(
        normalized_key=update_req.normalized_key,
        value=update_req.value,
        category=update_req.category,
        answer_type=update_req.answer_type,
        source=update_req.source or "candidate_approved",
        approved=update_req.approved if update_req.approved is not None else True,
        scope=update_req.scope or "global",
        platforms=update_req.platforms or ["indeed", "glassdoor"],
        requires_user_approval=update_req.requires_user_approval or False,
        description=update_req.description,
    )
    return {
        "status": "success",
        "message": f"Answer for '{update_req.normalized_key}' successfully updated.",
        "answer": answer.model_dump(),
        "total_answers": len(bank.list_answers()),
    }


@router.post(
    "/test",
    response_model=CandidateAnswerTestResponse,
    summary="Test question resolution against Answer Bank",
    description="Dry-run question normalization and answer resolution without running browser automation.",
)
def test_answer_resolution(
    test_req: CandidateAnswerTestRequest,
    bank: CandidateAnswerBank = Depends(get_candidate_answer_bank),
    profile_service: ProfileService = Depends(get_profile_service),
) -> CandidateAnswerTestResponse:
    """Perform a dry-run resolution of a candidate question."""
    norm_key = QuestionNormalizer.normalize(test_req.question_text)
    cat = QuestionNormalizer.classify_question_category(test_req.question_text)

    # Map input type string to QuestionInputType enum
    input_type_map = {
        "text": QuestionInputType.TEXT,
        "number": QuestionInputType.NUMBER,
        "dropdown": QuestionInputType.DROPDOWN,
        "select": QuestionInputType.DROPDOWN,
        "radio": QuestionInputType.RADIO,
        "checkbox": QuestionInputType.CHECKBOX,
        "checkbox_multi": QuestionInputType.CHECKBOX_MULTI,
    }
    q_input_type = input_type_map.get((test_req.input_type or "text").lower(), QuestionInputType.TEXT)

    app_q = ApplicationQuestion(
        text=test_req.question_text,
        normalized_key=norm_key,
        input_type=q_input_type,
        required=True,
        options=test_req.options or [],
    )

    resolver = FormAnswerResolver(
        answer_bank=bank,
        profile_service=profile_service,
        memory=None,
    )

    resolution = resolver.resolve(
        question=app_q,
        platform=test_req.platform or "glassdoor",
        host=test_req.host or "glassdoor",
    )

    return CandidateAnswerTestResponse(
        question=test_req.question_text,
        normalized_key=norm_key,
        category=cat.value,
        resolved_answer=resolution.resolved_value,
        answer_source=resolution.source,
        confidence=resolution.confidence,
        control_type=test_req.input_type or "text",
        action=resolution.action,
        is_resolved=resolution.is_resolved,
        reason=resolution.reason,
    )
