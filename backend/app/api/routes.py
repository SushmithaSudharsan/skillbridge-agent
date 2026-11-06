"""HTTP routes. Thin on purpose: validate, call the service, return."""

from fastapi import APIRouter, Depends, Header, Path, Request, Response

from app.schemas.api import SESSION_ID, AnalyzeRequest, AnswersRequest
from app.services.web import DemoService

router = APIRouter(prefix="/api")


def service(request: Request) -> DemoService:
    return request.app.state.service


def visitor(request: Request) -> str:
    """Who is calling, for the rate limit. Behind Azure's proxy the real IP is in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if request.app.state.settings.trust_proxy and forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def access(request: Request, x_access_code: str | None = Header(default=None)) -> None:
    service(request).check_access(x_access_code)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/samples", dependencies=[Depends(access)])
def samples(svc: DemoService = Depends(service)) -> dict:
    return svc.samples()


@router.post("/analyze", dependencies=[Depends(access)])
def analyze(body: AnalyzeRequest, svc: DemoService = Depends(service), who: str = Depends(visitor)) -> dict:
    return svc.analyze(who, body.sample_id, body.jd_text, body.title, body.years, body.internship_months)


@router.post("/sessions/{session_id}/answers", dependencies=[Depends(access)])
def answers(body: AnswersRequest, session_id: str = Path(pattern=SESSION_ID),
            svc: DemoService = Depends(service), who: str = Depends(visitor)) -> dict:
    return svc.answer(who, session_id, body.cleaned())


@router.get("/sessions/{session_id}/report.pdf", dependencies=[Depends(access)])
def report(session_id: str = Path(pattern=SESSION_ID), svc: DemoService = Depends(service)) -> Response:
    return Response(svc.report_pdf(session_id), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="skillbridge-report.pdf"',
                             "Cache-Control": "no-store"})


@router.post("/sessions/{session_id}/end")
def end(session_id: str = Path(pattern=SESSION_ID), svc: DemoService = Depends(service)) -> dict:
    """No access code: the browser calls this when the tab closes. Ending is always safe."""
    svc.end(session_id)
    return {"ended": True}
