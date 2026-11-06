"""Request bodies, validated at the boundary (type, size, range)."""

from pydantic import BaseModel, Field

SESSION_ID = r"^[0-9a-f]{32}$"


class AnalyzeRequest(BaseModel):
    sample_id: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    jd_text: str = Field(min_length=80, max_length=20000)
    title: str = Field(default="", max_length=120)
    years: float = Field(default=0, ge=0, le=50)
    internship_months: float = Field(default=0, ge=0, le=120)


class AnswersRequest(BaseModel):
    answers: list[str] = Field(max_length=12)

    def cleaned(self) -> list[str]:
        return [a[:1000] for a in self.answers]
