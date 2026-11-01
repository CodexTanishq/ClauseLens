"""Data models and serialization for ClauseLens Reports, Fields, Checks, and Evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """Exact character offset bounds and raw text snippet in the contract document."""

    start_char: int
    end_char: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        return cls(
            start_char=int(data.get("start_char", 0)),
            end_char=int(data.get("end_char", 0)),
            text=str(data.get("text", "")),
        )


@dataclass
class Field:
    """An individual extracted clause, date, amount, or contract entity."""

    name: str
    raw_text: str
    evidence: Evidence
    parsed: Optional[Any] = None
    score: float = 0.0
    reliability: str = "medium"  # "high" | "medium" | "low"
    status: str = "unverified"   # "found" | "not_found" | "unverified" | "not_checked"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Field:
        ev_data = data.get("evidence", {})
        ev = Evidence.from_dict(ev_data) if isinstance(ev_data, dict) else Evidence(0, 0, "")
        return cls(
            name=str(data.get("name", "")),
            raw_text=str(data.get("raw_text", "")),
            parsed=data.get("parsed"),
            evidence=ev,
            score=float(data.get("score", 0.0)),
            reliability=str(data.get("reliability", "medium")),
            status=str(data.get("status", "unverified")),
        )


@dataclass
class Check:
    """Result of an automated contract rule check with rationale and highlighted evidence."""

    id: str
    title: str
    severity: str        # "important" | "check" | "info"
    why: str             # One plain sentence explaining legal risk/relevance
    message: str         # Human-readable result text
    status: str          # "active" | "not_checked"
    evidence: Optional[Evidence] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.evidence:
            d["evidence"] = self.evidence.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Check:
        ev_data = data.get("evidence")
        ev = Evidence.from_dict(ev_data) if isinstance(ev_data, dict) else None
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            severity=str(data.get("severity", "check")),
            why=str(data.get("why", "")),
            message=str(data.get("message", "")),
            status=str(data.get("status", "active")),
            evidence=ev,
        )


@dataclass
class DocumentInfo:
    """Document metadata and classified type."""

    type: str                  # "rental_lease" | "loan" | "employment" | "other"
    type_confidence: float
    title: str
    n_chars: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentInfo:
        return cls(
            type=str(data.get("type", "other")),
            type_confidence=float(data.get("type_confidence", 0.0)),
            title=str(data.get("title", "")),
            n_chars=int(data.get("n_chars", 0)),
        )


@dataclass
class TenureInfo:
    """Computed or extracted contract tenure duration."""

    months: Optional[float] = None
    days: Optional[int] = None
    source: str = "unknown"    # "from dates" | "from duration text" | "unknown"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TenureInfo:
        return cls(
            months=data.get("months"),
            days=data.get("days"),
            source=str(data.get("source", "unknown")),
            note=str(data.get("note", "")),
        )


@dataclass
class MetaInfo:
    """Model information, app version, and standard legal disclaimer."""

    option_name: str
    model_id: str
    app_version: str
    created_at: str
    disclaimer: str = "This tool helps you read a contract. It is not legal advice."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetaInfo:
        return cls(
            option_name=str(data.get("option_name", "")),
            model_id=str(data.get("model_id", "")),
            app_version=str(data.get("app_version", "")),
            created_at=str(data.get("created_at", "")),
            disclaimer=str(data.get("disclaimer", "This tool helps you read a contract. It is not legal advice.")),
        )


@dataclass
class Report:
    """Consolidated contract analysis report containing all findings and checks."""

    document: DocumentInfo
    parties: List[Dict[str, Any]]
    dates: Dict[str, Optional[Field]]
    tenure: TenureInfo
    key_terms: Dict[str, Optional[Field]]
    checks: List[Check]
    summary: str
    meta: MetaInfo

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "parties": self.parties,
            "dates": {k: (v.to_dict() if v else None) for k, v in self.dates.items()},
            "tenure": self.tenure.to_dict(),
            "key_terms": {k: (v.to_dict() if v else None) for k, v in self.key_terms.items()},
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
            "meta": self.meta.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Report:
        doc_info = DocumentInfo.from_dict(data.get("document", {}))
        tenure_info = TenureInfo.from_dict(data.get("tenure", {}))
        meta_info = MetaInfo.from_dict(data.get("meta", {}))

        dates_raw = data.get("dates", {})
        dates_fields = {k: (Field.from_dict(v) if v else None) for k, v in dates_raw.items()}

        terms_raw = data.get("key_terms", {})
        terms_fields = {k: (Field.from_dict(v) if v else None) for k, v in terms_raw.items()}

        checks_list = [Check.from_dict(c) for c in data.get("checks", [])]

        return cls(
            document=doc_info,
            parties=data.get("parties", []),
            dates=dates_fields,
            tenure=tenure_info,
            key_terms=terms_fields,
            checks=checks_list,
            summary=str(data.get("summary", "")),
            meta=meta_info,
        )

