"""
Breast Density Assessment from Mammography — governance-aware Streamlit prototype.

IMPORTANT
---------
This application uses ONLY synthetic, simulated, de-identified demo data.
It is NOT a medical device and MUST NOT be used for clinical decision-making.

Purpose
-------
Provide a runnable, production-minded prototype that demonstrates:
- synthetic longitudinal case generation
- patient-level split policy and leakage checks
- DICOM-like metadata de-identification preview
- burned-in annotation / shortcut-learning demo
- ensemble-style uncertainty and deferral
- OOD and scanner-drift monitoring
- calibration diagnostics
- subgroup performance review
- structured report templates
- audit hash chain and retention preview
- role-based demo access control
- in-app verification/test suite
- auto-generated live demo media

Design notes
------------
- All code is intentionally kept in this single file as requested.
- No API keys are used.
- No real patient data is generated or assumed.
- Thresholds, paths, seeds, and demo parameters are configurable via environment variables.
- The simulated model is a deterministic stand-in for a future PyTorch CNN.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# -----------------------------------------------------------------------------
# Types and constants
# -----------------------------------------------------------------------------

DensityCategory = Literal["a", "b", "c", "d"]

CATEGORY_ORDER: Tuple[DensityCategory, ...] = ("a", "b", "c", "d")

CATEGORY_LABELS: Dict[DensityCategory, str] = {
    "a": "Almost entirely fatty",
    "b": "Scattered areas of fibroglandular density",
    "c": "Heterogeneously dense",
    "d": "Extremely dense",
}

CATEGORY_CENTERS: Dict[DensityCategory, float] = {
    "a": 0.15,
    "b": 0.35,
    "c": 0.60,
    "d": 0.85,
}

VIEWS: Tuple[str, ...] = ("CC", "MLO")

REQUIRED_DEID_TAGS: Tuple[str, ...] = (
    "PatientID",
    "StudyDate",
    "SiteCode",
    "ScannerModel",
    "Laterality",
    "View",
    "PixelSpacing",
    "ConsentStatus",
)

FORBIDDEN_DEID_TAGS: Tuple[str, ...] = (
    "PatientName",
    "AccessionNumber",
    "OperatorName",
    "StationName",
    "InstitutionName",
)

REPORT_TEMPLATES: Tuple[str, ...] = (
    "birads_like",
    "screening_risk_note",
    "audit_summary",
)

ROLE_PERMISSIONS: Dict[str, Dict[str, bool]] = {
    "admin": {
        "view_audit_all": True,
        "confirm": True,
        "feedback": True,
        "export": True,
        "edit_thresholds": True,
    },
    "radiologist": {
        "view_audit_all": False,
        "confirm": True,
        "feedback": True,
        "export": True,
        "edit_thresholds": True,
    },
    "technologist": {
        "view_audit_all": False,
        "confirm": False,
        "feedback": True,
        "export": False,
        "edit_thresholds": True,
    },
    "auditor": {
        "view_audit_all": True,
        "confirm": False,
        "feedback": False,
        "export": True,
        "edit_thresholds": False,
    },
    "viewer": {
        "view_audit_all": False,
        "confirm": False,
        "feedback": False,
        "export": False,
        "edit_thresholds": False,
    },
}


# -----------------------------------------------------------------------------
# Minimal i18n scaffold
# -----------------------------------------------------------------------------

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "Breast Density Assessment from Mammography",
        "subtitle": "Governance-aware Streamlit prototype using synthetic demo data only.",
        "warning": "Synthetic data only. Decision support only. Clinician remains responsible.",
        "tab_demo": "Live demo",
        "tab_case": "Case explorer",
        "tab_data": "Data governance",
        "tab_model": "Model diagnostics",
        "tab_clinical": "Clinical workflow",
        "tab_audit": "Audit & risk",
        "tab_validate": "Validation",
        "role": "Demo role",
        "language": "Language",
        "confidence_threshold": "Confidence threshold",
        "uncertainty_threshold": "Uncertainty defer threshold",
        "ood_threshold": "OOD defer threshold",
        "regenerate_media": "Regenerate demo video",
        "predicted_category": "Predicted category",
        "confidence": "Confidence",
        "uncertainty": "Uncertainty",
        "total_uncertainty": "Total uncertainty",
        "status": "Status",
        "defer": "Defer to radiologist",
        "no_defer": "Below defer thresholds",
        "download_manifest": "Download data manifest",
        "download_report": "Download report JSON",
        "run_tests": "Run verification suite",
        "synthetic_only": "Synthetic demo output only. Not for clinical use.",
    },
    "es": {
        "app_title": "Evaluación de densidad mamaria por mamografía",
        "subtitle": "Prototipo Streamlit con gobernanza usando solo datos sintéticos.",
        "warning": "Solo datos sintéticos. Solo apoyo a decisiones. El clínico sigue siendo responsable.",
        "tab_demo": "Demo en vivo",
        "tab_case": "Explorador de casos",
        "tab_data": "Gobernanza de datos",
        "tab_model": "Diagnóstico del modelo",
        "tab_clinical": "Flujo clínico",
        "tab_audit": "Auditoría y riesgo",
        "tab_validate": "Validación",
        "role": "Rol de demostración",
        "language": "Idioma",
        "confidence_threshold": "Umbral de confianza",
        "uncertainty_threshold": "Umbral de incertidumbre",
        "ood_threshold": "Umbral OOD",
        "regenerate_media": "Regenerar video de demostración",
        "predicted_category": "Categoría predicha",
        "confidence": "Confianza",
        "uncertainty": "Incertidumbre",
        "total_uncertainty": "Incertidumbre total",
        "status": "Estado",
        "defer": "Derivar al radiólogo",
        "no_defer": "Por debajo de umbrales de derivación",
        "download_manifest": "Descargar manifiesto de datos",
        "download_report": "Descargar informe JSON",
        "run_tests": "Ejecutar suite de verificación",
        "synthetic_only": "Solo salida sintética de demostración. No para uso clínico.",
    },
    "fr": {
        "app_title": "Évaluation de la densité mammaire par mammographie",
        "subtitle": "Prototype Streamlit gouverné utilisant uniquement des données synthétiques.",
        "warning": "Données synthétiques uniquement. Aide à la décision seulement. Le clinicien reste responsable.",
        "tab_demo": "Démo en direct",
        "tab_case": "Explorateur de cas",
        "tab_data": "Gouvernance des données",
        "tab_model": "Diagnostic du modèle",
        "tab_clinical": "Flux clinique",
        "tab_audit": "Audit et risque",
        "tab_validate": "Validation",
        "role": "Rôle de démo",
        "language": "Langue",
        "confidence_threshold": "Seuil de confiance",
        "uncertainty_threshold": "Seuil d'incertitude",
        "ood_threshold": "Seuil OOD",
        "regenerate_media": "Régénérer la vidéo de démo",
        "predicted_category": "Catégorie prédite",
        "confidence": "Confiance",
        "uncertainty": "Incertitude",
        "total_uncertainty": "Incertitude totale",
        "status": "Statut",
        "defer": "Transférer au radiologue",
        "no_defer": "Sous les seuils de transfert",
        "download_manifest": "Télécharger le manifeste de données",
        "download_report": "Télécharger le rapport JSON",
        "run_tests": "Exécuter la suite de vérification",
        "synthetic_only": "Sortie synthétique de démo uniquement. Pas pour usage clinique.",
    },
    "hi": {
        "app_title": "मैमोग्राफी से स्तन घनत्व आकलन",
        "subtitle": "केवल सिंथेटिक डेमो डेटा का उपयोग करने वाला गवर्नेंस-अवेयर स्ट्रीमलिट प्रोटोटाइप।",
        "warning": "केवल सिंथेटिक डेटा। केवल निर्णय-सहायता। अंतिम जिम्मेदारी चिकित्सक की है।",
        "tab_demo": "लाइव डेमो",
        "tab_case": "केस एक्सप्लोरर",
        "tab_data": "डेटा गवर्नेंस",
        "tab_model": "मॉडल डायग्नोस्टिक्स",
        "tab_clinical": "क्लिनिकल वर्कफ़्लो",
        "tab_audit": "ऑडिट और जोखिम",
        "tab_validate": "सत्यापन",
        "role": "डेमो भूमिका",
        "language": "भाषा",
        "confidence_threshold": "विश्वास सीमा",
        "uncertainty_threshold": "अनिश्चितता सीमा",
        "ood_threshold": "OOD सीमा",
        "regenerate_media": "डेमो वीडियो पुनः तैयार करें",
        "predicted_category": "अनुमानित श्रेणी",
        "confidence": "विश्वास",
        "uncertainty": "अनिश्चितता",
        "total_uncertainty": "कुल अनिश्चितता",
        "status": "स्थिति",
        "defer": "रेडियोलॉजिस्ट को भेजें",
        "no_defer": "सीमाओं से नीचे",
        "download_manifest": "डेटा मेनिफेस्ट डाउनलोड करें",
        "download_report": "रिपोर्ट JSON डाउनलोड करें",
        "run_tests": "सत्यापन सूट चलाएं",
        "synthetic_only": "केवल सिंथेटिक डेमो आउटपुट। नैदानिक उपयोग के लिए नहीं।",
    },
    "ar": {
        "app_title": "تقييم كثافة الثدي من التصوير الشعاعي للثدي",
        "subtitle": "نموذج أولي لـ Streamlit مع حوكمة باستخدام بيانات تجريبية اصطناعية فقط.",
        "warning": "بيانات اصطناعية فقط. دعم قرار فقط. يبقى الطبيب مسؤولًا.",
        "tab_demo": "عرض حي",
        "tab_case": "مستكشف الحالات",
        "tab_data": "حوكمة البيانات",
        "tab_model": "تشخيص النموذج",
        "tab_clinical": "سير العمل السريري",
        "tab_audit": "التدقيق والمخاطر",
        "tab_validate": "التحقق",
        "role": "دور العرض",
        "language": "اللغة",
        "confidence_threshold": "عتبة الثقة",
        "uncertainty_threshold": "عتبة عدم اليقين",
        "ood_threshold": "عتبة OOD",
        "regenerate_media": "إعادة توليد فيديو العرض",
        "predicted_category": "الفئة المتوقعة",
        "confidence": "الثقة",
        "uncertainty": "عدم اليقين",
        "total_uncertainty": "إجمالي عدم اليقين",
        "status": "الحالة",
        "defer": "تحويل إلى أخصائي أشعة",
        "no_defer": "أقل من عتبات التحويل",
        "download_manifest": "تنزيل بيان البيانات",
        "download_report": "تنزيل التقرير JSON",
        "run_tests": "تشغيل مجموعة التحقق",
        "synthetic_only": "مخرجات عرض اصطناعية فقط. ليس للاستخدام السريري.",
    },
}


def tr(key: str, lang: str = "en") -> str:
    """Translate a UI key with English fallback."""
    en = TRANSLATIONS["en"]
    cur = TRANSLATIONS.get(lang, en)
    return cur.get(key, en.get(key, key))


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoConfig:
    """Runtime configuration loaded from environment variables."""

    seed: int
    num_cases: int
    num_ensemble_members: int
    video_fps: int
    video_seconds: float
    confidence_threshold: float
    uncertainty_threshold: float
    ood_threshold: float
    split_train: float
    split_validation: float
    class_distribution: Tuple[float, ...]
    heat_thresholds: Tuple[float, ...]
    probability_temperature: float
    ensemble_noise: float
    reader_a_accuracy: float
    reader_b_accuracy: float
    reader_baseline_accuracy: float
    tool_assist_gain: float
    automation_bias_penalty: float
    baseline_reading_time_s: float
    tool_time_save_s: float
    drift_fraction: float
    marker_fraction: float
    consent_withdrawn_fraction: float
    remove_burned_in_annotations: bool
    target_sensitivity_dense: float
    target_specificity_non_dense: float
    max_deferral_rate: float
    language: str
    output_dir: Path
    audit_log_path: Path
    retention_days: int
    demo_mode: str
    model_version: str
    schema_version: str


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float_tuple(name: str, default: str, expected: int) -> Tuple[float, ...]:
    raw = _env_str(name, default)
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} comma-separated floats.")
    return tuple(values)


def load_config() -> DemoConfig:
    """Load configuration from environment variables with safe demo defaults."""
    output_dir = Path(_env_str("BDA_OUTPUT_DIR", tempfile.gettempdir()))
    audit_log_path = Path(
        _env_str("BDA_AUDIT_LOG", str(output_dir / "breast_density_audit.jsonl"))
    )

    split_train = _env_float("BDA_SPLIT_TRAIN", 0.70)
    split_validation = _env_float("BDA_SPLIT_VALIDATION", 0.15)
    if split_train <= 0 or split_validation <= 0 or (split_train + split_validation) >= 1.0:
        raise ValueError("BDA_SPLIT_TRAIN and BDA_SPLIT_VALIDATION must be positive and sum < 1.")

    class_distribution = _env_float_tuple(
        "BDA_CLASS_DISTRIBUTION",
        "0.20,0.35,0.30,0.15",
        4,
    )
    total = sum(class_distribution)
    class_distribution = tuple(float(x / total) for x in class_distribution)

    heat_thresholds = _env_float_tuple(
        "BDA_HEAT_THRESHOLDS",
        "0.58,0.48,0.38,0.28",
        4,
    )

    language = _env_str("BDA_LANGUAGE", "en").lower()
    if language not in TRANSLATIONS:
        language = "en"

    return DemoConfig(
        seed=_env_int("BDA_DEMO_SEED", 1234),
        num_cases=_env_int("BDA_NUM_CASES", 24),
        num_ensemble_members=_env_int("BDA_ENSEMBLE_MEMBERS", 5),
        video_fps=_env_int("BDA_VIDEO_FPS", 6),
        video_seconds=_env_float("BDA_VIDEO_SECONDS", 6.0),
        confidence_threshold=_env_float("BDA_CONFIDENCE_THRESHOLD", 0.72),
        uncertainty_threshold=_env_float("BDA_UNCERTAINTY_THRESHOLD", 0.35),
        ood_threshold=_env_float("BDA_OOD_THRESHOLD", 0.65),
        split_train=split_train,
        split_validation=split_validation,
        class_distribution=class_distribution,
        heat_thresholds=heat_thresholds,
        probability_temperature=_env_float("BDA_PROBABILITY_TEMPERATURE", 0.09),
        ensemble_noise=_env_float("BDA_ENSEMBLE_NOISE", 0.035),
        reader_a_accuracy=_env_float("BDA_READER_A_ACCURACY", 0.78),
        reader_b_accuracy=_env_float("BDA_READER_B_ACCURACY", 0.74),
        reader_baseline_accuracy=_env_float("BDA_READER_BASELINE_ACCURACY", 0.76),
        tool_assist_gain=_env_float("BDA_TOOL_ASSIST_GAIN", 0.08),
        automation_bias_penalty=_env_float("BDA_AUTOMATION_BIAS_PENALTY", 0.12),
        baseline_reading_time_s=_env_float("BDA_BASELINE_READING_TIME_S", 90.0),
        tool_time_save_s=_env_float("BDA_TOOL_TIME_SAVE_S", 12.0),
        drift_fraction=_env_float("BDA_DRIFT_FRACTION", 0.12),
        marker_fraction=_env_float("BDA_MARKER_FRACTION", 0.18),
        consent_withdrawn_fraction=_env_float("BDA_CONSENT_WITHDRAWN_FRACTION", 0.05),
        remove_burned_in_annotations=_env_bool("BDA_REMOVE_BURNED_IN_ANNOTATIONS", True),
        target_sensitivity_dense=_env_float("BDA_TARGET_SENSITIVITY_DENSE", 0.80),
        target_specificity_non_dense=_env_float("BDA_TARGET_SPECIFICITY_NON_DENSE", 0.70),
        max_deferral_rate=_env_float("BDA_MAX_DEFERRAL_RATE", 0.35),
        language=language,
        output_dir=output_dir,
        audit_log_path=audit_log_path,
        retention_days=_env_int("BDA_RETENTION_DAYS", 30),
        demo_mode=_env_str("BDA_DEMO_MODE", "synthetic").lower(),
        model_version=_env_str("BDA_MODEL_VERSION", "synthetic-baseline-0.2.1"),
        schema_version=_env_str("BDA_SCHEMA_VERSION", "bda.report.v2"),
    )


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert common non-JSON types to JSON-serializable values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hex digest for a JSON-like object."""
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_hash(value: Any, modulus: int) -> int:
    """Return a deterministic integer hash modulo a positive integer."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(stable_hash(value), 16) % modulus


def stable_seed(text: str) -> int:
    """Return a deterministic 32-bit seed from a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)


def config_hash(config: DemoConfig) -> str:
    """Hash the full configuration for auditability."""
    return stable_hash(asdict(config))


def rerun_app() -> None:
    """Compatibility wrapper for Streamlit rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def can(role: str, permission: str) -> bool:
    """Check demo role permission."""
    return bool(ROLE_PERMISSIONS.get(role, {}).get(permission, False))


# -----------------------------------------------------------------------------
# Math and metrics
# -----------------------------------------------------------------------------


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def normalized_entropy(probabilities: Dict[str, float]) -> float:
    """Return entropy normalized to [0, 1]."""
    p = np.array([probabilities.get(c, 0.0) for c in CATEGORY_ORDER], dtype=float)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(len(CATEGORY_ORDER)))


def average_probability_dicts(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average a sequence of probability dictionaries and renormalize."""
    if not dicts:
        return {c: 0.25 for c in CATEGORY_ORDER}
    avg = {c: float(np.mean([d.get(c, 0.0) for d in dicts])) for c in CATEGORY_ORDER}
    s = sum(avg.values())
    if s <= 0:
        return {c: 0.25 for c in CATEGORY_ORDER}
    return {c: float(v / s) for c, v in avg.items()}


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Compute Cohen's kappa for two label sequences."""
    a = list(labels_a)
    b = list(labels_b)
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")

    categories = sorted(set(a).union(set(b)))
    po = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    pe = 0.0
    for c in categories:
        pa = a.count(c) / float(n)
        pb = b.count(c) / float(n)
        pe += pa * pb

    if math.isclose(pe, 1.0, abs_tol=1e-12):
        return 1.0 if math.isclose(po, 1.0, abs_tol=1e-12) else 0.0

    return float((po - pe) / (1.0 - pe))


def weighted_kappa_ordinal(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Compute quadratic-weighted kappa for ordinal density categories."""
    a = list(labels_a)
    b = list(labels_b)
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")

    try:
        ia = np.array([CATEGORY_ORDER.index(x) for x in a], dtype=int)
        ib = np.array([CATEGORY_ORDER.index(x) for x in b], dtype=int)
    except ValueError:
        return float("nan")

    k = len(CATEGORY_ORDER)
    observed = np.zeros((k, k), dtype=float)
    for x, y in zip(ia, ib):
        observed[x, y] += 1.0

    marginal_a = observed.sum(axis=1) / n
    marginal_b = observed.sum(axis=0) / n
    expected = np.outer(marginal_a, marginal_b)

    weight = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            weight[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    num = float(np.sum(weight * observed))
    den = float(np.sum(weight * expected))
    if math.isclose(den, 0.0, abs_tol=1e-12):
        return 1.0 if math.isclose(num, 0.0, abs_tol=1e-12) else 0.0
    return float(1.0 - num / den)


def _average_rank(scores: np.ndarray) -> np.ndarray:
    """Compute average ranks with tie handling."""
    n = len(scores)
    sorter = np.argsort(scores)
    ranks = np.empty(n, dtype=float)
    sorted_scores = scores[sorter]

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[sorter[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary AUROC using rank statistic with tie handling."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]

    n_pos = int(np.sum(labels))
    n_neg = int(np.sum(~labels))
    if n_pos == 0 or n_neg == 0 or scores.size < 2:
        return float("nan")

    ranks = _average_rank(scores)
    rank_sum_pos = float(np.sum(ranks[labels]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def multiclass_auroc(eval_df: pd.DataFrame, outputs: Dict[str, "EnsembleOutput"]) -> float:
    """Compute macro one-vs-rest AUROC for density categories."""
    if eval_df.empty:
        return float("nan")

    aucs: List[float] = []
    for c in CATEGORY_ORDER:
        labels = (eval_df["consensus_label"] == c).to_numpy(dtype=bool)
        scores = np.array([outputs[str(cid)].mean_probabilities[c] for cid in eval_df["case_id"]], dtype=float)
        auc = binary_auroc(scores, labels)
        if not math.isnan(auc):
            aucs.append(auc)

    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Any,
    n_boot: int = 100,
    seed: int = 999,
) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval for a scalar metric function."""
    if df.empty or len(df) < 3:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values: List[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sub = df.iloc[idx]
        val = float(metric_fn(sub))
        if not math.isnan(val):
            values.append(val)

    if len(values) < 2:
        return float("nan"), float("nan")

    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def dense_sensitivity_specificity(eval_df: pd.DataFrame) -> Tuple[float, float]:
    """Compute sensitivity for dense (c/d) and specificity for non-dense (a/b)."""
    if eval_df.empty:
        return float("nan"), float("nan")

    y = eval_df["consensus_dense"].to_numpy(dtype=bool)
    p = eval_df["model_dense"].to_numpy(dtype=bool)

    tp = int(np.sum(y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    fp = int(np.sum(~y & p))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return sens, spec


def calibration_table(eval_df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Build reliability table from confidence and correctness."""
    if eval_df.empty:
        return pd.DataFrame(columns=["bin_center", "count", "avg_confidence", "accuracy"])

    conf = eval_df["confidence"].to_numpy(dtype=float)
    corr = eval_df["correct"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    rows: List[Dict[str, float]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        rows.append(
            {
                "bin_center": float((lo + hi) / 2.0),
                "count": float(count),
                "avg_confidence": float(np.mean(conf[mask])),
                "accuracy": float(np.mean(corr[mask])),
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Compute ECE from a calibration table."""
    if table.empty:
        return float("nan")
    n = float(table["count"].sum())
    if n <= 0:
        return float("nan")
    ece = float(np.sum((table["count"] / n) * np.abs(table["accuracy"] - table["avg_confidence"])))
    return ece


def calibrated_probabilities(probabilities: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Apply temperature scaling to a probability vector."""
    eps = 1e-12
    t = max(float(temperature), 1e-3)
    logits = np.array([math.log(probabilities.get(c, eps) + eps) for c in CATEGORY_ORDER], dtype=float) / t
    probs = softmax(logits)
    return {c: float(p) for c, p in zip(CATEGORY_ORDER, probs)}


def fit_temperature(
    eval_df: pd.DataFrame,
    outputs: Dict[str, "EnsembleOutput"],
    split: str = "validation",
) -> float:
    """Fit a temperature on a chosen split using NLL."""
    subset = eval_df[eval_df["split"] == split]
    if subset.empty:
        subset = eval_df[eval_df["consent_status"] == "active"]
    if subset.empty:
        return 1.0

    best_t = 1.0
    best_nll = float("inf")
    eps = 1e-12

    for t in np.linspace(0.5, 3.0, 26):
        nll = 0.0
        for _, row in subset.iterrows():
            out = outputs[str(row["case_id"])]
            cal = calibrated_probabilities(out.mean_probabilities, float(t))
            label = str(row["consensus_label"])
            nll -= math.log(cal.get(label, eps) + eps)
        nll /= max(1, len(subset))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return best_t


def compute_ece_for_cases(
    eval_df: pd.DataFrame,
    outputs: Dict[str, "EnsembleOutput"],
    temperature: float = 1.0,
) -> Tuple[float, pd.DataFrame]:
    """Compute ECE and reliability table using optionally calibrated confidences."""
    if eval_df.empty:
        return float("nan"), pd.DataFrame(columns=["bin_center", "count", "avg_confidence", "accuracy"])

    rows = []
    for _, row in eval_df.iterrows():
        out = outputs[str(row["case_id"])]
        cal = calibrated_probabilities(out.mean_probabilities, temperature)
        pred = max(cal, key=lambda k: cal[k])
        conf = float(cal[pred])
        correct = float(pred == row["consensus_label"])
        rows.append({"confidence": conf, "correct": correct})

    tmp = pd.DataFrame(rows)
    table = calibration_table(tmp)
    ece = expected_calibration_error(table)
    return ece, table


# -----------------------------------------------------------------------------
# Image and synthetic signal utilities
# -----------------------------------------------------------------------------


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """Create a simple elliptical breast-region mask."""
    y, x = np.ogrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.38), int(h * 0.43)
    return (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) <= 1.0


def _float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8."""
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def generate_synthetic_mammogram(
    rng: np.random.Generator,
    category: DensityCategory,
    site: str,
    scanner: str,
    view: str,
) -> np.ndarray:
    """
    Generate a synthetic mammogram-like image.

    This is NOT real anatomy. It is a controllable visual proxy for demo purposes.
    """
    h = w = 384
    mask = _ellipse_mask(h, w)

    base = CATEGORY_CENTERS[category]
    site_offset = {"SITE_A": 0.0, "SITE_B": 0.025, "SITE_C": -0.015}.get(site, 0.0)
    scanner_offset = 0.012 if "DRIFT" in scanner or scanner.endswith("v2") else 0.0
    view_offset = 0.01 if view == "MLO" else 0.0
    target = float(np.clip(base + site_offset + scanner_offset + view_offset + rng.normal(0.0, 0.018), 0.05, 0.95))

    img = np.full((h, w), 0.03, dtype=np.float32)
    img += rng.normal(0.0, 0.006, size=(h, w)).astype(np.float32)

    ys, xs = np.nonzero(mask)
    num_blobs = int(25 + target * 160)

    for _ in range(num_blobs):
        idx = int(rng.integers(0, len(xs)))
        cy = int(ys[idx])
        cx = int(xs[idx])
        r = int(rng.integers(4, 18))

        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

        region = img[y0:y1, x0:x1]
        intensity = float(np.clip(rng.normal(target * 0.95, 0.08), 0.05, 1.0))
        region[blob] = np.maximum(region[blob], intensity)
        img[y0:y1, x0:x1] = region

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img += rng.normal(0.0, 0.012, size=(h, w)).astype(np.float32)
    img[~mask] *= 0.18

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def add_burned_in_marker(image: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Add a synthetic burned-in marker and return marker plus boolean mask."""
    _ = rng
    marked = image.copy()
    h, w = marked.shape
    mask = np.zeros((h, w), dtype=bool)

    x0 = int(w * 0.04)
    y0 = int(h * 0.04)
    x1 = min(w, x0 + 42)
    y1 = min(h, y0 + 18)

    marked[y0:y1, x0:x1] = 0.98

    temp = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(
        temp,
        "SYN",
        (x0 + 3, y0 + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        255,
        1,
        cv2.LINE_AA,
    )
    marked[temp > 0] = 0.05

    mask[y0:y1, x0:x1] = True
    return marked, mask


def remove_burned_in_annotations(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove synthetic burned-in marker region by replacing with background."""
    cleaned = image.copy()
    cleaned[mask] = 0.04
    return cleaned


def generate_mammogram_pair(
    rng: np.random.Generator,
    category: DensityCategory,
    site: str,
    scanner: str,
    view: str,
    marker_present: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate clean, marked, and mask arrays for one synthetic view."""
    base = generate_synthetic_mammogram(rng, category, site, scanner, view)
    if marker_present:
        marked, mask = add_burned_in_marker(base, rng)
    else:
        marked = base.copy()
        mask = np.zeros_like(base, dtype=bool)
    clean = remove_burned_in_annotations(marked, mask)
    return clean, marked, mask


def estimate_glandular_fraction(image: np.ndarray) -> float:
    """
    Estimate a crude glandular-fraction-like scalar from the synthetic image.

    This is not a validated measurement. It is only a deterministic demo feature.
    """
    mask = _ellipse_mask(image.shape[0], image.shape[1])
    vals = image[mask]
    if vals.size == 0:
        return 0.0

    active = vals[vals > 0.25]
    activity = float(np.mean(vals > 0.25))
    intensity = float(np.mean(active)) if active.size else 0.0
    return float(np.clip(0.55 * activity + 0.45 * intensity, 0.0, 1.0))


def check_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """Run basic integrity checks on a synthetic image."""
    issues: List[str] = []

    finite = bool(np.all(np.isfinite(image)))
    if not finite:
        issues.append("non_finite_pixels")

    shape_ok = bool(image.ndim == 2 and image.shape[0] >= 128 and image.shape[1] >= 128)
    if not shape_ok:
        issues.append("invalid_shape")

    if finite:
        minv = float(np.min(image))
        maxv = float(np.max(image))
    else:
        minv = float("nan")
        maxv = float("nan")

    intensity_ok = bool(finite and minv >= -1e-6 and maxv <= 1.0 + 1e-6)
    if not intensity_ok:
        issues.append("intensity_out_of_range")

    contrast_ok = bool(finite and (maxv - minv) > 0.05)
    if not contrast_ok:
        issues.append("low_contrast")

    return {
        "passed": len(issues) == 0,
        "issues": tuple(issues),
        "min": minv,
        "max": maxv,
    }


def perturb_category(
    category: DensityCategory,
    rng: np.random.Generator,
    accuracy: float,
) -> DensityCategory:
    """Simulate reader disagreement by perturbing a label to an adjacent category."""
    if rng.random() < accuracy:
        return category

    idx = CATEGORY_ORDER.index(category)
    neighbors: List[DensityCategory] = []
    if idx > 0:
        neighbors.append(CATEGORY_ORDER[idx - 1])
    if idx < len(CATEGORY_ORDER) - 1:
        neighbors.append(CATEGORY_ORDER[idx + 1])

    chosen = str(rng.choice(neighbors))
    return chosen  # type: ignore[return-value]


def make_probabilities(
    fraction: float,
    rng: np.random.Generator,
    temperature: float,
) -> Dict[DensityCategory, float]:
    """Convert a scalar fraction estimate into a probability vector."""
    centers = np.array([CATEGORY_CENTERS[c] for c in CATEGORY_ORDER], dtype=float)
    temp = max(float(temperature), 1e-3)
    logits = -np.abs(float(fraction) - centers) / temp + rng.normal(0.0, 0.05, size=len(CATEGORY_ORDER))
    probs = softmax(logits)
    return {c: float(p) for c, p in zip(CATEGORY_ORDER, probs)}


# -----------------------------------------------------------------------------
# DICOM-like metadata simulation and de-identification
# -----------------------------------------------------------------------------


def make_raw_dicom_like(
    case_id: str,
    patient_id: str,
    site: str,
    scanner: str,
    acquisition_date: str,
    laterality: str,
    view: str,
    consent_status: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Create synthetic raw DICOM-like metadata containing simulated PHI fields."""
    _ = rng
    return {
        "PatientName": f"DOE^JON^{case_id}",
        "PatientID": patient_id,
        "AccessionNumber": f"ACC{case_id}",
        "InstitutionName": site,
        "StationName": scanner,
        "OperatorName": f"OP{numeric_hash(case_id + view, 900) + 100}",
        "StudyDate": acquisition_date,
        "Laterality": laterality,
        "View": view,
        "SeriesDescription": f"Mammo {laterality} {view}",
        "PixelSpacing": "0.085\\0.085",
        "Rows": 384,
        "Columns": 384,
        "BitsAllocated": 16,
        "HighBit": 12,
        "PhotometricInterpretation": "MONOCHROME2",
        "SOPInstanceUID": f"1.2.826.{numeric_hash(case_id + ':' + view, 100000000)}",
        "ConsentStatus": consent_status,
    }


def deidentify_dicom_like(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a de-identified DICOM-like metadata dictionary."""
    return {
        "PatientID": "SYN-" + stable_hash(raw["PatientID"])[:12],
        "PatientName": "REMOVED",
        "AccessionNumber": "REMOVED",
        "InstitutionName": "REMOVED",
        "SiteCode": raw["InstitutionName"],
        "ScannerModel": raw["StationName"],
        "StationName": "REMOVED",
        "OperatorName": "REMOVED",
        "StudyDate": raw["StudyDate"],
        "Laterality": raw["Laterality"],
        "View": raw["View"],
        "SeriesDescription": raw["SeriesDescription"],
        "PixelSpacing": raw["PixelSpacing"],
        "Rows": raw["Rows"],
        "Columns": raw["Columns"],
        "BitsAllocated": raw["BitsAllocated"],
        "HighBit": raw["HighBit"],
        "PhotometricInterpretation": raw["PhotometricInterpretation"],
        "SOPInstanceUID": "1.2.826." + str(numeric_hash(raw["SOPInstanceUID"], 100000000)),
        "ConsentStatus": raw["ConsentStatus"],
        "DeidentificationMethod": "sha256-pseudonymization+tag-removal",
    }


def verify_deidentification(raw: Dict[str, Any], deid: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Verify that forbidden PHI tags are removed and required tags are retained."""
    _ = raw
    retained_phi = [
        tag
        for tag in FORBIDDEN_DEID_TAGS
        if tag in deid and str(deid[tag]).upper() not in {"REMOVED", "", "ANONYMIZED", "PSEUDONYMIZED"}
    ]
    missing_required = [tag for tag in REQUIRED_DEID_TAGS if tag not in deid]
    ok = len(retained_phi) == 0 and len(missing_required) == 0
    return ok, retained_phi, missing_required


def check_metadata_quality(
    raw: Dict[str, Any],
    deid: Dict[str, Any],
    deid_ok: bool,
    retained_phi: Sequence[str],
    missing_required: Sequence[str],
) -> Dict[str, Any]:
    """Check synthetic DICOM-like metadata for unit/orientation/deid issues."""
    issues: List[str] = []

    spacing_raw = str(raw.get("PixelSpacing", ""))
    try:
        parts = [float(x) for x in spacing_raw.replace("\\", "/").split("/") if x]
    except ValueError:
        parts = []
        issues.append("pixel_spacing_parse_error")

    if not parts:
        issues.append("pixel_spacing_missing")
    elif any(p <= 0 for p in parts):
        issues.append("pixel_spacing_non_positive")
    elif any(p > 1.0 for p in parts):
        issues.append("pixel_spacing_unit_error_suspected")

    if int(raw.get("Rows", 0)) < 128 or int(raw.get("Columns", 0)) < 128:
        issues.append("invalid_matrix_size")

    if str(raw.get("PhotometricInterpretation", "")) != "MONOCHROME2":
        issues.append("unexpected_photometric_interpretation")

    if int(raw.get("HighBit", 0)) > int(raw.get("BitsAllocated", 0)):
        issues.append("high_bit_exceeds_bits_allocated")

    if str(deid.get("Laterality", "")) not in {"L", "R"}:
        issues.append("invalid_laterality")

    if str(deid.get("View", "")) not in VIEWS:
        issues.append("invalid_view")

    if not deid_ok:
        issues.append("deidentification_failed")

    if retained_phi:
        issues.append("phi_retained:" + ",".join(retained_phi))

    if missing_required:
        issues.append("required_tags_missing:" + ",".join(missing_required))

    return {
        "passed": len(issues) == 0,
        "issues": tuple(issues),
    }


# -----------------------------------------------------------------------------
# Model output dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewOutput:
    """Per-view simulated model output."""

    view: str
    glandular_fraction: float
    mean_probabilities: Dict[DensityCategory, float]
    member_probabilities: Tuple[Dict[DensityCategory, float], ...]
    confidence: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    quality_pass: bool
    quality_issues: Tuple[str, ...]


@dataclass(frozen=True)
class EnsembleOutput:
    """Case-level simulated ensemble output."""

    case_id: str
    predicted_category: DensityCategory
    mean_probabilities: Dict[DensityCategory, float]
    confidence: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    total_uncertainty: float
    glandular_fraction: float
    quality_pass: bool
    quality_issues: Tuple[str, ...]
    view_outputs: Tuple[ViewOutput, ...]
    ood_score: float = 0.0
    drift_flag: bool = False


@dataclass
class DatasetBundle:
    """Container for synthetic dataset artifacts."""

    df: pd.DataFrame
    images: Dict[str, Dict[str, np.ndarray]]
    marked_images: Dict[str, Dict[str, np.ndarray]]
    marker_masks: Dict[str, Dict[str, np.ndarray]]
    raw_metadata: Dict[str, Dict[str, Dict[str, Any]]]
    deidentified_metadata: Dict[str, Dict[str, Dict[str, Any]]]
    outputs: Dict[str, EnsembleOutput]


# -----------------------------------------------------------------------------
# Dataset construction, splitting, and quality reporting
# -----------------------------------------------------------------------------


def assign_splits(patient_ids: Sequence[str], config: DemoConfig) -> List[str]:
    """Assign patient-level splits deterministically."""
    train_cut = config.split_train
    val_cut = config.split_train + config.split_validation
    splits: List[str] = []

    for pid in patient_ids:
        bucket = int(hashlib.sha256(pid.encode("utf-8")).hexdigest(), 16) % 10000
        bucket_f = bucket / 10000.0
        if bucket_f < train_cut:
            splits.append("train")
        elif bucket_f < val_cut:
            splits.append("validation")
        else:
            splits.append("test")

    return splits


def verify_split_integrity(df: pd.DataFrame) -> Tuple[bool, int]:
    """Verify that no active patient appears in more than one split."""
    if df.empty:
        return True, 0

    active = df[df["consent_status"] == "active"]
    if active.empty:
        return True, 0

    bad = int((active.groupby("patient_id")["split"].nunique() > 1).sum())
    return bad == 0, bad


def build_dataset(config: DemoConfig) -> DatasetBundle:
    """Build the full synthetic demo dataset."""
    rng = np.random.default_rng(config.seed)

    sites = ("SITE_A", "SITE_B", "SITE_C")
    scanners_by_site = {
        "SITE_A": ("Scanner_X_v1", "Scanner_X_v2"),
        "SITE_B": ("Scanner_Y_v1",),
        "SITE_C": ("Scanner_Z_v1", "Scanner_Z_v2"),
    }
    age_bands = ("40-49", "50-59", "60-69", "70+")

    rows: List[Dict[str, Any]] = []
    images: Dict[str, Dict[str, np.ndarray]] = {}
    marked_images: Dict[str, Dict[str, np.ndarray]] = {}
    marker_masks: Dict[str, Dict[str, np.ndarray]] = {}
    raw_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
    deidentified_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
    outputs: Dict[str, EnsembleOutput] = {}

    case_idx = 0
    patient_idx = 0
    dist = np.array(config.class_distribution, dtype=float)

    while case_idx < config.num_cases:
        patient_idx += 1
        patient_id = f"SYN-P{patient_idx:04d}"
        site = sites[int(rng.integers(0, len(sites)))]
        scanner_base = scanners_by_site[site][int(rng.integers(0, len(scanners_by_site[site])))]
        base_category = CATEGORY_ORDER[int(rng.choice(len(CATEGORY_ORDER), p=dist))]
        age_band = age_bands[int(rng.integers(0, len(age_bands)))]
        consent_status = "withdrawn" if rng.random() < config.consent_withdrawn_fraction else "active"
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(rng.integers(0, 365)))

        cases_for_patient = min(int(rng.integers(1, 4)), config.num_cases - case_idx)

        for visit in range(cases_for_patient):
            case_idx += 1
            case_id = f"SYN-C{case_idx:04d}"

            category = base_category
            if visit > 0 and rng.random() < 0.25:
                idx = CATEGORY_ORDER.index(category)
                if idx > 0 and rng.random() < 0.5:
                    category = CATEGORY_ORDER[idx - 1]
                elif idx < len(CATEGORY_ORDER) - 1:
                    category = CATEGORY_ORDER[idx + 1]

            scanner = scanner_base
            drift = bool(visit > 0 and rng.random() < config.drift_fraction)
            if drift:
                scanner = scanner_base + "_DRIFT"

            acquisition_date = (
                start_date + timedelta(days=visit * int(rng.integers(300, 450)))
            ).date().isoformat()

            laterality = "L" if rng.random() < 0.5 else "R"
            marker_present = bool(rng.random() < config.marker_fraction)

            case_images: Dict[str, np.ndarray] = {}
            case_marked: Dict[str, np.ndarray] = {}
            case_masks: Dict[str, np.ndarray] = {}
            case_raw: Dict[str, Dict[str, Any]] = {}
            case_deid: Dict[str, Dict[str, Any]] = {}

            view_outputs: List[ViewOutput] = []
            all_member_probs: List[Dict[DensityCategory, float]] = []
            fractions: List[float] = []
            quality_issues: List[str] = []

            for view in VIEWS:
                clean, marked, mask = generate_mammogram_pair(
                    rng=rng,
                    category=category,
                    site=site,
                    scanner=scanner,
                    view=view,
                    marker_present=marker_present,
                )

                case_images[view] = clean
                case_marked[view] = marked
                case_masks[view] = mask

                raw = make_raw_dicom_like(
                    case_id=case_id,
                    patient_id=patient_id,
                    site=site,
                    scanner=scanner,
                    acquisition_date=acquisition_date,
                    laterality=laterality,
                    view=view,
                    consent_status=consent_status,
                    rng=rng,
                )
                deid = deidentify_dicom_like(raw)
                deid_ok, retained_phi, missing_required = verify_deidentification(raw, deid)
                meta_quality = check_metadata_quality(raw, deid, deid_ok, retained_phi, missing_required)

                case_raw[view] = raw
                case_deid[view] = deid

                model_image = clean if config.remove_burned_in_annotations else marked
                img_quality = check_image_quality(model_image)

                fraction = estimate_glandular_fraction(model_image)
                marker_issue = bool(marker_present and not config.remove_burned_in_annotations)
                if marker_issue:
                    fraction = float(np.clip(fraction + 0.18, 0.0, 1.0))
                    quality_issues.append("burned_in_annotation_not_removed")

                member_probs: List[Dict[DensityCategory, float]] = []
                for m in range(config.num_ensemble_members):
                    mrng = np.random.default_rng(stable_seed(f"{config.seed}:{case_id}:{view}:{m}"))
                    noisy_fraction = float(np.clip(fraction + mrng.normal(0.0, config.ensemble_noise), 0.0, 1.0))
                    probs = make_probabilities(noisy_fraction, mrng, config.probability_temperature)
                    member_probs.append(probs)

                view_mean = average_probability_dicts(member_probs)
                view_pred = max(view_mean, key=lambda k: view_mean[k])
                view_conf = float(view_mean[view_pred])
                view_ale = normalized_entropy(view_mean)
                view_epis = float(np.std([p[view_pred] for p in member_probs])) if len(member_probs) > 1 else 0.0

                view_issues = list(img_quality["issues"]) + list(meta_quality["issues"])
                if marker_issue:
                    view_issues.append("burned_in_annotation_not_removed")

                view_quality_pass = bool(
                    img_quality["passed"]
                    and meta_quality["passed"]
                    and not marker_issue
                )

                if not view_quality_pass:
                    quality_issues.extend(view_issues)

                view_outputs.append(
                    ViewOutput(
                        view=view,
                        glandular_fraction=float(fraction),
                        mean_probabilities=view_mean,
                        member_probabilities=tuple(member_probs),
                        confidence=view_conf,
                        aleatoric_uncertainty=view_ale,
                        epistemic_uncertainty=view_epis,
                        quality_pass=view_quality_pass,
                        quality_issues=tuple(sorted(set(view_issues))),
                    )
                )

                all_member_probs.extend(member_probs)
                fractions.append(float(fraction))

            if consent_status == "withdrawn":
                quality_issues.append("consent_withdrawn")

            mean_probs = average_probability_dicts(all_member_probs)
            predicted = max(mean_probs, key=lambda k: mean_probs[k])
            confidence = float(mean_probs[predicted])
            aleatoric = normalized_entropy(mean_probs)
            epistemic = float(np.std([p[predicted] for p in all_member_probs])) if len(all_member_probs) > 1 else 0.0
            total_uncertainty = float(np.clip(math.sqrt(aleatoric**2 + epistemic**2), 0.0, 1.0))
            mean_fraction = float(np.mean(fractions)) if fractions else 0.0

            quality_pass = bool(all(vo.quality_pass for vo in view_outputs) and consent_status == "active")
            unique_issues = tuple(sorted(set(quality_issues)))

            out = EnsembleOutput(
                case_id=case_id,
                predicted_category=predicted,
                mean_probabilities=mean_probs,
                confidence=confidence,
                aleatoric_uncertainty=aleatoric,
                epistemic_uncertainty=epistemic,
                total_uncertainty=total_uncertainty,
                glandular_fraction=mean_fraction,
                quality_pass=quality_pass,
                quality_issues=unique_issues,
                view_outputs=tuple(view_outputs),
                ood_score=0.0,
                drift_flag=False,
            )

            reader_a = perturb_category(category, rng, config.reader_a_accuracy)
            reader_b = perturb_category(category, rng, config.reader_b_accuracy)

            cc_hash = hashlib.sha256(case_images["CC"].tobytes()).hexdigest()
            mlo_hash = hashlib.sha256(case_images["MLO"].tobytes()).hexdigest()
            study_hash = hashlib.sha256((cc_hash + mlo_hash).encode("utf-8")).hexdigest()

            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "site": site,
                    "scanner_model": scanner,
                    "scanner_drift": drift,
                    "acquisition_date": acquisition_date,
                    "age_band": age_band,
                    "laterality": laterality,
                    "views": "+".join(VIEWS),
                    "reader_a": reader_a,
                    "reader_b": reader_b,
                    "consensus_label": category,
                    "cc_sha256": cc_hash,
                    "mlo_sha256": mlo_hash,
                    "study_sha256": study_hash,
                    "marker_present": marker_present,
                    "marker_removed": bool(config.remove_burned_in_annotations),
                    "consent_status": consent_status,
                    "glandular_fraction": mean_fraction,
                    "predicted_category": predicted,
                    "confidence": confidence,
                    "aleatoric_uncertainty": aleatoric,
                    "epistemic_uncertainty": epistemic,
                    "total_uncertainty": total_uncertainty,
                    "quality_pass": quality_pass,
                    "quality_issues": "; ".join(unique_issues),
                    "ood_score": 0.0,
                    "drift_flag": False,
                }
            )

            images[case_id] = case_images
            marked_images[case_id] = case_marked
            marker_masks[case_id] = case_masks
            raw_metadata[case_id] = case_raw
            deidentified_metadata[case_id] = case_deid
            outputs[case_id] = out

            if case_idx >= config.num_cases:
                break

    df = pd.DataFrame(rows)
    df["split"] = assign_splits(df["patient_id"].tolist(), config)
    df.loc[df["consent_status"] == "withdrawn", "split"] = "excluded"

    # Compute OOD scores using train-split feature statistics.
    active_train = df[(df["split"] == "train") & (df["consent_status"] == "active")]
    if len(active_train) >= 2:
        feats = np.array(
            [
                [outputs[str(cid)].glandular_fraction, outputs[str(cid)].total_uncertainty]
                for cid in active_train["case_id"]
            ],
            dtype=float,
        )
        centroid = feats.mean(axis=0)
        scale = feats.std(axis=0) + 1e-3
    else:
        centroid = np.array([0.5, 0.3], dtype=float)
        scale = np.array([0.2, 0.2], dtype=float)

    for idx, row in df.iterrows():
        cid = str(row["case_id"])
        out = outputs[cid]
        feat = np.array([out.glandular_fraction, out.total_uncertainty], dtype=float)
        z = np.abs((feat - centroid) / scale)
        ood = float(
            np.clip(
                math.sqrt(float(np.mean(z * z))) / 3.0
                + (0.25 if bool(row["scanner_drift"]) else 0.0)
                + (0.15 if not bool(row["quality_pass"]) else 0.0),
                0.0,
                1.0,
            )
        )
        drift = bool(row["scanner_drift"] or ood > config.ood_threshold)

        outputs[cid] = replace(out, ood_score=ood, drift_flag=drift)
        df.at[idx, "ood_score"] = ood
        df.at[idx, "drift_flag"] = drift

    return DatasetBundle(
        df=df,
        images=images,
        marked_images=marked_images,
        marker_masks=marker_masks,
        raw_metadata=raw_metadata,
        deidentified_metadata=deidentified_metadata,
        outputs=outputs,
    )


def build_data_quality_report(
    df: pd.DataFrame,
    outputs: Dict[str, EnsembleOutput],
    config: DemoConfig,
    raw_metadata: Dict[str, Dict[str, Dict[str, Any]]],
    deidentified_metadata: Dict[str, Dict[str, Dict[str, Any]]],
) -> pd.DataFrame:
    """Build a data-quality report for the synthetic dataset."""
    checks: List[Tuple[str, bool, str]] = []

    checks.append(("dataset_non_empty", not df.empty, f"{len(df)} synthetic cases"))

    label_ok = bool(set(df["consensus_label"]).issubset(set(CATEGORY_ORDER))) if not df.empty else False
    checks.append(("label_schema", label_ok, "categories are a/b/c/d"))

    split_ok, bad_patients = verify_split_integrity(df)
    checks.append(
        (
            "patient_level_split_no_leak",
            split_ok,
            f"{bad_patients} active patients appear in multiple splits",
        )
    )

    dup = int(df["study_sha256"].duplicated().sum()) if not df.empty else 0
    checks.append(("no_duplicate_study_hashes", dup == 0, f"{dup} duplicate study hashes"))

    if df.empty:
        required_missing = 0
    else:
        required_cols = [
            "case_id",
            "patient_id",
            "site",
            "scanner_model",
            "acquisition_date",
            "age_band",
            "laterality",
            "consensus_label",
            "split",
            "consent_status",
        ]
        required_missing = int(df[required_cols].isna().any(axis=1).sum())
    checks.append(
        (
            "required_metadata_present",
            required_missing == 0,
            f"{required_missing} rows missing required metadata",
        )
    )

    active_df = df[df["consent_status"] == "active"] if not df.empty else df
    quality_fail = int((~active_df["quality_pass"]).sum()) if not active_df.empty else 0
    checks.append(
        (
            "active_case_quality_pass",
            quality_fail == 0,
            f"{quality_fail} active cases failed image/metadata checks",
        )
    )

    if not df.empty:
        counts = df["consensus_label"].value_counts()
        min_count = int(counts.min()) if len(counts) else 0
        max_count = int(counts.max()) if len(counts) else 0
        imbalance = float(max_count / max(1, min_count)) if min_count > 0 else float("inf")
    else:
        min_count = 0
        max_count = 0
        imbalance = 0.0
    checks.append(
        (
            "class_balance_informational",
            True,
            f"max_count={max_count}, min_count={min_count}, max/min ratio={imbalance:.2f}",
        )
    )

    prob_sum_ok = all(
        math.isclose(sum(out.mean_probabilities.values()), 1.0, abs_tol=1e-5)
        for out in outputs.values()
    )
    checks.append(("probabilities_sum_to_one", prob_sum_ok, "model probability vectors"))

    deid_fail = 0
    deid_details: List[str] = []
    for case_id, views in deidentified_metadata.items():
        for view, deid in views.items():
            raw = raw_metadata.get(case_id, {}).get(view, {})
            ok, retained, missing = verify_deidentification(raw, deid)
            if not ok:
                deid_fail += 1
                deid_details.append(f"{case_id}:{view}:retained={retained}:missing={missing}")
    checks.append(
        (
            "deidentification_preview_pass",
            deid_fail == 0,
            f"{deid_fail} de-identification preview failures; {'; '.join(deid_details[:3])}",
        )
    )

    withdrawn_in_splits = (
        int(((df["consent_status"] == "withdrawn") & (df["split"] != "excluded")).sum())
        if not df.empty
        else 0
    )
    checks.append(
        (
            "withdrawn_consent_excluded",
            withdrawn_in_splits == 0,
            f"{withdrawn_in_splits} withdrawn cases not excluded",
        )
    )

    marker_unremoved = (
        int(((df["marker_present"]) & (~df["marker_removed"])).sum())
        if not df.empty
        else 0
    )
    checks.append(
        (
            "burned_in_annotations_removed",
            marker_unremoved == 0,
            f"{marker_unremoved} marker cases not removed; config_remove={config.remove_burned_in_annotations}",
        )
    )

    timestamp_issue = 0
    if not df.empty:
        for _, g in df.groupby("patient_id"):
            g = g.sort_values("acquisition_date")
            dates = pd.to_datetime(g["acquisition_date"], errors="coerce")
            diffs = dates.diff().dt.days.dropna()
            if ((diffs < 0) | (diffs > 1500)).any():
                timestamp_issue += 1
    checks.append(
        (
            "longitudinal_timestamp_gaps_reasonable",
            timestamp_issue == 0,
            f"{timestamp_issue} patients with suspicious timestamp gaps",
        )
    )

    drift_cases = int(df["scanner_drift"].sum()) if not df.empty else 0
    checks.append(
        (
            "drift_coverage_informational",
            True,
            f"{drift_cases} drift-flagged synthetic cases",
        )
    )

    return pd.DataFrame(checks, columns=["check", "passed", "details"])


def condition_coverage(df: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Return counts by condition slice."""
    cols = list(group_columns)
    if df.empty:
        return pd.DataFrame(columns=cols + ["cases"])
    return df.groupby(cols, dropna=False).size().reset_index(name="cases")


def build_manifest(
    df: pd.DataFrame,
    outputs: Dict[str, EnsembleOutput],
    config: DemoConfig,
) -> pd.DataFrame:
    """Build a versioned manifest of the synthetic dataset."""
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        out = outputs[case_id]
        records.append(
            {
                "case_id": case_id,
                "patient_id": row["patient_id"],
                "split": row["split"],
                "consent_status": row["consent_status"],
                "site": row["site"],
                "scanner_model": row["scanner_model"],
                "scanner_drift": bool(row["scanner_drift"]),
                "acquisition_date": row["acquisition_date"],
                "age_band": row["age_band"],
                "laterality": row["laterality"],
                "views": row["views"],
                "reader_a": row["reader_a"],
                "reader_b": row["reader_b"],
                "consensus_label": row["consensus_label"],
                "cc_sha256": row["cc_sha256"],
                "mlo_sha256": row["mlo_sha256"],
                "study_sha256": row["study_sha256"],
                "marker_present": bool(row["marker_present"]),
                "marker_removed": bool(row["marker_removed"]),
                "model_predicted_category": out.predicted_category,
                "model_confidence": out.confidence,
                "aleatoric_uncertainty": out.aleatoric_uncertainty,
                "epistemic_uncertainty": out.epistemic_uncertainty,
                "total_uncertainty": out.total_uncertainty,
                "ood_score": out.ood_score,
                "drift_flag": out.drift_flag,
                "probabilities_json": json.dumps(to_jsonable(out.mean_probabilities), sort_keys=True),
                "model_version": config.model_version,
                "config_hash": config_hash(config),
            }
        )

    return pd.DataFrame(records)


def data_inventory(df: pd.DataFrame, config: DemoConfig) -> pd.DataFrame:
    """Return a version-controlled synthetic data inventory table."""
    if df.empty:
        date_min = date_max = "n/a"
    else:
        date_min = str(df["acquisition_date"].min())
        date_max = str(df["acquisition_date"].max())

    return pd.DataFrame(
        [
            {
                "field": "source",
                "value": "synthetic generator in app.py",
                "notes": "No real patient images or DICOM files are used.",
            },
            {
                "field": "owner",
                "value": "demo project owner",
                "notes": "Replace with actual data steward for real deployments.",
            },
            {
                "field": "license_or_consent_basis",
                "value": "synthetic-demo-use-only",
                "notes": "Not applicable to real clinical data; must be replaced by ethics/DUA terms.",
            },
            {
                "field": "size",
                "value": f"{len(df)} synthetic cases",
                "notes": f"Configured target: {config.num_cases}.",
            },
            {
                "field": "collection_dates",
                "value": f"{date_min} to {date_max}",
                "notes": "Simulated longitudinal acquisition dates.",
            },
            {
                "field": "device_or_sensor_settings",
                "value": "simulated sites/scanners/views/spacing",
                "notes": "Includes synthetic drift and marker conditions.",
            },
            {
                "field": "known_gaps",
                "value": "no real anatomy, no reader ground truth, no external validation",
                "notes": "Must be resolved before any clinical claim.",
            },
        ]
    )


def deletion_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Return simulated consent-withdrawal deletion queue."""
    if df.empty:
        return pd.DataFrame(columns=["case_id", "patient_id", "acquisition_date", "status"])
    q = df[df["consent_status"] == "withdrawn"].copy()
    q["status"] = "queued_for_deletion_dry_run_only"
    return q[["case_id", "patient_id", "acquisition_date", "status"]]


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def is_defer(
    output: EnsembleOutput,
    confidence_threshold: float,
    uncertainty_threshold: float,
    ood_threshold: float,
) -> bool:
    """Apply the deferral policy."""
    return bool(
        output.confidence < confidence_threshold
        or output.total_uncertainty > uncertainty_threshold
        or output.ood_score > ood_threshold
        or not output.quality_pass
    )


def compute_eval_df(
    df: pd.DataFrame,
    outputs: Dict[str, EnsembleOutput],
    confidence_threshold: float,
    uncertainty_threshold: float,
    ood_threshold: float,
) -> pd.DataFrame:
    """Compute case-level evaluation fields for current thresholds."""
    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        cid = str(row["case_id"])
        out = outputs[cid]
        deferred = is_defer(out, confidence_threshold, uncertainty_threshold, ood_threshold)
        correct = bool(out.predicted_category == row["consensus_label"])
        consensus_dense = bool(row["consensus_label"] in ("c", "d"))
        model_dense = bool(out.predicted_category in ("c", "d"))

        rows.append(
            {
                "case_id": cid,
                "patient_id": row["patient_id"],
                "split": row["split"],
                "site": row["site"],
                "scanner_model": row["scanner_model"],
                "scanner_drift": bool(row["scanner_drift"]),
                "age_band": row["age_band"],
                "laterality": row["laterality"],
                "views": row["views"],
                "consensus_label": row["consensus_label"],
                "predicted_category": out.predicted_category,
                "confidence": out.confidence,
                "aleatoric_uncertainty": out.aleatoric_uncertainty,
                "epistemic_uncertainty": out.epistemic_uncertainty,
                "total_uncertainty": out.total_uncertainty,
                "ood_score": out.ood_score,
                "drift_flag": out.drift_flag,
                "quality_pass": out.quality_pass,
                "deferred": deferred,
                "correct": correct,
                "consensus_dense": consensus_dense,
                "model_dense": model_dense,
                "consent_status": row["consent_status"],
            }
        )

    return pd.DataFrame(rows)


def threshold_sweep(
    eval_df: pd.DataFrame,
    uncertainty_threshold: float,
    ood_threshold: float,
    config: DemoConfig,
) -> pd.DataFrame:
    """Sweep confidence threshold and compute selective metrics."""
    if eval_df.empty:
        return pd.DataFrame(
            columns=[
                "confidence_threshold",
                "selective_accuracy",
                "deferral_rate",
                "dense_sensitivity",
                "non_dense_specificity",
            ]
        )

    rows: List[Dict[str, float]] = []
    confs = np.arange(0.50, 0.991, 0.01)

    conf_arr = eval_df["confidence"].to_numpy(dtype=float)
    unc_arr = eval_df["total_uncertainty"].to_numpy(dtype=float)
    ood_arr = eval_df["ood_score"].to_numpy(dtype=float)
    quality_arr = eval_df["quality_pass"].to_numpy(dtype=bool)
    correct_arr = eval_df["correct"].to_numpy(dtype=bool)
    consensus_dense_arr = eval_df["consensus_dense"].to_numpy(dtype=bool)
    model_dense_arr = eval_df["model_dense"].to_numpy(dtype=bool)

    for c in confs:
        deferred = (conf_arr < c) | (unc_arr > uncertainty_threshold) | (ood_arr > ood_threshold) | (~quality_arr)
        nondef = ~deferred

        sel_acc = float(np.mean(correct_arr[nondef])) if np.any(nondef) else float("nan")
        defer_rate = float(np.mean(deferred))

        if np.any(nondef):
            y = consensus_dense_arr[nondef]
            p = model_dense_arr[nondef]
            tp = int(np.sum(y & p))
            fn = int(np.sum(y & ~p))
            tn = int(np.sum(~y & ~p))
            fp = int(np.sum(~y & p))

            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        else:
            sens = float("nan")
            spec = float("nan")

        rows.append(
            {
                "confidence_threshold": float(c),
                "selective_accuracy": sel_acc,
                "deferral_rate": defer_rate,
                "dense_sensitivity": sens,
                "non_dense_specificity": spec,
            }
        )

    return pd.DataFrame(rows)


def choose_best_threshold(
    sweep_df: pd.DataFrame,
    config: DemoConfig,
) -> Optional[Dict[str, float]]:
    """Choose a threshold meeting targets if possible, else best selective accuracy."""
    if sweep_df.empty:
        return None

    feasible = sweep_df[
        (sweep_df["dense_sensitivity"] >= config.target_sensitivity_dense)
        & (sweep_df["non_dense_specificity"] >= config.target_specificity_non_dense)
        & (sweep_df["deferral_rate"] <= config.max_deferral_rate)
    ].copy()

    if not feasible.empty:
        feasible["score"] = feasible["selective_accuracy"].fillna(-1.0)
        best = feasible.sort_values("score", ascending=False).iloc[0]
        return {k: float(best[k]) for k in sweep_df.columns}

    fallback = sweep_df.copy()
    fallback["score"] = fallback["selective_accuracy"].fillna(-1.0)
    best = fallback.sort_values("score", ascending=False).iloc[0]
    return {k: float(best[k]) for k in sweep_df.columns}


def subgroup_summary(eval_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Summarize performance by a categorical dimension."""
    if eval_df.empty or dimension not in eval_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for name, g in eval_df.groupby(dimension, dropna=False):
        n = len(g)
        acc = float(g["correct"].mean()) if n else float("nan")
        defer = float(g["deferred"].mean()) if n else float("nan")
        unc = float(g["total_uncertainty"].mean()) if n else float("nan")
        sens, spec = dense_sensitivity_specificity(g)
        rows.append(
            {
                dimension: name,
                "n": n,
                "accuracy": acc,
                "deferral_rate": defer,
                "mean_total_uncertainty": unc,
                "dense_sensitivity": sens,
                "non_dense_specificity": spec,
                "small_sample_warning": n < 5,
            }
        )

    return pd.DataFrame(rows)


def simulate_reader_study(
    eval_df: pd.DataFrame,
    config: DemoConfig,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a reader study with and without decision support."""
    if eval_df.empty:
        empty = pd.DataFrame(columns=["metric", "without_tool", "with_tool", "delta"])
        return empty, pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for _, r in eval_df.iterrows():
        without_correct = bool(rng.random() < config.reader_baseline_accuracy)

        if r["deferred"]:
            with_correct = bool(rng.random() < config.reader_baseline_accuracy)
            with_time = float(max(5.0, config.baseline_reading_time_s + rng.normal(0.0, 2.0)))
        else:
            if r["correct"]:
                p = float(min(1.0, config.reader_baseline_accuracy + config.tool_assist_gain))
            else:
                p = float(max(0.0, config.reader_baseline_accuracy - config.automation_bias_penalty * r["confidence"]))
            with_correct = bool(rng.random() < p)
            with_time = float(
                max(
                    5.0,
                    config.baseline_reading_time_s
                    - config.tool_time_save_s
                    + r["total_uncertainty"] * 5.0
                    + rng.normal(0.0, 2.0),
                )
            )

        without_time = float(max(5.0, config.baseline_reading_time_s + rng.normal(0.0, 2.0)))

        rows.append(
            {
                "case_id": r["case_id"],
                "without_correct": without_correct,
                "with_correct": with_correct,
                "without_time_s": without_time,
                "with_time_s": with_time,
                "deferred": bool(r["deferred"]),
            }
        )

    case_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "without_tool": float(case_df["without_correct"].mean()),
                "with_tool": float(case_df["with_correct"].mean()),
                "delta": float(case_df["with_correct"].mean() - case_df["without_correct"].mean()),
            },
            {
                "metric": "mean_reading_time_s",
                "without_tool": float(case_df["without_time_s"].mean()),
                "with_tool": float(case_df["with_time_s"].mean()),
                "delta": float(case_df["with_time_s"].mean() - case_df["without_time_s"].mean()),
            },
        ]
    )

    return summary, case_df


# -----------------------------------------------------------------------------
# Structured report contract
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredReport:
    """Structured report output contract."""

    schema_version: str
    template: str
    case_id: str
    patient_id: str
    site: str
    scanner_model: str
    laterality: str
    views: str
    density_category: DensityCategory
    category_label: str
    confidence: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    total_uncertainty: float
    ood_score: float
    drift_flag: bool
    defer_to_human: bool
    recommendation: str
    disclaimer: str
    generated_utc: str
    model_version: str
    config_hash: str
    probability_vector: Dict[str, float]
    reader_a: str
    reader_b: str
    consensus_label: str
    evidence_refs: Tuple[str, ...]


def template_recommendation(
    template: str,
    category: DensityCategory,
    defer: bool,
    drift: bool,
) -> str:
    """Generate a template-specific recommendation string."""
    if defer:
        base = "Defer to radiologist review. This simulated output must not be used alone."
    elif category in ("c", "d"):
        base = (
            "Dense pattern suggested. Radiologist confirmation required; local protocol "
            "may consider supplemental screening."
        )
    else:
        base = "Non-dense pattern suggested. Radiologist confirmation required before reporting."

    if drift:
        base += " Scanner-drift/OOD flag present: re-validate before operational reliance."

    if template == "screening_risk_note":
        return "Risk-based screening note: " + base
    if template == "audit_summary":
        return "Audit summary: " + base
    return base


def render_report(
    row: Dict[str, Any],
    output: EnsembleOutput,
    eval_row: Dict[str, Any],
    confidence_threshold: float,
    uncertainty_threshold: float,
    ood_threshold: float,
    config: DemoConfig,
    template: str = "birads_like",
) -> StructuredReport:
    """Render a structured report from a case row, model output, and evaluation row."""
    _ = eval_row
    defer = is_defer(output, confidence_threshold, uncertainty_threshold, ood_threshold)

    return StructuredReport(
        schema_version=config.schema_version,
        template=template,
        case_id=str(row["case_id"]),
        patient_id=str(row["patient_id"]),
        site=str(row["site"]),
        scanner_model=str(row["scanner_model"]),
        laterality=str(row["laterality"]),
        views=str(row["views"]),
        density_category=output.predicted_category,
        category_label=CATEGORY_LABELS[output.predicted_category],
        confidence=float(output.confidence),
        aleatoric_uncertainty=float(output.aleatoric_uncertainty),
        epistemic_uncertainty=float(output.epistemic_uncertainty),
        total_uncertainty=float(output.total_uncertainty),
        ood_score=float(output.ood_score),
        drift_flag=bool(output.drift_flag),
        defer_to_human=defer,
        recommendation=template_recommendation(template, output.predicted_category, defer, output.drift_flag),
        disclaimer=(
            "Synthetic demo output only. Not for clinical use. "
            "Clinician remains responsible for the final decision."
        ),
        generated_utc=utc_now_iso(),
        model_version=config.model_version,
        config_hash=config_hash(config),
        probability_vector={k: float(v) for k, v in output.mean_probabilities.items()},
        reader_a=str(row["reader_a"]),
        reader_b=str(row["reader_b"]),
        consensus_label=str(row["consensus_label"]),
        evidence_refs=(
            f"synthetic://{row['case_id']}/CC",
            f"synthetic://{row['case_id']}/MLO",
        ),
    )


def validate_report(report: StructuredReport) -> Tuple[bool, List[str]]:
    """Validate the structured report contract."""
    errors: List[str] = []

    if not report.schema_version:
        errors.append("schema_version is required")
    if report.template not in REPORT_TEMPLATES:
        errors.append("template must be one of the allowed report templates")
    if not report.case_id:
        errors.append("case_id is required")
    if report.density_category not in CATEGORY_ORDER:
        errors.append("density_category must be one of a/b/c/d")
    if not (0.0 <= report.confidence <= 1.0):
        errors.append("confidence must be in [0, 1]")
    if not (0.0 <= report.total_uncertainty <= 1.0):
        errors.append("total_uncertainty must be in [0, 1]")
    if not (0.0 <= report.ood_score <= 1.0):
        errors.append("ood_score must be in [0, 1]")
    if not report.disclaimer:
        errors.append("disclaimer is required")
    if not report.evidence_refs:
        errors.append("evidence_refs is required")

    try:
        datetime.fromisoformat(report.generated_utc)
    except ValueError:
        errors.append("generated_utc must be ISO-8601")

    return len(errors) == 0, errors


def report_to_markdown(report: StructuredReport) -> str:
    """Convert a structured report to markdown."""
    prob_lines = "\n".join(
        f"- `{k}`: {v:.3f}" for k, v in report.probability_vector.items()
    )
    evidence_lines = "\n".join(f"- `{ref}`" for ref in report.evidence_refs)
    return f"""### Structured density report (simulated)

- **Template**: `{report.template}`
- **Schema**: `{report.schema_version}`
- **Case**: `{report.case_id}`
- **Patient**: `{report.patient_id}`
- **Site / Scanner**: `{report.site}` / `{report.scanner_model}`
- **Acquisition**: `{report.laterality}` `{report.views}`
- **Predicted category**: `{report.density_category.upper()}` — {report.category_label}
- **Confidence**: `{report.confidence:.2%}`
- **Aleatoric uncertainty**: `{report.aleatoric_uncertainty:.2%}`
- **Epistemic uncertainty**: `{report.epistemic_uncertainty:.2%}`
- **Total uncertainty**: `{report.total_uncertainty:.2%}`
- **OOD score**: `{report.ood_score:.2%}`
- **Drift flag**: `{report.drift_flag}`
- **Defer to human**: `{report.defer_to_human}`

#### Recommendation
{report.recommendation}

#### Probability vector
{prob_lines}

#### Reader context
- **Reader A**: `{report.reader_a}`
- **Reader B**: `{report.reader_b}`
- **Consensus label**: `{report.consensus_label}`

#### Evidence references
{evidence_lines}

#### Disclaimer
{report.disclaimer}

#### Provenance
- **Model version**: `{report.model_version}`
- **Config hash**: `{report.config_hash}`
- **Generated UTC**: `{report.generated_utc}`
"""


# -----------------------------------------------------------------------------
# Audit log with hash chain
# -----------------------------------------------------------------------------


def _last_audit_hash(path: Path) -> str:
    """Return last event hash or genesis."""
    if not path.exists():
        return "0" * 64
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            return str(obj.get("event_hash", "0" * 64))
        except json.JSONDecodeError:
            continue
    return "0" * 64


def append_audit_event(config: DemoConfig, event: Dict[str, Any]) -> None:
    """Append a hash-chained audit event to the local audit log."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prev = _last_audit_hash(config.audit_log_path)
    payload = {
        "timestamp_utc": utc_now_iso(),
        "config_hash": config_hash(config),
        "previous_hash": prev,
        **to_jsonable(event),
    }
    event_hash = stable_hash({k: v for k, v in payload.items() if k != "event_hash"})
    payload["event_hash"] = event_hash

    with config.audit_log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def read_audit_events(path: Path, limit: int = 300) -> List[Dict[str, Any]]:
    """Read the most recent audit events from disk."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    events: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(events))


def verify_audit_chain(path: Path) -> Tuple[bool, List[str], int]:
    """Verify audit hash chain integrity."""
    if not path.exists():
        return True, [], 0

    prev = "0" * 64
    issues: List[str] = []
    count = 0

    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                issues.append(f"line {i}: invalid JSON")
                continue

            count += 1
            if obj.get("previous_hash") != prev:
                issues.append(f"line {i}: previous_hash mismatch")

            stored_hash = obj.get("event_hash")
            calc_hash = stable_hash({k: v for k, v in obj.items() if k != "event_hash"})
            if stored_hash != calc_hash:
                issues.append(f"line {i}: event_hash mismatch")

            prev = str(stored_hash)

    return len(issues) == 0, issues, count


def audit_retention_preview(path: Path, retention_days: int) -> Dict[str, Any]:
    """Preview events older than retention window. Dry-run only."""
    events = read_audit_events(path, limit=10000)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    old: List[Dict[str, Any]] = []

    for e in events:
        try:
            ts = datetime.fromisoformat(str(e.get("timestamp_utc")))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                old.append(e)
        except Exception:
            continue

    return {
        "retention_days": retention_days,
        "cutoff_utc": cutoff.isoformat(),
        "total_events": len(events),
        "events_older_than_retention": len(old),
        "sample_old_events": old[:5],
        "actual_deletion_performed": False,
    }


# -----------------------------------------------------------------------------
# Media generation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaBundle:
    """Generated demo media."""

    mp4_bytes: Optional[bytes]
    gif_bytes: Optional[bytes]
    poster_bytes: Optional[bytes]
    display_kind: Literal["mp4", "gif", "poster"]
    codec: Optional[str]


def _draw_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    """Draw a simple horizontal bar on an OpenCV canvas."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 90), 1)
    fill = int(max(0.0, min(1.0, value)) * w)
    cv2.rectangle(canvas, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(
        canvas,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def render_frame(
    row: Dict[str, Any],
    image: np.ndarray,
    output: EnsembleOutput,
    deferred: bool,
    frame_idx: int,
    total_frames: int,
    config: DemoConfig,
) -> np.ndarray:
    """Render one demo frame as a BGR uint8 image."""
    W, H = 640, 360
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    cv2.putText(
        canvas,
        "SYNTHETIC Mammography Density Demo - decision support only",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    size = 240
    img_u8 = _float_to_uint8(image)
    img_resized = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA)
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

    small_float = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    heat_threshold = config.heat_thresholds[CATEGORY_ORDER.index(output.predicted_category)]

    heat = np.zeros((size, size, 3), dtype=np.uint8)
    heat[small_float > heat_threshold] = (0, 0, 210)
    blended = cv2.addWeighted(img_bgr, 0.78, heat, 0.22, 0)

    canvas[50:50 + size, 15:15 + size] = blended
    cv2.rectangle(canvas, (15, 50), (15 + size, 50 + size), (200, 200, 200), 1)

    x = 280
    info_lines = [
        f"Case {row['case_id']}",
        f"Site {row['site']}",
        f"Scanner {row['scanner_model']}",
        f"Views {row['views']}",
        f"Age band {row['age_band']}",
        f"Reader A/B {row['reader_a']}/{row['reader_b']}",
        f"Consensus {row['consensus_label']}",
    ]

    for i, text in enumerate(info_lines):
        cv2.putText(
            canvas,
            text,
            (x, 70 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"Predicted: {output.predicted_category.upper()}",
        (x, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    _draw_bar(canvas, x, 200, 180, 10, output.confidence, "Confidence", (80, 200, 80))
    _draw_bar(canvas, x, 230, 180, 10, output.total_uncertainty, "Total unc.", (0, 165, 255))
    _draw_bar(canvas, x, 260, 180, 10, output.ood_score, "OOD score", (180, 105, 255))

    status = "DEFER TO RADIOLOGIST" if deferred else "BELOW DEFER THRESHOLDS"
    status_color = (0, 0, 255) if deferred else (0, 200, 0)

    cv2.putText(
        canvas,
        status,
        (x, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        status_color,
        1,
        cv2.LINE_AA,
    )

    if output.drift_flag:
        cv2.putText(
            canvas,
            "DRIFT/OOD FLAG",
            (x, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    scan_y = 50 + ((frame_idx * 9) % size)
    cv2.line(canvas, (15, scan_y), (15 + size, scan_y), (0, 255, 255), 1)

    cv2.putText(
        canvas,
        f"frame {frame_idx + 1}/{total_frames}",
        (15, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "SYNTHETIC / NOT FOR CLINICAL USE",
        (330, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )

    return canvas


def generate_media_bundle(
    df: pd.DataFrame,
    images: Dict[str, Dict[str, np.ndarray]],
    outputs: Dict[str, EnsembleOutput],
    confidence_threshold: float,
    uncertainty_threshold: float,
    ood_threshold: float,
    config: DemoConfig,
) -> MediaBundle:
    """Generate an MP4 when possible, with GIF fallback for reliable playback."""
    if df.empty:
        return MediaBundle(None, None, None, "poster", None)

    active = df[df["consent_status"] == "active"]
    selected = active.head(min(6, len(active)))
    if selected.empty:
        selected = df.head(min(6, len(df)))

    frame_count = max(2, int(config.video_fps * config.video_seconds))

    frames: List[np.ndarray] = []
    poster_bytes: Optional[bytes] = None

    for f in range(frame_count):
        row = selected.iloc[f % len(selected)]
        case_id = str(row["case_id"])
        out = outputs[case_id]
        deferred = is_defer(out, confidence_threshold, uncertainty_threshold, ood_threshold)
        frame = render_frame(
            row.to_dict(),
            images[case_id]["CC"],
            out,
            deferred,
            f,
            frame_count,
            config,
        )
        if f == 0:
            ok, encoded = cv2.imencode(".png", frame)
            poster_bytes = encoded.tobytes() if ok else None
        frames.append(frame)

    mp4_bytes: Optional[bytes] = None
    codec_used: Optional[str] = None

    for codec in ("avc1", "H264", "mp4v"):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()

        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*codec),
            float(config.video_fps),
            (640, 360),
        )

        if writer.isOpened():
            for fr in frames:
                writer.write(fr)
            writer.release()

            path = Path(tmp.name)
            if path.exists() and path.stat().st_size > 0:
                mp4_bytes = path.read_bytes()
                codec_used = codec

            path.unlink(missing_ok=True)

            if mp4_bytes is not None:
                break
        else:
            Path(tmp.name).unlink(missing_ok=True)

    gif_bytes: Optional[bytes] = None
    display_kind: Literal["mp4", "gif", "poster"] = "poster"

    if mp4_bytes is not None and codec_used in ("avc1", "H264"):
        display_kind = "mp4"
    else:
        target_gif_frames = 12
        step = max(1, len(frames) // target_gif_frames)
        gif_pil: List[Image.Image] = []

        for fr in frames[::step]:
            small = cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA)
            gif_pil.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

        if gif_pil:
            buf = io.BytesIO()
            duration = int(1000 * step / max(1, config.video_fps))
            gif_pil[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=gif_pil[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )
            gif_bytes = buf.getvalue()
            display_kind = "gif"

    return MediaBundle(mp4_bytes, gif_bytes, poster_bytes, display_kind, codec_used)


def _b64(data: bytes) -> str:
    """Base64-encode bytes to ASCII string."""
    return base64.b64encode(data).decode("ascii")


def show_media(media: MediaBundle, can_export: bool = True) -> None:
    """Display generated media and provide download buttons."""
    if media.display_kind == "mp4" and media.mp4_bytes:
        video_b64 = _b64(media.mp4_bytes)
        poster_attr = ""
        if media.poster_bytes:
            poster_attr = f' poster="data:image/png;base64,{_b64(media.poster_bytes)}"'

        video_tag = (
            '<video autoplay muted loop controls width="100%" '
            f'style="border-radius:8px;background:#000;"{poster_attr}>'
            f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">'
            "Your browser does not support embedded video.</video>"
        )
        st.markdown(video_tag, unsafe_allow_html=True)

    elif media.gif_bytes:
        img_tag = (
            '<img src="data:image/gif;base64,'
            + _b64(media.gif_bytes)
            + '" alt="Synthetic demo animation" style="width:100%;border-radius:8px;">'
        )
        st.markdown(img_tag, unsafe_allow_html=True)

    elif media.poster_bytes:
        st.image(media.poster_bytes, use_container_width=True)

    else:
        st.info("No media was generated.")

    c1, c2, c3 = st.columns(3)

    if media.mp4_bytes:
        c1.download_button(
            "Download MP4",
            media.mp4_bytes,
            "breast_density_demo.mp4",
            "video/mp4",
            key="download_mp4",
            disabled=not can_export,
        )

    if media.gif_bytes:
        c2.download_button(
            "Download GIF",
            media.gif_bytes,
            "breast_density_demo.gif",
            "image/gif",
            key="download_gif",
            disabled=not can_export,
        )

    if media.poster_bytes:
        c3.download_button(
            "Download poster",
            media.poster_bytes,
            "demo_poster.png",
            "image/png",
            key="download_poster",
            disabled=not can_export,
        )


def create_evidence_image(
    image: np.ndarray,
    output: EnsembleOutput,
    config: DemoConfig,
    marked_image: Optional[np.ndarray] = None,
    show_raw_marker: bool = False,
) -> bytes:
    """Create a PNG evidence image with a simple density heatmap overlay."""
    use_image = marked_image if (show_raw_marker and marked_image is not None) else image
    img_u8 = _float_to_uint8(use_image)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    threshold = config.heat_thresholds[CATEGORY_ORDER.index(output.predicted_category)]
    heat = np.zeros_like(img_rgb, dtype=np.uint8)
    heat[use_image > threshold] = (255, 80, 0)

    blended = cv2.addWeighted(img_rgb, 0.75, heat, 0.25, 0)

    if show_raw_marker and marked_image is not None:
        cv2.putText(
            blended,
            "RAW MARKER VIEW",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    if not ok:
        raise RuntimeError("Failed to encode evidence image.")

    return encoded.tobytes()


# -----------------------------------------------------------------------------
# Model card, risk register, and test suite
# -----------------------------------------------------------------------------


def model_card_markdown(
    config: DemoConfig,
    eval_df: pd.DataFrame,
    outputs: Dict[str, EnsembleOutput],
    confidence_threshold: float,
    uncertainty_threshold: float,
    ood_threshold: float,
) -> str:
    """Return a concise model card for the simulated baseline."""
    if eval_df.empty:
        accuracy_text = "n/a"
        defer_text = "n/a"
        auc_text = "n/a"
        ece_text = "n/a"
    else:
        accuracy_text = f"{float(eval_df['correct'].mean()):.1%}"
        defer_text = f"{float(eval_df['deferred'].mean()):.1%}"
        auc = multiclass_auroc(eval_df, outputs)
        auc_text = "n/a" if math.isnan(auc) else f"{auc:.3f}"
        ece, _ = compute_ece_for_cases(eval_df, outputs, temperature=1.0)
        ece_text = "n/a" if math.isnan(ece) else f"{ece:.3f}"

    return f"""
### Model card (simulated baseline)

- **System name**: Breast Density Assessment from Mammography
- **Version**: `{config.model_version}`
- **Intended use (demo)**: decision-support prototype for breast density category suggestion with uncertainty-based deferral
- **Data**: synthetic, simulated, de-identified demo data only
- **Training**: none; deterministic simulated ensemble stands in for a future PyTorch CNN
- **Demo accuracy**: {accuracy_text}
- **Demo macro AUROC**: {auc_text}
- **Demo ECE**: {ece_text}
- **Demo deferral rate**: {defer_text}
- **Known limitations**:
  - not trained on real mammograms
  - no clinical validation
  - no external test set
  - no regulator-reviewed performance claims
- **Next step**: replace simulated scorer with a trained CNN only after ethics-approved data, labeling protocol, and validation plan are in place
- **Provenance**:
  - config hash: `{config_hash(config)}`
  - schema version: `{config.schema_version}`
  - confidence threshold: `{confidence_threshold:.2f}`
  - uncertainty threshold: `{uncertainty_threshold:.2f}`
  - OOD threshold: `{ood_threshold:.2f}`
"""


def risk_register() -> pd.DataFrame:
    """Return a starter risk register for the demo system."""
    return pd.DataFrame(
        [
            {
                "risk": "Use on real patients without validation",
                "category": "patient safety",
                "likelihood": "high if deployed prematurely",
                "impact": "severe",
                "mitigation": "restrict to synthetic demo; require ethics approval and clinical validation",
                "owner": "project owner / clinical lead",
            },
            {
                "risk": "Bias across sites, scanners, age bands, or density classes",
                "category": "fairness",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "subgroup reporting, external validation, drift monitoring",
                "owner": "ML engineer / clinical lead",
            },
            {
                "risk": "Automation bias from confident-looking outputs",
                "category": "human factors",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "uncertainty display, deferral policy, explicit clinician responsibility",
                "owner": "clinical informatics",
            },
            {
                "risk": "Privacy breach from DICOM metadata",
                "category": "privacy",
                "likelihood": "medium",
                "impact": "severe",
                "mitigation": "de-identification pipeline, independent audit sample, access controls",
                "owner": "data protection officer",
            },
            {
                "risk": "Regulatory non-compliance if marketed as medical device",
                "category": "regulatory",
                "likelihood": "high if unmanaged",
                "impact": "severe",
                "mitigation": "early QMS, ISO 14971 risk management, clinical evaluation plan",
                "owner": "regulatory affairs",
            },
        ]
    )


def run_test_suite() -> pd.DataFrame:
    """Run in-app unit and integration tests without Streamlit runtime calls."""
    results: List[Dict[str, Any]] = []

    def add(name: str, fn: Any) -> None:
        try:
            ok, details = fn()
        except Exception as exc:
            ok, details = False, repr(exc)
        results.append({"test": name, "passed": bool(ok), "details": str(details)})

    def test_translations() -> Tuple[bool, str]:
        missing = [k for k in TRANSLATIONS["en"] if k not in TRANSLATIONS["es"]]
        return len(missing) == 0, f"missing_es_keys={missing}"

    def test_softmax() -> Tuple[bool, str]:
        p = softmax(np.array([1.0, 2.0, 3.0, 4.0]))
        return math.isclose(float(np.sum(p)), 1.0, abs_tol=1e-6), f"sum={float(np.sum(p))}"

    def test_entropy() -> Tuple[bool, str]:
        e_uniform = normalized_entropy({c: 0.25 for c in CATEGORY_ORDER})
        e_onehot = normalized_entropy({"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0})
        return (
            math.isclose(e_uniform, 1.0, abs_tol=1e-6) and math.isclose(e_onehot, 0.0, abs_tol=1e-6),
            f"uniform={e_uniform}, onehot={e_onehot}",
        )

    def test_kappa() -> Tuple[bool, str]:
        k1 = cohen_kappa(["a", "b", "c"], ["a", "b", "c"])
        k2 = cohen_kappa(["a", "a", "a"], ["b", "b", "b"])
        return math.isclose(k1, 1.0, abs_tol=1e-6) and k2 < 0.1, f"perfect={k1}, discordant={k2}"

    def test_weighted_kappa() -> Tuple[bool, str]:
        k_exact = weighted_kappa_ordinal(["a", "b", "c", "d"], ["a", "b", "c", "d"])
        k_adj = weighted_kappa_ordinal(["a", "b", "c", "d"], ["b", "a", "d", "c"])
        return k_exact > 0.99 and k_adj < k_exact, f"exact={k_exact}, adjacent={k_adj}"

    def test_auroc() -> Tuple[bool, str]:
        auc_perfect = binary_auroc(np.array([0.1, 0.9]), np.array([0, 1], dtype=bool))
        auc_inverse = binary_auroc(np.array([0.9, 0.1]), np.array([0, 1], dtype=bool))
        return (
            math.isclose(auc_perfect, 1.0, abs_tol=1e-6)
            and math.isclose(auc_inverse, 0.0, abs_tol=1e-6),
            f"perfect={auc_perfect}, inverse={auc_inverse}",
        )

    def test_numeric_hash() -> Tuple[bool, str]:
        h1 = numeric_hash("abc", 100000000)
        h2 = numeric_hash("abc", 100000000)
        h3 = numeric_hash("abd", 100000000)
        return isinstance(h1, int) and h1 == h2 and h1 != h3, f"h1={h1}, h3={h3}"

    def test_config_load() -> Tuple[bool, str]:
        cfg = load_config()
        return (
            cfg.num_cases > 0
            and len(cfg.class_distribution) == 4
            and len(cfg.heat_thresholds) == 4,
            f"num_cases={cfg.num_cases}",
        )

    def test_dataset_build() -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(
                load_config(),
                num_cases=8,
                num_ensemble_members=2,
                consent_withdrawn_fraction=0.0,
                marker_fraction=0.0,
                drift_fraction=0.0,
                output_dir=Path(td),
                audit_log_path=Path(td) / "audit.jsonl",
            )
            bundle = build_dataset(cfg)
            ok = (
                not bundle.df.empty
                and len(bundle.df) == cfg.num_cases
                and all(cid in bundle.outputs for cid in bundle.df["case_id"])
            )
            return ok, f"rows={len(bundle.df)}"

    def test_split_integrity() -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(
                load_config(),
                num_cases=10,
                num_ensemble_members=2,
                consent_withdrawn_fraction=0.0,
                output_dir=Path(td),
                audit_log_path=Path(td) / "audit.jsonl",
            )
            bundle = build_dataset(cfg)
            ok, bad = verify_split_integrity(bundle.df)
            return ok, f"bad_patients={bad}"

    def test_deidentification() -> Tuple[bool, str]:
        raw = make_raw_dicom_like(
            case_id="X",
            patient_id="P",
            site="S",
            scanner="M",
            acquisition_date="2025-01-01",
            laterality="L",
            view="CC",
            consent_status="active",
            rng=np.random.default_rng(0),
        )
        deid = deidentify_dicom_like(raw)
        ok, retained, missing = verify_deidentification(raw, deid)
        return ok and not retained and not missing, f"retained={retained}, missing={missing}"

    def test_data_quality_report() -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(
                load_config(),
                num_cases=8,
                num_ensemble_members=2,
                consent_withdrawn_fraction=0.0,
                marker_fraction=0.0,
                drift_fraction=0.0,
                output_dir=Path(td),
                audit_log_path=Path(td) / "audit.jsonl",
            )
            bundle = build_dataset(cfg)
            report = build_data_quality_report(
                bundle.df,
                bundle.outputs,
                cfg,
                bundle.raw_metadata,
                bundle.deidentified_metadata,
            )
            required = {
                "dataset_non_empty",
                "patient_level_split_no_leak",
                "deidentification_preview_pass",
                "active_case_quality_pass",
            }
            ok = (not report.empty) and required.issubset(set(report["check"]))
            return ok, f"rows={len(report)}"

    def test_report_contract() -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(
                load_config(),
                num_cases=2,
                num_ensemble_members=2,
                consent_withdrawn_fraction=0.0,
                output_dir=Path(td),
                audit_log_path=Path(td) / "audit.jsonl",
            )
            bundle = build_dataset(cfg)
            row = bundle.df.iloc[0].to_dict()
            out = bundle.outputs[str(row["case_id"])]
            ev = compute_eval_df(
                bundle.df,
                bundle.outputs,
                cfg.confidence_threshold,
                cfg.uncertainty_threshold,
                cfg.ood_threshold,
            )
            ev_row = ev.iloc[0].to_dict()
            rep = render_report(
                row,
                out,
                ev_row,
                cfg.confidence_threshold,
                cfg.uncertainty_threshold,
                cfg.ood_threshold,
                cfg,
            )
            ok, errors = validate_report(rep)
            return ok, "; ".join(errors) if errors else "ok"

    def test_audit_chain() -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(
                load_config(),
                output_dir=Path(td),
                audit_log_path=Path(td) / "audit.jsonl",
            )
            append_audit_event(cfg, {"action": "test1"})
            append_audit_event(cfg, {"action": "test2"})
            ok, issues, count = verify_audit_chain(cfg.audit_log_path)
            return ok and count == 2, f"count={count}, issues={issues}"

    def test_media_generation() -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(
                load_config(),
                num_cases=2,
                num_ensemble_members=2,
                video_fps=2,
                video_seconds=0.5,
                consent_withdrawn_fraction=0.0,
                output_dir=Path(td),
                audit_log_path=Path(td) / "audit.jsonl",
            )
            bundle = build_dataset(cfg)
            media = generate_media_bundle(
                bundle.df,
                bundle.images,
                bundle.outputs,
                cfg.confidence_threshold,
                cfg.uncertainty_threshold,
                cfg.ood_threshold,
                cfg,
            )
            ok = (media.mp4_bytes is not None and len(media.mp4_bytes) > 0) or (
                media.gif_bytes is not None and len(media.gif_bytes) > 0
            )
            return ok, f"display={media.display_kind}, codec={media.codec}"

    def test_threshold_sweep() -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(
                load_config(),
                num_cases=8,
                num_ensemble_members=2,
                consent_withdrawn_fraction=0.0,
                output_dir=Path(td),
                audit_log_path=Path(td) / "audit.jsonl",
            )
            bundle = build_dataset(cfg)
            ev = compute_eval_df(
                bundle.df,
                bundle.outputs,
                cfg.confidence_threshold,
                cfg.uncertainty_threshold,
                cfg.ood_threshold,
            )
            sweep = threshold_sweep(ev, cfg.uncertainty_threshold, cfg.ood_threshold, cfg)
            best = choose_best_threshold(sweep, cfg)
            return (not sweep.empty) and (best is not None), f"sweep_rows={len(sweep)}"

    add("translations", test_translations)
    add("softmax", test_softmax)
    add("normalized_entropy", test_entropy)
    add("cohen_kappa", test_kappa)
    add("weighted_kappa_ordinal", test_weighted_kappa)
    add("binary_auroc", test_auroc)
    add("numeric_hash", test_numeric_hash)
    add("config_load", test_config_load)
    add("dataset_build", test_dataset_build)
    add("split_integrity", test_split_integrity)
    add("deidentification", test_deidentification)
    add("data_quality_report", test_data_quality_report)
    add("report_contract", test_report_contract)
    add("audit_chain", test_audit_chain)
    add("media_generation", test_media_generation)
    add("threshold_sweep", test_threshold_sweep)

    return pd.DataFrame(results)


def run_cli_self_test() -> int:
    """Run test suite from CLI when BDA_SELF_TEST=1."""
    results = run_test_suite()
    print(results.to_string(index=False))
    return 0 if bool(results["passed"].all()) else 1


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Breast Density Assessment Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "config" not in st.session_state:
        st.session_state.config = load_config()

    config: DemoConfig = st.session_state.config

    if config.demo_mode != "synthetic":
        st.error("This build only supports synthetic demo mode. Set BDA_DEMO_MODE=synthetic.")
        st.stop()

    if "language" not in st.session_state:
        st.session_state.language = config.language
    lang = st.session_state.language

    if "role" not in st.session_state:
        st.session_state.role = "radiologist"

    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = config.confidence_threshold
    if "uncertainty_threshold" not in st.session_state:
        st.session_state.uncertainty_threshold = config.uncertainty_threshold
    if "ood_threshold" not in st.session_state:
        st.session_state.ood_threshold = config.ood_threshold

    if "dataset" not in st.session_state:
        with st.spinner("Loading synthetic demo data..."):
            bundle = build_dataset(config)
            st.session_state.dataset = bundle

    bundle: DatasetBundle = st.session_state.dataset
    df = bundle.df

    if df.empty:
        st.error("Dataset is empty. Check BDA_NUM_CASES.")
        st.stop()

    if "media" not in st.session_state:
        with st.spinner("Generating live demo media..."):
            st.session_state.media = generate_media_bundle(
                df,
                bundle.images,
                bundle.outputs,
                float(st.session_state.confidence_threshold),
                float(st.session_state.uncertainty_threshold),
                float(st.session_state.ood_threshold),
                config,
            )

    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "confirmations" not in st.session_state:
        st.session_state.confirmations = {}
    if "reader_study" not in st.session_state:
        st.session_state.reader_study = None
    if "test_results" not in st.session_state:
        st.session_state.test_results = None

    media: MediaBundle = st.session_state.media

    with st.sidebar:
        st.title(tr("app_title", lang))
        st.caption(tr("subtitle", lang))

        language = st.selectbox(
            tr("language", lang),
            options=list(TRANSLATIONS.keys()),
            index=list(TRANSLATIONS.keys()).index(lang) if lang in TRANSLATIONS else 0,
            key="language_select",
        )
        if language != st.session_state.language:
            st.session_state.language = language
            rerun_app()

        role = st.selectbox(
            tr("role", lang),
            options=list(ROLE_PERMISSIONS.keys()),
            index=list(ROLE_PERMISSIONS.keys()).index(st.session_state.role),
            key="role_select",
        )
        st.session_state.role = role

        edit_allowed = can(role, "edit_thresholds")

        conf_thr = st.slider(
            tr("confidence_threshold", lang),
            min_value=0.50,
            max_value=0.99,
            value=float(st.session_state.confidence_threshold),
            step=0.01,
            disabled=not edit_allowed,
            key="conf_slider",
        )

        unc_thr = st.slider(
            tr("uncertainty_threshold", lang),
            min_value=0.05,
            max_value=0.80,
            value=float(st.session_state.uncertainty_threshold),
            step=0.01,
            disabled=not edit_allowed,
            key="unc_slider",
        )

        ood_thr = st.slider(
            tr("ood_threshold", lang),
            min_value=0.30,
            max_value=1.00,
            value=float(st.session_state.ood_threshold),
            step=0.01,
            disabled=not edit_allowed,
            key="ood_slider",
        )

        st.session_state.confidence_threshold = conf_thr
        st.session_state.uncertainty_threshold = unc_thr
        st.session_state.ood_threshold = ood_thr

        if st.button(tr("regenerate_media", lang), key="regen_media_btn"):
            with st.spinner("Regenerating demo media..."):
                st.session_state.media = generate_media_bundle(
                    df,
                    bundle.images,
                    bundle.outputs,
                    conf_thr,
                    unc_thr,
                    ood_thr,
                    config,
                )
            append_audit_event(
                config,
                {
                    "action": "regenerate_media",
                    "role": role,
                    "confidence_threshold": conf_thr,
                    "uncertainty_threshold": unc_thr,
                    "ood_threshold": ood_thr,
                },
            )

        st.markdown("---")
        st.warning(tr("warning", lang))
        st.info(f"Audit log path: `{config.audit_log_path}`")

    eval_df = compute_eval_df(df, bundle.outputs, conf_thr, unc_thr, ood_thr)

    tabs = st.tabs(
        [
            tr("tab_demo", lang),
            tr("tab_case", lang),
            tr("tab_data", lang),
            tr("tab_model", lang),
            tr("tab_clinical", lang),
            tr("tab_audit", lang),
            tr("tab_validate", lang),
        ]
    )

    # -------------------------------------------------------------------------
    # Live demo tab
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader(tr("tab_demo", lang))
        st.success("Demo media generated automatically on launch. Use the controls to interact.")
        show_media(media, can_export=can(role, "export"))

        active_df = eval_df[eval_df["consent_status"] == "active"]
        choice_df = active_df if not active_df.empty else eval_df
        selected_case = st.selectbox(
            "Select case",
            options=choice_df["case_id"].tolist(),
            key="live_case",
        )

        row = df[df["case_id"] == selected_case].iloc[0]
        out = bundle.outputs[selected_case]
        ev = eval_df[eval_df["case_id"] == selected_case].iloc[0]
        defer = bool(ev["deferred"])

        left, right = st.columns([1, 1])

        with left:
            evidence_png = create_evidence_image(
                bundle.images[selected_case]["CC"],
                out,
                config,
                marked_image=bundle.marked_images[selected_case]["CC"],
                show_raw_marker=False,
            )
            st.image(
                evidence_png,
                caption="Synthetic CC view with density heatmap overlay",
                use_container_width=True,
            )

        with right:
            st.metric(
                tr("predicted_category", lang),
                f"{out.predicted_category.upper()} — {CATEGORY_LABELS[out.predicted_category]}",
            )
            st.metric(tr("confidence", lang), f"{out.confidence:.2%}")
            st.metric(tr("total_uncertainty", lang), f"{out.total_uncertainty:.2%}")
            st.metric("OOD score", f"{out.ood_score:.2%}")
            st.metric(tr("status", lang), tr("defer", lang) if defer else tr("no_defer", lang))

            prob_df = pd.DataFrame(
                [
                    {"category": c, "probability": out.mean_probabilities[c]}
                    for c in CATEGORY_ORDER
                ]
            )
            st.bar_chart(prob_df.set_index("category"))

            st.write(
                f"**Reader A / B:** `{row['reader_a']}` / `{row['reader_b']}` | "
                f"**Consensus:** `{row['consensus_label']}`"
            )

            if not out.quality_pass:
                st.error(f"Quality issues: {', '.join(out.quality_issues)}")

            st.markdown("#### Feedback")
            fb1, fb2, fb3 = st.columns(3)

            def record_feedback(label: str) -> None:
                entry = {
                    "label": label,
                    "role": role,
                    "timestamp_utc": utc_now_iso(),
                }
                st.session_state.feedback.setdefault(selected_case, []).append(entry)
                append_audit_event(
                    config,
                    {
                        "action": "feedback",
                        "role": role,
                        "case_id": selected_case,
                        "feedback": label,
                    },
                )

            if fb1.button("Useful", disabled=not can(role, "feedback"), key=f"fb_useful_{selected_case}"):
                record_feedback("useful")
            if fb2.button("Not useful", disabled=not can(role, "feedback"), key=f"fb_not_useful_{selected_case}"):
                record_feedback("not_useful")
            if fb3.button("Wrong", disabled=not can(role, "feedback"), key=f"fb_wrong_{selected_case}"):
                record_feedback("wrong")

            if st.session_state.feedback.get(selected_case):
                st.caption(f"Stored feedback events: {len(st.session_state.feedback[selected_case])}")

            st.markdown("#### Radiologist confirmation")
            if can(role, "confirm"):
                existing = st.session_state.confirmations.get(selected_case, {})
                default_cat = existing.get("category", out.predicted_category)
                if default_cat not in CATEGORY_ORDER:
                    default_cat = out.predicted_category

                confirmed_cat = st.selectbox(
                    "Confirmed category",
                    options=CATEGORY_ORDER,
                    index=CATEGORY_ORDER.index(default_cat),
                    format_func=lambda x: f"{x.upper()} — {CATEGORY_LABELS[x]}",
                    key=f"confirm_cat_{selected_case}",
                )

                comment = st.text_input(
                    "Comment",
                    value=existing.get("comment", ""),
                    key=f"confirm_comment_{selected_case}",
                )

                if st.button("Save confirmation", key=f"save_confirm_{selected_case}"):
                    st.session_state.confirmations[selected_case] = {
                        "category": confirmed_cat,
                        "comment": comment,
                        "timestamp_utc": utc_now_iso(),
                        "role": role,
                    }
                    append_audit_event(
                        config,
                        {
                            "action": "confirmation",
                            "role": role,
                            "case_id": selected_case,
                            "confirmed_category": confirmed_cat,
                            "comment": comment,
                        },
                    )
                    st.success("Confirmation stored in local session and audit log.")
            else:
                st.info("Only radiologist/admin roles can confirm the category in this demo.")

    # -------------------------------------------------------------------------
    # Case explorer tab
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader(tr("tab_case", lang))

        selected_case = st.selectbox(
            "Case",
            options=df["case_id"].tolist(),
            key="explorer_case",
        )

        row = df[df["case_id"] == selected_case].iloc[0]
        out = bundle.outputs[selected_case]

        show_raw = st.checkbox(
            "Show burned-in marker raw view (demo only)",
            value=False,
            key=f"show_raw_{selected_case}",
        )

        if show_raw:
            st.warning("Raw marker view may contain synthetic burned-in annotations. Do not use for model interpretation.")

        c1, c2 = st.columns(2)
        with c1:
            img = create_evidence_image(
                bundle.images[selected_case]["CC"],
                out,
                config,
                marked_image=bundle.marked_images[selected_case]["CC"],
                show_raw_marker=show_raw,
            )
            st.image(img, caption="CC view", use_container_width=True)
        with c2:
            img = create_evidence_image(
                bundle.images[selected_case]["MLO"],
                out,
                config,
                marked_image=bundle.marked_images[selected_case]["MLO"],
                show_raw_marker=show_raw,
            )
            st.image(img, caption="MLO view", use_container_width=True)

        st.markdown("#### Ensemble member probabilities")
        member_rows: List[Dict[str, Any]] = []
        for vo in out.view_outputs:
            for i, mp in enumerate(vo.member_probabilities):
                pred = max(mp, key=lambda k: mp[k])
                member_rows.append(
                    {
                        "view": vo.view,
                        "member": i,
                        **{c: mp[c] for c in CATEGORY_ORDER},
                        "predicted": pred,
                    }
                )
        st.dataframe(pd.DataFrame(member_rows), use_container_width=True)

        st.markdown("#### DICOM-like metadata")
        meta_tab_raw, meta_tab_deid, meta_tab_check = st.tabs(["Raw simulated", "De-identified", "De-id check"])

        with meta_tab_raw:
            st.json(to_jsonable(bundle.raw_metadata[selected_case]))
        with meta_tab_deid:
            st.json(to_jsonable(bundle.deidentified_metadata[selected_case]))
        with meta_tab_check:
            checks = []
            for view in VIEWS:
                raw = bundle.raw_metadata[selected_case][view]
                deid = bundle.deidentified_metadata[selected_case][view]
                ok, retained, missing = verify_deidentification(raw, deid)
                checks.append({"view": view, "passed": ok, "retained_phi": retained, "missing_required": missing})
            st.dataframe(pd.DataFrame(checks), use_container_width=True)

    # -------------------------------------------------------------------------
    # Data governance tab
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader(tr("tab_data", lang))

        st.markdown("### Data inventory")
        st.dataframe(data_inventory(df, config), use_container_width=True)

        st.markdown("### Integrity checks")
        quality = build_data_quality_report(
            df,
            bundle.outputs,
            config,
            bundle.raw_metadata,
            bundle.deidentified_metadata,
        )
        st.dataframe(quality, use_container_width=True)

        st.markdown("### Condition coverage")
        st.dataframe(
            condition_coverage(df, ["site", "scanner_model", "consensus_label"]),
            use_container_width=True,
        )
        st.dataframe(
            condition_coverage(df, ["age_band", "laterality", "views"]),
            use_container_width=True,
        )

        st.markdown("### Split policy")
        st.write(df.groupby("split").size())

        st.markdown("### Manifest")
        manifest = build_manifest(df, bundle.outputs, config)
        st.dataframe(manifest.drop(columns=["probabilities_json"]).head(20), use_container_width=True)

        manifest_json = json.dumps(
            to_jsonable(manifest.to_dict(orient="records")),
            indent=2,
        ).encode("utf-8")

        st.download_button(
            tr("download_manifest", lang),
            manifest_json,
            "data_manifest.json",
            "application/json",
            key="download_manifest_btn",
            disabled=not can(role, "export"),
        )

        st.markdown("### Consent withdrawal / deletion queue (dry-run)")
        st.dataframe(deletion_queue(df), use_container_width=True)

    # -------------------------------------------------------------------------
    # Model diagnostics tab
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader(tr("tab_model", lang))

        active_eval = eval_df[eval_df["consent_status"] == "active"]

        accuracy = float(active_eval["correct"].mean()) if not active_eval.empty else float("nan")
        defer_rate = float(active_eval["deferred"].mean()) if not active_eval.empty else float("nan")
        sens, spec = dense_sensitivity_specificity(active_eval)
        macro_auc = multiclass_auroc(active_eval, bundle.outputs)
        reader_agreement = float(np.mean(df["reader_a"] == df["reader_b"])) if not df.empty else float("nan")
        kappa = cohen_kappa(df["reader_a"], df["reader_b"]) if not df.empty else float("nan")
        wkappa = weighted_kappa_ordinal(df["reader_a"], df["reader_b"]) if not df.empty else float("nan")

        acc_ci = bootstrap_ci(active_eval, lambda d: float(d["correct"].mean()), n_boot=80, seed=config.seed)
        sens_ci = bootstrap_ci(active_eval, lambda d: dense_sensitivity_specificity(d)[0], n_boot=80, seed=config.seed + 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.1%}" if not math.isnan(accuracy) else "n/a")
        c2.metric("Macro AUROC", f"{macro_auc:.3f}" if not math.isnan(macro_auc) else "n/a")
        c3.metric("Dense sensitivity", f"{sens:.1%}" if not math.isnan(sens) else "n/a")
        c4.metric("Non-dense specificity", f"{spec:.1%}" if not math.isnan(spec) else "n/a")

        if not math.isnan(acc_ci[0]) and not math.isnan(sens_ci[0]):
            st.caption(
                f"Accuracy 95% CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}] | "
                f"Dense sensitivity 95% CI: [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}]"
            )
        else:
            st.caption("Confidence intervals unavailable for this small demo sample.")

        st.metric("Deferral rate", f"{defer_rate:.1%}" if not math.isnan(defer_rate) else "n/a")
        st.metric("Reader exact agreement", f"{reader_agreement:.1%}" if not math.isnan(reader_agreement) else "n/a")
        st.metric("Cohen kappa", f"{kappa:.2f}" if not math.isnan(kappa) else "n/a")
        st.metric("Weighted kappa", f"{wkappa:.2f}" if not math.isnan(wkappa) else "n/a")

        st.markdown("### Calibration")
        temp = fit_temperature(active_eval, bundle.outputs, split="validation")
        ece_raw, table_raw = compute_ece_for_cases(active_eval, bundle.outputs, temperature=1.0)
        ece_cal, table_cal = compute_ece_for_cases(active_eval, bundle.outputs, temperature=temp)

        st.write(f"Fitted temperature (simulated): `{temp:.2f}`")
        st.write(f"ECE raw: `{ece_raw:.3f}` | ECE calibrated: `{ece_cal:.3f}`")

        if not table_raw.empty and not table_cal.empty:
            merged = table_raw.merge(
                table_cal[["bin_center", "accuracy"]],
                on="bin_center",
                suffixes=("", "_calibrated"),
            )
            st.line_chart(
                merged.set_index("bin_center")[
                    ["accuracy", "accuracy_calibrated", "avg_confidence"]
                ]
            )
        else:
            st.info("Not enough cases for calibration plot.")

        st.markdown("### Threshold optimizer")
        sweep = threshold_sweep(active_eval, unc_thr, ood_thr, config)
        best = choose_best_threshold(sweep, config)

        if not sweep.empty:
            st.line_chart(
                sweep.set_index("confidence_threshold")[
                    ["selective_accuracy", "dense_sensitivity", "deferral_rate"]
                ]
            )
            st.dataframe(sweep.tail(15), use_container_width=True)

        if best is not None:
            st.success(
                f"Recommended confidence threshold: `{best['confidence_threshold']:.2f}` "
                f"(selective accuracy `{best['selective_accuracy']:.2%}`, "
                f"deferral `{best['deferral_rate']:.2%}`)"
            )
            if st.button("Apply recommended threshold", key="apply_best_threshold"):
                st.session_state.confidence_threshold = float(best["confidence_threshold"])
                append_audit_event(
                    config,
                    {
                        "action": "apply_recommended_threshold",
                        "role": role,
                        "confidence_threshold": float(best["confidence_threshold"]),
                    },
                )
                rerun_app()
        else:
            st.info("No threshold recommendation available.")

        st.markdown("### Subgroup performance")
        dimension = st.selectbox(
            "Dimension",
            options=["site", "scanner_model", "age_band", "laterality", "views", "consensus_label"],
            key="subgroup_dim",
        )
        sub = subgroup_summary(active_eval, dimension)
        st.dataframe(sub, use_container_width=True)
        if not sub.empty and bool(sub["small_sample_warning"].any()):
            st.warning("Some subgroups have n < 5. Treat those estimates as unstable.")

        st.markdown("### OOD and scanner drift")
        drift_df = active_eval[active_eval["drift_flag"]].copy()
        st.write(f"Drift/OOD flagged cases: `{len(drift_df)}`")
        if not drift_df.empty:
            st.dataframe(
                drift_df[
                    ["case_id", "site", "scanner_model", "ood_score", "total_uncertainty", "deferred"]
                ],
                use_container_width=True,
            )
        else:
            st.info("No drift/OOD flagged cases in current active demo set.")

    # -------------------------------------------------------------------------
    # Clinical workflow tab
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader(tr("tab_clinical", lang))

        selected_case = st.selectbox(
            "Case for report",
            options=df["case_id"].tolist(),
            key="report_case",
        )

        row = df[df["case_id"] == selected_case].iloc[0]
        out = bundle.outputs[selected_case]
        ev = eval_df[eval_df["case_id"] == selected_case].iloc[0]

        template = st.selectbox(
            "Report template",
            options=REPORT_TEMPLATES,
            index=0,
            key="report_template",
        )

        report = render_report(
            row.to_dict(),
            out,
            ev.to_dict(),
            conf_thr,
            unc_thr,
            ood_thr,
            config,
            template=template,
        )

        valid, errors = validate_report(report)

        if valid:
            st.success("Output contract is valid.")
        else:
            st.error("Output contract validation failed: " + "; ".join(errors))

        st.markdown(report_to_markdown(report))

        report_json = json.dumps(to_jsonable(asdict(report)), indent=2)
        st.code(report_json, language="json")

        st.download_button(
            tr("download_report", lang),
            report_json.encode("utf-8"),
            f"{report.case_id}_report.json",
            "application/json",
            key="download_report_btn",
            disabled=not can(role, "export"),
        )

        st.markdown("### Reader-study simulator (simulated only)")
        st.caption(
            "This is not a real reader study. It simulates how decision support might affect "
            "accuracy and reading time under configurable assumptions."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            baseline_acc = st.slider(
                "Baseline reader accuracy",
                min_value=0.50,
                max_value=0.95,
                value=float(config.reader_baseline_accuracy),
                step=0.01,
                key="rs_baseline_acc",
            )
        with col2:
            tool_gain = st.slider(
                "Tool assist gain",
                min_value=0.00,
                max_value=0.20,
                value=float(config.tool_assist_gain),
                step=0.01,
                key="rs_tool_gain",
            )
        with col3:
            automation_penalty = st.slider(
                "Automation bias penalty",
                min_value=0.00,
                max_value=0.30,
                value=float(config.automation_bias_penalty),
                step=0.01,
                key="rs_auto_penalty",
            )

        sim_cfg = replace(
            config,
            reader_baseline_accuracy=baseline_acc,
            tool_assist_gain=tool_gain,
            automation_bias_penalty=automation_penalty,
        )

        if st.button("Run reader-study simulation", key="run_reader_study"):
            summary, case_df = simulate_reader_study(active_eval, sim_cfg, seed=sim_cfg.seed)
            st.session_state.reader_study = (summary, case_df)
            append_audit_event(
                config,
                {
                    "action": "simulate_reader_study",
                    "role": role,
                    "baseline_accuracy": baseline_acc,
                    "tool_gain": tool_gain,
                    "automation_penalty": automation_penalty,
                },
            )

        if st.session_state.reader_study is not None:
            summary, case_df = st.session_state.reader_study
            st.dataframe(summary, use_container_width=True)
            st.dataframe(case_df.head(20), use_container_width=True)

    # -------------------------------------------------------------------------
    # Audit & risk tab
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.subheader(tr("tab_audit", lang))

        st.markdown("### Role permissions")
        perm_df = pd.DataFrame(
            [
                {"role": r, **p}
                for r, p in ROLE_PERMISSIONS.items()
            ]
        )
        st.dataframe(perm_df, use_container_width=True)

        st.markdown("### Audit trail")
        events = read_audit_events(config.audit_log_path, limit=300)

        if not can(role, "view_audit_all"):
            events = [e for e in events if e.get("role") == role]

        if events:
            audit_df = pd.json_normalize(events)
            st.dataframe(audit_df, use_container_width=True)

            audit_json = json.dumps(to_jsonable(events), indent=2).encode("utf-8")
            st.download_button(
                "Download audit JSON",
                audit_json,
                "audit_events.json",
                "application/json",
                key="download_audit_json",
                disabled=not can(role, "export"),
            )
        else:
            st.info("No audit events yet for this role/view.")

        if st.button("Verify audit chain", key="verify_audit_chain_btn"):
            ok, issues, count = verify_audit_chain(config.audit_log_path)
            if ok:
                st.success(f"Audit chain verified. Events checked: {count}")
            else:
                st.error(f"Audit chain failed: {issues}")

        st.markdown("### Retention preview (dry-run only)")
        preview = audit_retention_preview(config.audit_log_path, config.retention_days)
        st.json(to_jsonable(preview))
        st.info("No actual deletion is performed in this demo.")

        st.markdown("### Risk register")
        st.dataframe(risk_register(), use_container_width=True)

        st.markdown("### Model card")
        st.markdown(
            model_card_markdown(
                config,
                active_eval,
                bundle.outputs,
                conf_thr,
                unc_thr,
                ood_thr,
            )
        )

    # -------------------------------------------------------------------------
    # Validation tab
    # -------------------------------------------------------------------------
    with tabs[6]:
        st.subheader(tr("tab_validate", lang))

        if st.button(tr("run_tests", lang), key="run_tests_btn"):
            with st.spinner("Running verification suite..."):
                st.session_state.test_results = run_test_suite()
            append_audit_event(
                config,
                {
                    "action": "run_test_suite",
                    "role": role,
                },
            )

        if st.session_state.test_results is not None:
            results: pd.DataFrame = st.session_state.test_results
            st.dataframe(results, use_container_width=True)

            if bool(results["passed"].all()):
                st.success("All verification checks passed.")
            else:
                st.error("Some verification checks failed. Review the table above.")

            results_json = json.dumps(to_jsonable(results.to_dict(orient="records")), indent=2).encode("utf-8")
            st.download_button(
                "Download test results",
                results_json,
                "verification_results.json",
                "application/json",
                key="download_test_results",
                disabled=not can(role, "export"),
            )
        else:
            st.info("No verification suite has been run yet in this session.")


if __name__ == "__main__":
    if os.getenv("BDA_SELF_TEST", "0") == "1":
        sys.exit(run_cli_self_test())
    main()