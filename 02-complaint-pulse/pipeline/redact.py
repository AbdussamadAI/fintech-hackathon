"""PII redaction before anything is stored, embedded or sent to a model.

Masks card numbers, BVN/NIN (11 digits), account numbers (10 digits), phone numbers, e-mail addresses and money
amounts. Amounts are kept separately as structured fields (the LLM extracts them); topics should not depend on them.
"""
import re

_RULES = [
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<CARD>"),
    (re.compile(r"(?:\+?234|\b0)[789][01]\d{8}\b"), "<PHONE>"),
    (re.compile(r"\b\d{11}\b"), "<BVN_NIN>"),
    (re.compile(r"\b\d{10}\b"), "<ACCOUNT>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    # money: currency marker, or k/naira suffix, or thousands separators (never versions like 4.2.1)
    (re.compile(r"(?:₦|\bNGN ?|\bN)\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:k|K|naira|Naira)\b|\b\d{1,3}(?:,\d{3})+\b"), "<AMOUNT>"),
]


def redact(text: str) -> str:
    for rx, tag in _RULES:
        text = rx.sub(tag, text)
    return text
