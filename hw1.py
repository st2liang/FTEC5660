#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    import os

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    system_prompt = (
        "You are a meticulous supermarket-receipt auditor. "
        "Read the supplied receipt image carefully and report only amounts "
        "that are visibly printed on that receipt. Do not estimate, infer "
        "from another receipt, or include explanations outside the requested "
        "JSON."
    )
    human_prompt = """
Audit the attached supermarket receipt.

Use these definitions exactly:
- subtotal_after_discounts_before_rounding: the receipt's printed SUBTOTAL
  after all discounts have been applied but before any ROUNDING line.
- discount_total: the sum of the absolute values of every discount line,
  including promotion, coupon, member, app, packaging-damage, percentage-off,
  and similar discounts. Report it as a positive number. Do not include the
  ROUNDING line here.
- paid_amount: the final amount actually paid after ROUNDING. Use the final
  payment/tender line (for example OCTOPUS, CASH, CARD, or PAYMENT), not the
  SUBTOTAL.
- original_total: subtotal_after_discounts_before_rounding + discount_total.
  Do not add or subtract the ROUNDING amount.

Read every line, including small print, and reconcile the printed values.
Return only one valid JSON object with exactly these four decimal-string keys:
paid_amount, subtotal_after_discounts_before_rounding, discount_total,
original_total. Do not use currency symbols, markdown, comments, or extra keys.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                    {"type": "text", "text": human_prompt},
                ],
            ),
        ]
    )
    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
        max_retries=2,
    )
    return prompt | llm


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    import os

    verbose = os.environ.get("HW1_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _parse_json_object(text: str) -> dict[str, Any]:
        """Extract the first JSON object from a model response."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start() :])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("the model response did not contain a JSON object")

    def _as_decimal(value: Any) -> Decimal | None:
        """Convert a model-supplied number to Decimal without changing it."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, Decimal):
            return value
        text = str(value).strip()
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
        if match is None:
            return None
        try:
            number = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None
        if text.startswith("(") and text.endswith(")"):
            number = -abs(number)
        return number

    def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key).strip().lower().replace(" ", "_").replace("-", "_"): value
            for key, value in payload.items()
        }

    def _find_decimal(
        payload: dict[str, Any], names: tuple[str, ...]
    ) -> Decimal | None:
        for name in names:
            if name in payload:
                number = _as_decimal(payload[name])
                if number is not None:
                    return number
        return None

    def _find_decimal_in_text(text: str, names: tuple[str, ...]) -> Decimal | None:
        for name in names:
            pattern = (
                r"(?i)[\"']?"
                + re.escape(name)
                + r"[\"']?\s*[:=]\s*[^\d\-]*"
                + r"(-?\d[\d,]*(?:\.\d+)?)"
            )
            match = re.search(pattern, text)
            if match is None:
                continue
            number = _as_decimal(match.group(1))
            if number is not None:
                return number
        return None

    paid_names = (
        "paid_amount",
        "amount_paid",
        "amount_paid_after_rounding",
        "final_payment",
        "final_amount",
        "payment_amount",
        "total_paid",
    )
    subtotal_names = (
        "subtotal_after_discounts_before_rounding",
        "subtotal_after_discounts",
        "subtotal_before_rounding",
        "subtotal",
    )
    discount_names = (
        "discount_total",
        "total_discount",
        "discount_amount",
    )
    original_names = (
        "original_total",
        "amount_without_discounts",
        "original_amount",
        "without_discount_total",
        "total_before_discounts",
    )

    batch_inputs = [{"image_url": image_data_url(path)} for path in images]
    responses = chain.batch(batch_inputs)
    if len(responses) != len(images):
        raise ValueError("the chain returned a different number of responses")

    total_spent = Decimal("0.00")
    total_original = Decimal("0.00")

    for image, response in zip(images, responses):
        text = response_text(response)
        try:
            payload = _normalise_payload(_parse_json_object(text))
        except ValueError:
            payload = {}

        paid = _find_decimal(payload, paid_names)
        subtotal = _find_decimal(payload, subtotal_names)
        discount = _find_decimal(payload, discount_names)
        original = _find_decimal(payload, original_names)

        if paid is None:
            paid = _find_decimal_in_text(text, paid_names)
        if subtotal is None:
            subtotal = _find_decimal_in_text(text, subtotal_names)
        if discount is None:
            discount = _find_decimal_in_text(text, discount_names)
        if original is None:
            original = _find_decimal_in_text(text, original_names)

        if discount is not None:
            discount = abs(discount)

        model_original = original
        computed_original: Decimal | None = None

        # The assignment defines the second answer as SUBTOTAL plus every
        # discount. Recomputing it here also catches arithmetic mistakes made
        # by the model when it fills the original_total field.
        if subtotal is not None:
            computed_original = subtotal + (discount or Decimal("0.00"))
            if original is None or discount is not None:
                original = computed_original

        if paid is None or original is None:
            raise ValueError(
                f"could not extract both required amounts from {image.name}"
            )

        if verbose:
            print(f"\n[{image.name}] raw model response:")
            print(text)
            print(
                f"[{image.name}] parsed: "
                f"paid={paid}, subtotal={subtotal}, discount={discount}, "
                f"model_original={model_original}, "
                f"computed_original={computed_original}, "
                f"used_original={original}"
            )

        total_spent += paid
        total_original += original

        if verbose:
            print(
                f"[{image.name}] running totals: "
                f"paid={total_spent:.2f}, original={total_original:.2f}"
            )

    if verbose:
        print(
            "\nFinal totals: "
            f"paid=HK${total_spent:.2f}, original=HK${total_original:.2f}"
        )

    return {
        QUERY_1: f"HK${total_spent:.2f}",
        QUERY_2: f"HK${total_original:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
