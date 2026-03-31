"""
json_to_alpaca.py
=================
Converts Indian Legal Act JSON files → Alpaca JSONL (instruction / input / output)

Handles the full nesting hierarchy:
  Act → Section → Subsection → Clause → Sub-clause

Usage
-----
python json_to_alpaca.py --data_dir "./data" --output_dir "./output"

Output
------
  output/train.jsonl   — training split
  output/eval.jsonl    — evaluation split
  output/report.txt    — run summary
"""

import json
import glob
import re
import random
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Optional

# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = "./data"
DEFAULT_OUTPUT_DIR = "./alpaca_output"
TRAIN_SPLIT        = 0.85
SEED               = 42

WORD_LIMITS = {
    "qa":          160,
    "definition":  120,
    "explain":     160,
    "summary":     110,
    "clause":      130,
    "sub_clause":  120,
    "application": 150,
    "preamble":    130,
}

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# ── SUPERSCRIPT NORMALISATION ──────────────────────────────────────────────────
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

# ── TEXT CLEANING ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalise and sanitise a legal text string."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.translate(_SUPERSCRIPTS)
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", text)
    text = re.sub(r"\[\d+\]", "", text)          # [1] footnote markers
    text = re.sub(r"[\n\r\t\v\f]+", " ", text)   # collapse whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def sanitise(text: str) -> str:
    """Final safety pass — strip anything that would break a JSONL line."""
    return text.replace("\n", " ").replace("\r", "").replace("\t", " ").strip()


# ── TRUNCATION ─────────────────────────────────────────────────────────────────
def truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    candidate = " ".join(words[:max_words])
    for punct in (".", ";", ":", "!", "?"):
        idx = candidate.rfind(punct)
        if idx > len(candidate) // 2:
            return candidate[: idx + 1]
    return candidate


# ── VALIDATION ─────────────────────────────────────────────────────────────────
def is_valid(text: str, min_words: int = 5) -> bool:
    if not text or not text.strip():
        return False
    t = text.strip()
    if len(t.split()) < min_words:
        return False
    if t.isdigit():
        return False
    # reject bare "In this Act, unless the context otherwise requires,—" stubs
    if t.endswith("—") and len(t.split()) < 10:
        return False
    return True


# ── DEDUPLICATION ──────────────────────────────────────────────────────────────
def fingerprint(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()


# ── ACT METADATA ───────────────────────────────────────────────────────────────
def act_name(act: dict) -> str:
    title = clean_text(act.get("act_title", "Unknown Act"))
    raw_date = act.get("act_enactment_date", "") or ""
    match = YEAR_RE.search(raw_date)
    year = match.group() if match else ""
    return f"{title} ({year})" if year else title


# ── PLAIN-ENGLISH REWRITER ─────────────────────────────────────────────────────
_REPLACEMENTS = [
    # Preserve legal force while simplifying language
    (r"\bshall\b",           "is required to"),
    (r"\bshall not\b",       "is not permitted to"),
    (r"\bmay\b",             "is permitted to"),

    # Safe simplifications
    (r"\bnotwithstanding\b", "despite"),
    (r"\bprovided that\b",   "however"),
    (r"\bherein\b",          "in this Act"),
    (r"\bthereof\b",         "of that"),
    (r"\bpursuant to\b",     "in accordance with"),
    (r"\bwhereby\b",         "by which"),
    (r"\baforesaid\b",       "mentioned above"),
    (r"\bhereunder\b",       "under this Act"),
    (r"\bviz\.\b",           "namely"),
    (r"\binter alia\b",      "among other things"),
    (r"\bab initio\b",       "from the beginning"),
]


def rewrite_plain_english(text: str) -> str:
    text = clean_text(text)

    # Apply replacements
    for pattern, repl in _REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    lower = text.lower()

    # Determine explanatory prefix
    if "is required to" in lower or "shall not" in lower or "not permitted" in lower:
        prefix = "This provision requires that "
    elif "is permitted to" in lower:
        prefix = "This provision allows that "
    elif "means" in lower or "includes" in lower:
        prefix = "This provision defines that "
    elif "prohibit" in lower:
        prefix = "This provision prohibits "
    else:
        prefix = "This provision states that "

    body = text[0].lower() + text[1:] if text else text
    return prefix + body


# ── EXTRACTIVE SUMMARISER ──────────────────────────────────────────────────────
_SUMMARY_WEIGHTS = {
    "shall": 3, "must": 3, "means": 2, "includes": 2,
    "liable": 2, "penalty": 2, "prohibited": 2, "offence": 2,
    "may": 1, "provided": 1, "right": 1, "duty": 1, "appointed": 1,
}

def extract_summary(text: str, max_words: int) -> str:
    sentences = re.split(r"(?<=[.!?;]) +", clean_text(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 2:
        return truncate(clean_text(text), max_words)
    scored      = sorted(enumerate(sentences),
                         key=lambda x: sum(w for k, w in _SUMMARY_WEIGHTS.items()
                                           if k in x[1].lower()),
                         reverse=True)
    top_indices = sorted(i for i, _ in scored[:2])
    return truncate(" ".join(sentences[i] for i in top_indices), max_words)


# ── EXAMPLE BUILDER ────────────────────────────────────────────────────────────
def make_example(
    instruction: str,
    input_text:  str,
    output:      str,
    task:        str,
    min_words:   int = 5,
) -> Optional[dict]:
    output = clean_text(output)
    output = truncate(output, WORD_LIMITS.get(task, 130))
    if not is_valid(output, min_words):
        return None
    if input_text and fingerprint(output) == fingerprint(input_text):
        return None
    return {
        "instruction": clean_text(instruction),
        "input":       clean_text(input_text),
        "output":      output,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

# QA — Section level
SECTION_QA = [
    "What does Section {sec} of {act} say?",
    "Explain Section {sec} of {act}.",
    "What is covered under Section {sec} of {act}?",
    "Describe the legal provisions under Section {sec} of {act}.",
    "What are the key legal points of Section {sec} of {act}?",
    "State the content of Section {sec} as prescribed under {act}.",
    "What legal obligations arise under Section {sec} of {act}?",
    "What is the heading of Section {sec} under {act} and what does it cover?",
]

# QA — Subsection level
SUBSECTION_QA = [
    "What does subsection {sub} of Section {sec} of {act} provide?",
    "State the contents of subsection {sub} of Section {sec} under {act}.",
    "What legal obligations or rights does subsection {sub} of Section {sec} of {act} impose?",
    "Cite the provisions of subsection {sub} under Section {sec} of {act}.",
    "What does the legislature intend by subsection {sub} of Section {sec} of {act}?",
    "What is the legal effect of subsection {sub} of Section {sec} of {act}?",
    "How does subsection {sub} of Section {sec} of {act} read?",
]

# EXPLAIN — Subsection level
EXPLAIN_TMPL = [
    "Explain the legal significance of subsection {sub} of Section {sec} of {act}.",
    "What is the legal interpretation of subsection {sub} under Section {sec} of {act}?",
    "How should subsection {sub} of Section {sec} of {act} be construed under law?",
    "Elaborate on the statutory scope of subsection {sub} of Section {sec} of {act}.",
    "What rights, duties, or liabilities arise from subsection {sub} of Section {sec} of {act}?",
    "Provide a legal explanation of subsection {sub} as defined in Section {sec} of {act}.",
    "In plain language, what does subsection {sub} of Section {sec} of {act} mean?",
]

# SUMMARY — Subsection level
SUMMARY_TMPL = [
    "Provide a legal summary of subsection {sub} of Section {sec} of {act}.",
    "Briefly state the legal import of subsection {sub} under Section {sec} of {act}.",
    "What is the essence of the provision in subsection {sub} of Section {sec} of {act}?",
    "Summarize the statutory mandate of subsection {sub} of Section {sec} of {act}.",
    "Give a concise legal overview of subsection {sub} under Section {sec} of {act}.",
    "In brief, what does subsection {sub} of Section {sec} of {act} legally prescribe?",
]

# CLAUSE — Clause level
CLAUSE_TMPL = [
    "What is the legal effect of clause {clause} under subsection {sub} of Section {sec} of {act}?",
    "State the provision of clause {clause} in subsection {sub} of Section {sec} of {act}.",
    "What does clause {clause} of subsection {sub}, Section {sec} of {act} legally mandate?",
    "Explain the statutory significance of clause {clause} under Section {sec} of {act}.",
    "How does clause {clause} of subsection {sub} of Section {sec} of {act} affect legal rights?",
    "What legal conditions are imposed by clause {clause} of subsection {sub} under {act}?",
    "Cite clause {clause} of subsection {sub} of Section {sec} of {act}.",
]

# DEFINITION — for Section 2 / definitions sections
DEFINITION_TMPL = [
    "How is \"{term}\" defined under {act}?",
    "What does \"{term}\" mean as per {act}?",
    "Define \"{term}\" as used in {act}.",
    "What is the statutory meaning of \"{term}\" under {act}?",
    "According to {act}, what is meant by \"{term}\"?",
    "What legal definition does {act} assign to the term \"{term}\"?",
    "Under {act}, how is the term \"{term}\" to be understood?",
]

# SUB-CLAUSE level
SUB_CLAUSE_TMPL = [
    "What does sub-clause {sub_c} of clause {clause} under Section {sec} of {act} state?",
    "Explain sub-clause {sub_c} of clause {clause} of Section {sec} of {act}.",
    "What legal provision is made by sub-clause {sub_c} under clause {clause} of Section {sec} of {act}?",
    "State the contents of sub-clause {sub_c} under clause {clause}, Section {sec} of {act}.",
    "What is the legal significance of sub-clause {sub_c} of clause {clause} in Section {sec} of {act}?",
]

# APPLICATION — Section level
APPLICATION_TMPL = [
    "Give a practical example of how Section {sec} of {act} would apply.",
    "In what real-world situation would Section {sec} of {act} be invoked?",
    "How does Section {sec} of {act} affect the rights of an individual in practice?",
    "Provide a scenario where subsection {sub} of Section {sec} of {act} becomes relevant.",
    "Illustrate the practical application of subsection {sub} under Section {sec} of {act}.",
    "Under what circumstances would a person be affected by Section {sec} of {act}?",
]

_APPLICATION_PREFIXES = [
    "In practice, this provision applies when ",
    "A practical application of this section arises when ",
    "This legal provision becomes relevant when ",
    "Under real-world circumstances, this section governs situations where ",
    "This section would be invoked in a scenario where ",
]

# PREAMBLE
PREAMBLE_TMPL = [
    "What is the purpose and objective of {act}?",
    "What does the preamble of {act} state?",
    "Why was {act} enacted?",
    "What are the stated objectives behind the enactment of {act}?",
    "What problem does {act} seek to address as per its preamble?",
    "Summarize the preamble of {act}.",
]


# ══════════════════════════════════════════════════════════════════════════════
# TASK GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def task_preamble(act: dict, rng: random.Random) -> list[dict]:
    """Generate QA pairs from the preamble."""
    examples = []
    name     = act_name(act)
    preamble = clean_text(act.get("act_preamble", ""))
    if not is_valid(preamble, min_words=20):
        return examples
    inst = rng.choice(PREAMBLE_TMPL).format(act=name)
    ex   = make_example(inst, "", preamble, "preamble")
    if ex:
        examples.append(ex)
    return examples


def task_qa(act: dict, rng: random.Random) -> list[dict]:
    """Generate QA pairs from section and subsection text."""
    examples, name = [], act_name(act)
    for sec in act.get("act_sections", []):
        sec_num  = sec.get("section_number", "")
        sec_text = clean_text(sec.get("section_text", ""))

        # Section-level QA (only if section_text is non-empty)
        if is_valid(sec_text):
            inst = rng.choice(SECTION_QA).format(sec=sec_num, act=name)
            context = f"Section {sec_num} of {name}"
            ex = make_example(inst, context, sec_text, "qa")
            if ex:
                examples.append(ex)

        # Subsection-level QA
        for sub in sec.get("subsections", []):
            sub_num  = sub.get("subsection_number", "")
            sub_text = clean_text(sub.get("subsection_text", ""))
            if not is_valid(sub_text):
                continue
            context = f"Section {sec_num} of {name}:\n{sec_text}"

            inst = rng.choice(SUBSECTION_QA).format(
            sub=sub_num, sec=sec_num, act=name)

            ex = make_example(inst, context, sub_text, "qa")
            if ex:
                examples.append(ex)
                return examples


def task_explain(act: dict, rng: random.Random) -> list[dict]:
    """Rewrite subsections in plain English."""
    examples, name = [], act_name(act)
    for sec in act.get("act_sections", []):
        sec_num = sec.get("section_number", "")
        for sub in sec.get("subsections", []):
            raw  = sub.get("subsection_text", "")
            if len(clean_text(raw).split()) < 8:
                continue
            sub_num   = sub.get("subsection_number", "")
            explained = rewrite_plain_english(raw)
            inst = rng.choice(EXPLAIN_TMPL).format(
                sub=sub_num, sec=sec_num, act=name)
            ex = make_example(inst, clean_text(raw), explained, "explain")
            if ex:
                examples.append(ex)
    return examples


def task_summary(act: dict, rng: random.Random) -> list[dict]:
    """Extractive summary of subsections."""
    examples, name = [], act_name(act)
    for sec in act.get("act_sections", []):
        sec_num = sec.get("section_number", "")
        for sub in sec.get("subsections", []):
            raw     = sub.get("subsection_text", "")
            sub_num = sub.get("subsection_number", "")
            if len(clean_text(raw).split()) < 12:
                continue
            summary = extract_summary(raw, WORD_LIMITS["summary"])
            if not is_valid(summary):
                continue
            inst = rng.choice(SUMMARY_TMPL).format(
                sub=sub_num, sec=sec_num, act=name)
            ex = make_example(inst, clean_text(raw), summary, "summary")
            if ex:
                examples.append(ex)
    return examples


def task_definition(act: dict, rng: random.Random) -> list[dict]:
    """
    Extract definition QA pairs from clauses.
    Works for Section 2 (Definitions) and any clause containing 'means' or 'includes'.
    Handles sub-clauses for multi-part definitions (e.g. 'competent authority').
    """
    examples, name = [], act_name(act)
    for sec in act.get("act_sections", []):
        sec_num = sec.get("section_number", "")
        for sub in sec.get("subsections", []):
            for clause in sub.get("clauses", []):
                clause_text = clean_text(clause.get("clause_text", ""))
                if not clause_text:
                    continue

                # Extract term from "term" means / includes …
                term_match = re.match(
                    r'^["\u201c]?([A-Za-z][A-Za-z\s\-]{1,40})["\u201d]?\s+'
                    r'(?:means|includes|refers to)',
                    clause_text, re.IGNORECASE
                )
                if not term_match:
                    continue

                term = term_match.group(1).strip().strip('"').strip('\u201c\u201d')

                # If there are sub-clauses, build a combined output
                sub_clauses = clause.get("sub_clauses", [])
                if sub_clauses:
                    parts = [clause_text]
                    for sc in sub_clauses:
                        sc_text = clean_text(sc.get("sub_clause_text", ""))
                        if sc_text:
                            sc_num = sc.get("sub_clause_number", "")
                            parts.append(f"{sc_num} {sc_text}")
                    output = " ".join(parts)
                else:
                    output = clause_text

                inst = rng.choice(DEFINITION_TMPL).format(term=term, act=name)
                context = f"Definitions section of {name}"

                ex = make_example(inst, context, output, "definition")
                if ex:
                    examples.append(ex)
    return examples


def task_clause(act: dict, rng: random.Random) -> list[dict]:
    """Generate QA pairs from individual clauses."""
    examples, name = [], act_name(act)
    for sec in act.get("act_sections", []):
        sec_num = sec.get("section_number", "")
        for sub in sec.get("subsections", []):
            sub_num = sub.get("subsection_number", "")
            for clause in sub.get("clauses", []):
                clause_num  = clause.get("clause_number", "")
                clause_text = clean_text(clause.get("clause_text", ""))
                if not is_valid(clause_text):
                    continue
                inst = rng.choice(CLAUSE_TMPL).format(
                    clause=clause_num, sub=sub_num, sec=sec_num, act=name)
                context = f"Section {sec_num}, Subsection {sub_num} of {name}"

                ex = make_example(inst, context, clause_text, "clause")
                if ex:
                    examples.append(ex)
    return examples


def task_sub_clause(act: dict, rng: random.Random) -> list[dict]:
    """Generate QA pairs from sub-clauses (deepest nesting level)."""
    examples, name = [], act_name(act)
    for sec in act.get("act_sections", []):
        sec_num = sec.get("section_number", "")
        for sub in sec.get("subsections", []):
            for clause in sub.get("clauses", []):
                clause_num = clause.get("clause_number", "")
                for sc in clause.get("sub_clauses", []):
                    sc_num  = sc.get("sub_clause_number", "")
                    sc_text = clean_text(sc.get("sub_clause_text", ""))
                    if not is_valid(sc_text):
                        continue
                    inst = rng.choice(SUB_CLAUSE_TMPL).format(
                        sub_c=sc_num, clause=clause_num, sec=sec_num, act=name)
                    context = f"Section {sec_num}, Clause {clause_num} of {name}"

                    ex = make_example(inst, context, sc_text, "sub_clause")
                    if ex:
                        examples.append(ex)
    return examples


def task_application(act: dict, rng: random.Random) -> list[dict]:
    """Generate practical application examples from section + subsection pairs."""
    examples, name = [], act_name(act)
    for sec in act.get("act_sections", []):
        sec_num  = sec.get("section_number", "")
        sec_text = clean_text(sec.get("section_text", ""))
        subs     = sec.get("subsections", [])

        # Fall back to first long subsection if section_text is empty
        if not is_valid(sec_text, min_words=12):
            for sub in subs:
                candidate = clean_text(sub.get("subsection_text", ""))
                if len(candidate.split()) >= 12:
                    sec_text = candidate
                    break

        if not is_valid(sec_text, min_words=12):
            continue

        prefix = rng.choice(_APPLICATION_PREFIXES)
        body   = sec_text[0].lower() + sec_text[1:] if sec_text else sec_text

        output = f"For example, {prefix.lower()} {body}"

        # Pair with first valid subsection as input context
        for sub in subs:
            sub_num  = sub.get("subsection_number", "")
            sub_text = clean_text(sub.get("subsection_text", ""))
            if not is_valid(sub_text, min_words=10):
                continue
            tmpl = rng.choice(APPLICATION_TMPL)
            # Some templates need {sub}, some don't
            try:
                inst = tmpl.format(sec=sec_num, sub=sub_num, act=name)
            except KeyError:
                inst = tmpl.format(sec=sec_num, act=name)
            context = f"Section {sec_num} of {name}:\n{sub_text}"
            ex = make_example(inst, context, output, "application")
            if ex:
                examples.append(ex)
                break   # one application example per section
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# SAVE & VERIFY
# ══════════════════════════════════════════════════════════════════════════════

def save_jsonl(path: Path, data: list[dict]) -> None:
    """Write Alpaca JSONL — one sanitised JSON object per line."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for item in data:
            record = {
                "instruction": sanitise(item["instruction"]),
                "input":       sanitise(item["input"]),
                "output":      sanitise(item["output"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    written = sum(1 for _ in open(path, encoding="utf-8"))
    if written != len(data):
        log.error("WRITE MISMATCH: expected %d lines, file has %d → %s",
                  len(data), written, path)
    else:
        log.info("Saved & verified %d examples → %s", len(data), path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

TASKS = [
 ("qa", task_qa),
 ("definition", task_definition),
 ("explain", task_explain),
 ("clause", task_clause),
 ("sub_clause", task_sub_clause),
 ("application", task_application),
]


def convert(data_dir: str, output_dir: str, split: float, seed: int) -> None:
    files = sorted(
        glob.glob(str(Path(data_dir) / "*.json")) +
        glob.glob(str(Path(data_dir) / "**" / "*.json"), recursive=True)
    )
    # deduplicate in case recursive glob overlaps
    files = list(dict.fromkeys(files))

    if not files:
        log.error("No JSON files found in: %s", data_dir)
        return

    log.info("Found %d JSON file(s).", len(files))

    rng           = random.Random(seed)
    all_examples: list[dict] = []
    seen_pairs:   set[str]   = set()
    seen_outputs: set[str]   = set()
    task_counts   = {name: 0 for name, _ in TASKS}
    files_ok = files_err = 0

    for file in files:
        try:
            raw = Path(file).read_text(encoding="utf-8")
            # Handle files that are a bare object (not wrapped in [])
            data = json.loads(raw)
            acts = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s — %s", file, exc)
            files_err += 1
            continue
        files_ok += 1

        for act in acts:
            for task_name, task_fn in TASKS:
                for ex in task_fn(act, rng):
                    pair_key = ex["instruction"] + ex["output"][:80]
                    fp       = fingerprint(ex["output"])
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    seen_outputs.add(fp)
                    all_examples.append(ex)
                    task_counts[task_name] += 1

    if not all_examples:
        log.error("No valid examples were generated. Check your JSON structure.")
        return

    rng.shuffle(all_examples)
    split_idx = int(len(all_examples) * split)
    train     = all_examples[:split_idx]
    eval_     = all_examples[split_idx:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_jsonl(out / "train.jsonl", train)
    save_jsonl(out / "eval.jsonl",  eval_)

    