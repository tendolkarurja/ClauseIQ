"""
ClauseIQ Engine: Calibrated Legal Analysis & Due Diligence Memorandum Pipeline
Tailored specifically for Indian Buy, Sell, and Lease Contracts.
Under the statutory framework of:
- Transfer of Property Act, 1882 (TPA)
- Indian Contract Act, 1872 (ICA)
- Model Tenancy Act, 2021 (MTA)
- Registration Act, 1908 & Indian Stamp Act, 1899
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
import re
import json
from typing import Dict, List, Any, Optional, Tuple
import pymupdf as fitz
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Reconfigure stdout for Windows console UTF-8 safety
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global singletons for models
_EMBED_MODEL = None
_NLI_TOKENIZER = None
_NLI_MODEL = None
_DEVICE = None
_ID2LABEL = None
_CONTRA_IDX = None

def get_models():
    """Lazy load models once and cache in memory."""
    global _EMBED_MODEL, _NLI_TOKENIZER, _NLI_MODEL, _DEVICE, _ID2LABEL, _CONTRA_IDX
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained('microsoft/deberta-base-mnli', local_files_only=True)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-base-mnli', local_files_only=True)
        _NLI_MODEL.eval()
        _DEVICE = next(_NLI_MODEL.parameters()).device
        _ID2LABEL = {int(k): str(v).lower() for k, v in _NLI_MODEL.config.id2label.items()}
        _CONTRA_IDX = [k for k, v in _ID2LABEL.items() if "contradiction" in v][0]
    return _EMBED_MODEL, _NLI_TOKENIZER, _NLI_MODEL, _DEVICE, _ID2LABEL, _CONTRA_IDX


def clean_indian_legal_text(text: str) -> str:
    """Removes placeholder dots, unicode replacement artifacts, and normalizes formatting."""
    t = text.replace('\ufffd', ' ').replace('\u2026', ' ')
    
    # 1. Clean currency and amount interleaving:
    # E.g. 'Rs... (Rupees 25000 Twenty ... five thousand ... only)' -> 'Rs. 25000 (Rupees Twenty five thousand only)'
    m_amt = re.search(r'Rs\.?\s*[\.\_\s]*\(Rupees\s*(\d+)\s+([A-Za-z\s]+?)\s*only\)', t, re.I)
    if m_amt:
        num, words = m_amt.group(1), m_amt.group(2).strip()
        t = re.sub(r'Rs\.?\s*[\.\_\s]*\(Rupees\s*\d+\s+[A-Za-z\s]+?\s*only\)', f'Rs. {num} (Rupees {words} only)', t, flags=re.I)

    # Pattern: Rs... <num> (Rupees ... <words> only) <trailing_word>
    m_amt2 = re.search(r'Rs\.?\s*[\.\_\s]*(\d+)\s*\(Rupees\s*[\.\_\s]*([A-Za-z\s]+?)\s*only\)\s*,?\s*([A-Za-z]+)?', t, re.I)
    if m_amt2:
        num = m_amt2.group(1)
        words = m_amt2.group(2).strip()
        trailing = m_amt2.group(3) or ""
        combined_words = f"{words} {trailing}".strip()
        t = re.sub(r'Rs\.?\s*[\.\_\s]*\d+\s*\(Rupees\s*[\.\_\s]*[A-Za-z\s]+?\s*only\)\s*,?\s*[A-Za-z]*', f'Rs. {num} (Rupees {combined_words} only)', t, flags=re.I)

    # Pattern: From ... To ... <date1> <date2>
    m_dates = re.search(r'From\s*[\.\_\s]*To\s*[\.\_\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\s+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', t, re.I)
    if m_dates:
        d1, d2 = m_dates.group(1), m_dates.group(2)
        t = re.sub(r'From\s*[\.\_\s]*To\s*[\.\_\s]*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\s+\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', f'From {d1} To {d2}', t, flags=re.I)

    # 2. Clean header fragments that leak into text
    t = re.sub(r'Final\s+model\s+draft\s+agreed\s+to\s+by\s+(?:and)?', '', t, flags=re.I)
    t = re.sub(r'model\s+draft\s+agreed\s+to\s+by\s+[^\n]*', '', t, flags=re.I)

    # 3. Remove all remaining sequences of 2 or more dots or underscores
    t = re.sub(r'[\.\_]{2,}', ' ', t)
    
    # 4. Clean broken spacing around punctuation
    t = re.sub(r'\s+([,:\.;])', r'\1', t)
    t = re.sub(r'([,:\.;])\s*', r'\1 ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def refine_party_text(party_str: str) -> str:
    """Formats raw party lines into clean legal definitions."""
    p = re.sub(r'^[,\.\s\(\)\/]+', '', party_str)
    p = re.sub(r'^\s*\(?\s*s/o\s*(?:or|and)?\s*d/o\s*\)?\s*[\.,\s]*', '', p, flags=re.I)
    p = re.sub(r'^\s*s/o\s*[\.,\s]*', '', p, flags=re.I)
    p = re.sub(r'^\s*d/o\s*[\.,\s]*', '', p, flags=re.I)
    
    # Clean mid-string template placeholders
    p = re.sub(r'[,;\s]*\.\s*,?\s*s/o\s*d/o\s*[\.,\s]*', ' s/o ', p, flags=re.I)
    p = re.sub(r'[,;\s]*s/o\s*d/o\s*[\.,\s]*', ' s/o ', p, flags=re.I)
    p = re.sub(r'[,;\s]*r/o\s*[\.,\s]*$', '', p, flags=re.I)
    
    m_two = re.search(r'((?:Mr\.|Ms\.|Shri|Smt\.|Mrs\.)\s+[A-Za-z\s]+?)\s+((?:Mr\.|Ms\.|Shri|Smt\.|Mrs\.)\s+[A-Za-z\s]+?)(,\s*r/o.*)', p, re.I)
    if m_two:
        p1 = m_two.group(1).strip()
        p2 = m_two.group(2).strip()
        tail = m_two.group(3).strip()
        p = f"{p1} (s/o {p2}){tail}"
        
    m_two_rel = re.search(r'((?:Mr\.|Ms\.|Shri|Smt\.|Mrs\.)\s+[A-Za-z\s]+?)\s+d/o\s*[\.,\s]*((?:Mr\.|Ms\.|Shri|Smt\.|Mrs\.)\s+[A-Za-z\s]+?)(,\s*r/o.*)', p, re.I)
    if m_two_rel:
        p1 = m_two_rel.group(1).strip()
        p2 = m_two_rel.group(2).strip()
        tail = m_two_rel.group(3).strip()
        p = f"{p1} (d/o {p2}){tail}"
        
    p = re.sub(r'\s+([,:\.;])', r'\1', p)
    p = re.sub(r'([,:\.;])\s*', r'\1 ', p)
    p = re.sub(r'\s+', ' ', p).strip()
    return p


def extract_contract(file_input: Any) -> Dict[str, Any]:
    """
    Advanced layout-aware contract extractor for Indian real estate deeds.
    Eliminates template placeholder dots, cleans overprinted text,
    and extracts parties and covenants in proper sequential order.
    """
    if isinstance(file_input, str):
        doc = fitz.open(file_input)
        filename = os.path.basename(file_input)
    elif hasattr(file_input, "name"):
        doc = fitz.open(file_input.name)
        filename = os.path.basename(file_input.name)
    else:
        doc = fitz.open(stream=file_input, filetype="pdf")
        filename = "uploaded_contract.pdf"

    # Step 1: Reconstruct clean lines page-by-page using vertical clustering
    pages_lines = []
    for p in doc:
        words = p.get_text("words")
        valid_words = []
        for w in words:
            # Header filter: top 45pt with MoUD/DoLR/draft text
            if w[1] < 45 and any(h in w[4].lower() for h in ["model draft", "moud", "dolr", "final model"]):
                continue
            # Footer filter: bottom 40pt with standalone page number only
            if w[1] > 710 and re.fullmatch(r'\d{1,2}', w[4].strip()):
                continue
            # Filter pure dot/underscore tokens
            cln = re.sub(r'[\.\u2026\ufffd\_]{2,}', '', w[4]).strip()
            if not cln and len(w[4].strip()) > 1:
                continue
            valid_words.append(w)

        # Sort by vertical Y (clustering into lines), then horizontal X
        valid_words.sort(key=lambda w: (w[1], w[0]))
        lines = []
        cur_line = []
        cur_y = None
        for w in valid_words:
            if cur_y is None or abs(w[1] - cur_y) <= 4.0:
                cur_line.append(w)
                cur_y = sum(x[1] for x in cur_line) / len(cur_line)
            else:
                cur_line.sort(key=lambda x: x[0])
                lines.append(" ".join(x[4] for x in cur_line))
                cur_line = [w]
                cur_y = w[1]
        if cur_line:
            cur_line.sort(key=lambda x: x[0])
            lines.append(" ".join(x[4] for x in cur_line))

        pages_lines.append("\n".join(lines))

    full_text = "\n".join(pages_lines)

    # Determine contract type
    is_lease = bool(
        re.search(r'lease', filename, re.I) or 
        re.search(r'lease\s+deed', full_text[:1200], re.I) or
        re.search(r'lessor', full_text[:1200], re.I)
    )
    doc_type = "lease_deed" if is_lease else "sale_deed"

    # Step 2: Extract Structured Parties & Metadata in strict order
    parties = []
    if doc_type == "lease_deed":
        # Extract Lessor (between BETWEEN and (hereinafter called the Lessor))
        m_lessor = re.search(r'BETWEEN\s+(.*?)\s*\(hereinafter\s+called\s+the\s+Lessor\)', full_text, re.DOTALL | re.I)
        if m_lessor:
            raw_l = clean_indian_legal_text(m_lessor.group(1))
            refined_l = refine_party_text(raw_l)
            parties.append(f"Lessor: {refined_l}")

            # Extract Lessee starting strictly after the Lessor block to prevent cross-contamination
            m_lessee = re.search(r'(?:AND)\s+(.*?)\s*\(hereinafter\s+called\s+the\s+Lessee\)', full_text[m_lessor.end():], re.DOTALL | re.I)
            if m_lessee:
                raw_lessee = clean_indian_legal_text(m_lessee.group(1))
                refined_lessee = refine_party_text(raw_lessee)
                parties.append(f"Lessee: {refined_lessee}")

        # Extract Property from recitals
        m_prop = re.search(r'property\s+no\.?\s*(.*?)\s*\(hereinafter\s+referred\s+to\s+as\s+the\s+SAID\s+PROPERTY\)', full_text, re.DOTALL | re.I)
        if m_prop:
            raw_prop = clean_indian_legal_text(m_prop.group(1))
            parties.append(f"Property: {raw_prop}")

    else: # sale_deed
        m_seller = re.search(r'Name\s+of\s+the\s+seller\s*:\s*([^\n]+)', full_text, re.I)
        m_buyer = re.search(r'Name\s+of\s+the\s+buyer\s*:\s*([^\n]+)', full_text, re.I)
        seller_val = clean_indian_legal_text(m_seller.group(1)) if m_seller else ""
        buyer_val = clean_indian_legal_text(m_buyer.group(1)) if m_buyer else ""
        if seller_val and len(seller_val) > 2: parties.append(f"Vendor / Seller: {seller_val}")
        if buyer_val and len(buyer_val) > 2: parties.append(f"Purchaser / Buyer: {buyer_val}")
        if not parties:
            parties.append("Vendor / Seller: (Parties specified in execution schedule)")
            parties.append("Purchaser / Buyer: (Parties specified in execution schedule)")

    # Step 3: Extract Operative Covenants with clean numerical ordering
    clauses = {}
    if doc_type == "lease_deed":
        start_match = re.search(r'NOW THIS LEASE DEED WITNESS(?:ETH)? AS UNDER\s*:-?', full_text, re.I)
        end_match = re.search(r'SCHEDULE OF THE LEASED PROPERTY\s*:', full_text, re.I)
        if not end_match:
            end_match = re.search(r'IN WITNESS WHEREOF', full_text, re.I)

        covenants_text = full_text[start_match.end():end_match.start()].strip() if (start_match and end_match) else full_text
        raw_chunks = re.split(r'(?:\n\s*|^)(?=\d+\.\s+)', covenants_text)
        
        idx = 1
        for c in raw_chunks:
            clean_c = clean_indian_legal_text(re.sub(r'^\d+\.\s*', '', c).strip())
            if len(clean_c) > 15:
                clauses[f'clause_{idx}'] = clean_c
                idx += 1

    else: # sale_deed
        # Locate operative conditions of sale
        m_cond = re.search(r'(?:Conditions\s+of\s+Sale|Conditions\s+of\s+Transfer)\s*:?(.*?)(?:IN\s+WITNESS\s+WHEREOF|10\.\s+Any\s+other)', full_text, re.DOTALL | re.I)
        if m_cond:
            block = m_cond.group(1)
            raw_chunks = re.split(r'(?:\n\s*|^)(?=\([ivx0-9]+\)|\bThe\s+buyer\s+confirms\b|\bThe\s+seller\s+confirms\b)', block, flags=re.I)
            idx = 1
            for c in raw_chunks:
                clean_c = clean_indian_legal_text(c)
                if len(clean_c) > 20:
                    clauses[f'clause_{idx}'] = clean_c
                    idx += 1
        else:
            raw_chunks = re.split(r'(?:\n\s*|^)(?=\d+\.\s+)', full_text)
            idx = 1
            for c in raw_chunks:
                clean_c = clean_indian_legal_text(c)
                if len(clean_c) > 20:
                    clauses[f'clause_{idx}'] = clean_c
                    idx += 1

    return {
        "filename": filename,
        "doc_type": doc_type,
        "title": "Model Lease Deed" if is_lease else "Model Deed of Sale / Conveyance",
        "preamble": full_text[:600],
        "schedule": "",
        "clauses": clauses,
        "total_clauses": len(clauses),
        "parties": parties
    }


def index_covenants(clauses: Dict[str, str]):
    """Embed clauses and construct FAISS inner product (cosine) index."""
    embed_model, _, _, _, _, _ = get_models()
    clause_ids = list(clauses.keys())
    clause_texts = list(clauses.values())
    if not clause_texts:
        return None, None, {}, {}
    
    embeddings = embed_model.encode(clause_texts, normalize_embeddings=True).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    clause_to_id = {v: k for k, v in clauses.items()}
    clause_to_idx = {cid: idx for idx, cid in enumerate(clause_ids)}
    return index, embeddings, clause_to_id, clause_to_idx


FACET_KEYWORDS = {
    "DEPOSIT": {
        "primary": [
            "security deposit", "advance amount", "amount of advance", "advance/premium",
            "interest free advance", "earnest money", "refundable upon expiry",
            "refund of the deposit", "refund of deposit", "deposit refundable"
        ],
        "secondary": [
            "interest free", "refundable", "deduction", "deductions", 
            "forfeiture", "set-off", "claims if any of the lessor", "breaches on part of the lessee"
        ]
    },
    "RENT_PAYMENT": {
        "primary": [
            "monthly rent", "periodicity & mode of payment", "pay the rent", 
            "rent as agreed", "default in payment of rent", "rate of rent"
        ],
        "secondary": [
            "mode of payment", "calendar month", "7th day", "due date", 
            "cheque", "net banking", "zbanking", "neft", "rtgs", "escalation"
        ]
    },
    "TENURE_TERMINATION": {
        "primary": [
            "period of lease", "lease shall terminate", "term of the lease", "term agreed", 
            "lock-in period", "lock in", "notice to quit", "determination of lease", 
            "surrender of lease", "vacated before", "premises vacated"
        ],
        "secondary": [
            "option to renew", "renewal", "notice period", "prior written notice", 
            "sooner determination", "expiry of the lease", "one month notice"
        ]
    },
    "UTILITIES_MAINTENANCE": {
        "primary": [
            "electricity and water", "consumption charges", "electricity charges", 
            "water consumption", "utility charges"
        ],
        "secondary": [
            "electricity", "water", "meter reading", "power", 
            "society maintenance", "common area", "maintenance charges"
        ]
    },
    "TAXES_OUTGOINGS": {
        "primary": [
            "property/ house tax", "house tax", "property tax", "municipal tax", 
            "tax liability", "proportionate increase in the tax"
        ],
        "secondary": [
            "cess", "rates and taxes", "municipal outgoings", "local taxes"
        ]
    },
    "INSPECTION_POSSESSION": {
        "primary": [
            "inspect the said property", "right to inspect", "re-entry", 
            "quiet enjoyment", "peaceful possession", "vacant physical possession"
        ],
        "secondary": [
            "reasonable time during the day", "condition of the building", 
            "fixtures and fittings", "inspection"
        ]
    },
    "USE_RESTRICTIONS": {
        "primary": [
            "legally permitted use", "prohibited use", "sub-let", "sub-lease", 
            "subletting", "underlet", "assign the tenancy", "lawful purpose"
        ],
        "secondary": [
            "permitted use", "residential purpose", "commercial purpose"
        ]
    },
    "REPAIRS_ALTERATION": {
        "primary": [
            "structural changes", "structural alteration", "structural addition", 
            "damage to the property", "major repairs", "white washing"
        ],
        "secondary": [
            "repairs", "maintenance", "tenantable condition", "good condition"
        ]
    },
    "STAMP_REGISTRATION": {
        "primary": [
            "stamp duty paid", "stamp duty", "stamp paper", 
            "registration charges", "registration fee"
        ],
        "secondary": [
            "sub-registrar", "indian stamp act", "registration act"
        ]
    },
    "TITLE_ENCUMBRANCE": {
        "primary": [
            "free from all encumbrances", "covenant for title", "defect in title", 
            "clear and marketable title", "sole/ joint/ co-owner"
        ],
        "secondary": [
            "mortgage", "charge", "lien", "lis pendens", "title deeds", 
            "indemnify", "indemnity", "chain of documents"
        ]
    }
}

COMPATIBLE_FACET_PAIRS = {
    # Intra-facet (same topic)
    ("DEPOSIT", "DEPOSIT"),
    ("RENT_PAYMENT", "RENT_PAYMENT"),
    ("TENURE_TERMINATION", "TENURE_TERMINATION"),
    ("UTILITIES_MAINTENANCE", "UTILITIES_MAINTENANCE"),
    ("TAXES_OUTGOINGS", "TAXES_OUTGOINGS"),
    ("INSPECTION_POSSESSION", "INSPECTION_POSSESSION"),
    ("USE_RESTRICTIONS", "USE_RESTRICTIONS"),
    ("REPAIRS_ALTERATION", "REPAIRS_ALTERATION"),
    ("TITLE_ENCUMBRANCE", "TITLE_ENCUMBRANCE"),
    ("STAMP_REGISTRATION", "STAMP_REGISTRATION"),

    # Direct legal tension pairs in Indian real estate:
    ("INSPECTION_POSSESSION", "USE_RESTRICTIONS"),
    ("TENURE_TERMINATION", "INSPECTION_POSSESSION")
}

def get_clause_facets(text: str) -> List[str]:
    """Classifies an Indian contract covenant into primary and secondary functional facets."""
    lower = text.lower()
    scores = {}
    for facet, kw_groups in FACET_KEYWORDS.items():
        score = 0
        for kw in kw_groups.get("primary", []):
            if kw in lower:
                score += 5
        for kw in kw_groups.get("secondary", []):
            if kw in lower:
                score += 1
        if score > 0:
            scores[facet] = score

    if not scores:
        return ["GENERAL_COVENANT"]
    
    sorted_facets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_facets[0][1]
    
    selected = [sorted_facets[0][0]]
    for facet, sc in sorted_facets[1:]:
        if sc >= 3 and sc >= 0.6 * top_score:
            selected.append(facet)
    return selected

def are_facets_compatible(facets_a: List[str], facets_b: List[str]) -> bool:
    """Returns True if the two clauses share the same facet or a legally clashing facet pair."""
    for fa in facets_a:
        for fb in facets_b:
            if (fa, fb) in COMPATIBLE_FACET_PAIRS or (fb, fa) in COMPATIBLE_FACET_PAIRS:
                return True
    return False


def detect_contradictions(clauses: Dict[str, str], index, embeddings, clause_to_id, clause_to_idx, top_k_exclude: int = 0) -> List[Dict[str, Any]]:
    """
    Domain-Gated Bidirectional Contradiction Detection:
    - First gates candidates by Legal Facet (preventing cross-topic clashes like Utilities vs Termination)
    - Bidirectional NLI using calibrated thresholds:
      - Theta >= 0.95 -> HIGH_CERTAINTY (Direct Mutual Legal Exclusion)
      - 0.85 <= Theta < 0.95 -> POTENTIAL_TENSION (Operational Inconsistency)
    - Filters out scores < 0.85
    - Deduplicates unordered pairs (clause_A, clause_B)
    """
    _, nli_tokenizer, nli_model, device, id2label, contra_idx = get_models()
    clause_ids = list(clauses.keys())
    seen_pairs = set()
    found_contradictions = []

    # Pre-classify all clauses into legal facets
    clause_facets = {cid: get_clause_facets(txt) for cid, txt in clauses.items()}

    # Pairwise evaluation across domain-compatible covenants
    for i in range(len(clause_ids)):
        cid = clause_ids[i]
        text = clauses[cid]
        fa = clause_facets.get(cid, ["GENERAL_COVENANT"])

        for j in range(i + 1, len(clause_ids)):
            cand_id = clause_ids[j]
            cand_text = clauses[cand_id]
            fb = clause_facets.get(cand_id, ["GENERAL_COVENANT"])

            # DOMAIN-GATING: Skip orthogonal clauses that belong to unrelated facets
            if not are_facets_compatible(fa, fb):
                continue

            pair_key = tuple(sorted([cid, cand_id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Bidirectional NLI on unpadded pair
            in1 = nli_tokenizer(text, cand_text, truncation=True, max_length=256, return_tensors="pt")
            in1 = {k: v.to(device) for k, v in in1.items()}
            with torch.no_grad():
                p1 = F.softmax(nli_model(**in1).logits, dim=-1)[0]
            lbl1 = id2label[int(p1.argmax().item())]
            sc1 = float(p1[contra_idx].item())

            in2 = nli_tokenizer(cand_text, text, truncation=True, max_length=256, return_tensors="pt")
            in2 = {k: v.to(device) for k, v in in2.items()}
            with torch.no_grad():
                p2 = F.softmax(nli_model(**in2).logits, dim=-1)[0]
            lbl2 = id2label[int(p2.argmax().item())]
            sc2 = float(p2[contra_idx].item())

            max_sc = max(sc1, sc2)
            winning_lbl = lbl1 if sc1 >= sc2 else lbl2
            direction = f"{cid} -> {cand_id}" if sc1 >= sc2 else f"{cand_id} -> {cid}"

            if ("contradiction" in winning_lbl) and (max_sc >= 0.85):
                tier = "HIGH_CERTAINTY" if max_sc >= 0.95 else "POTENTIAL_TENSION"
                found_contradictions.append({
                    "clause_a_id": cid,
                    "clause_a_text": text,
                    "clause_b_id": cand_id,
                    "clause_b_text": cand_text,
                    "facet_a": fa[0],
                    "facet_b": fb[0],
                    "contradiction_score": round(max_sc, 4),
                    "confidence_tier": tier,
                    "winning_direction": direction,
                    "diagnostic_note": (
                        f"Direct mutual exclusion in [{fa[0]} vs {fb[0]}] (Calibrated Theta >= 0.95)"
                        if tier == "HIGH_CERTAINTY" else
                        f"Operational friction in [{fa[0]} vs {fb[0]}] (Calibrated 0.85 <= Theta < 0.95)"
                    )
                })

    found_contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
    return found_contradictions


def profile_clause_ambiguity(clause_text: str) -> List[Dict[str, str]]:
    """
    Contextual Ambiguity Profiler:
    Detects subjective standards, undefined metrics, and uncalibrated windows under Indian statutory law.
    Skips bounded controls (e.g. clauses bounded by 24 hours or explicit mathematical percentages).
    """
    lower = clause_text.lower()
    unbounded_triggers = {
        "reasonable time": {
            "flaw": "Subjective inspection / entry window lacking statutory notice period under Section 15 of Model Tenancy Act, 2021 (mandatory 24-hr advance written notice).",
            "statute": "Model Tenancy Act, 2021 § 15"
        },
        "condition of the building": {
            "flaw": "Subjective inspection scope without an agreed baseline fixtures and fittings inventory schedule.",
            "statute": "Model Tenancy Act, 2021 § 15 / TPA 1882 § 108(m)"
        },
        "regular intervals": {
            "flaw": "Undefined temporal maintenance cadence; creates uncertainty regarding periodic inspection and repair liabilities.",
            "statute": "Model Tenancy Act, 2021 Schedule II"
        },
        "tenantable condition": {
            "flaw": "Undefined subjective threshold for interior maintenance; prone to disputes upon surrender of tenancy.",
            "statute": "Transfer of Property Act, 1882 § 108(m)"
        },
        "market trends": {
            "flaw": "Unbounded economic standard lacking objective mathematical escalation formula or linkage to RBI CPI/WPI index.",
            "statute": "Model Tenancy Act, 2021 § 9 & 10"
        },
        "satisfactory progress": {
            "flaw": "Subjective condition precedent without independent registered architect or structural engineer certification.",
            "statute": "Indian Contract Act, 1872 § 31"
        },
        "amicably as soon as": {
            "flaw": "Pathological dispute resolution covenant lacking clear institutional arbitration rules, seat, and legal jurisdiction.",
            "statute": "Arbitration and Conciliation Act, 1996 § 7 & 20"
        },
        "pristine condition": {
            "flaw": "Undefined handover condition conflicting with ordinary wear and tear exceptions protected under law.",
            "statute": "Transfer of Property Act, 1882 § 108(m)"
        }
    }

    detected = []
    for trigger, meta in unbounded_triggers.items():
        if trigger in lower:
            # Check for adversarial bounding qualifiers
            if "not less than twenty-four" in lower or "at least 24 hours" in lower or "exactly seven percent" in lower or "capped at" in lower:
                continue
            detected.append({
                "vague_term": trigger,
                "flaw_description": meta["flaw"],
                "statutory_reference": meta["statute"],
                "diagnostic_severity": "MEDIUM"
            })
    return detected


def detect_all_ambiguities(clauses: Dict[str, str]) -> List[Dict[str, Any]]:
    """Scan all covenants for contextual ambiguities."""
    results = []
    for cid, text in clauses.items():
        ambs = profile_clause_ambiguity(text)
        if ambs:
            results.append({
                "clause_id": cid,
                "clause_text": text,
                "ambiguities": ambs
            })
    return results


def check_statutory_omissions(clauses: Dict[str, str], doc_type: str) -> List[Dict[str, Any]]:
    """
    Checks the document against mandatory statutory covenants and market benchmarks:
    - Model Tenancy Act, 2021 (for Lease Deeds)
    - Transfer of Property Act, 1882 & Registration Act, 1908 (for Sale Deeds)
    """
    all_text = " ".join(clauses.values()).lower()
    omissions = []

    if doc_type == "lease_deed":
        # 1. 24-Hour Notice for Inspection
        if not re.search(r'(?:24\s*hours?|twenty-four\s*hours?|one\s*day)\s*(?:prior|advance)?\s*notice', all_text):
            omissions.append({
                "covenant_name": "24-Hour Prior Written Notice for Landlord Entry",
                "statutory_mandate": "Model Tenancy Act, 2021 — Section 15",
                "severity": "HIGH",
                "diagnostic_observation": "The lease authorizes lessor inspection or entry without explicitly mandating 24-hour advance written notice specifying date and time, creating privacy and quiet enjoyment risks."
            })
        
        # 2. Security Deposit Refund Window
        if not re.search(r'deposit.*(?:refund|return).*(?:within|\d+\s*days|one\s*month)', all_text):
            omissions.append({
                "covenant_name": "Definite Security Deposit Refund Timeline",
                "statutory_mandate": "Model Tenancy Act, 2021 — Section 13",
                "severity": "MEDIUM",
                "diagnostic_observation": "No strict calendar timeline (e.g. within 30 days of vacant possession) is stipulated for the refund of security deposit after legitimate deductions."
            })

        # 3. Force Majeure & Frustration of Lease
        if not re.search(r'force\s+majeure|act\s+of\s+god|unforeseen\s+calamity|frustration', all_text):
            omissions.append({
                "covenant_name": "Express Force Majeure and Rent Abatement Clause",
                "statutory_mandate": "Indian Contract Act, 1872 — Section 56 & TPA § 108(e)",
                "severity": "MEDIUM",
                "diagnostic_observation": "Deed lacks an express Force Majeure mechanism for rent suspension or termination in the event of natural disaster, structural collapse, or government lockdown."
            })

        # 4. Clear Structural vs Non-Structural Division
        if not re.search(r'structural\s+repair', all_text):
            omissions.append({
                "covenant_name": "Structural vs Day-to-Day Maintenance Division",
                "statutory_mandate": "Model Tenancy Act, 2021 — Second Schedule",
                "severity": "LOW",
                "diagnostic_observation": "Deed does not explicitly apportion major structural repairs (landlord) versus minor day-to-day electrical/sanitary replacements (tenant)."
            })

    else: # sale_deed
        # 1. Title Indemnity & Encumbrance Free Warranty
        if not re.search(r'indemnif(?:y|ication)|free\s+from\s+all\s+encumbrances', all_text):
            omissions.append({
                "covenant_name": "Covenant for Title & Express Indemnity for Defect in Title",
                "statutory_mandate": "Transfer of Property Act, 1882 — Section 55(2)",
                "severity": "HIGH",
                "diagnostic_observation": "Deed lacks an express covenant of title indemnity protecting the purchaser against prior undisclosed mortgages, charges, or litigations."
            })

        # 2. Handover of Original Chain of Title Deeds
        if not re.search(r'title\s+deeds?|chain\s+of\s+documents?|original\s+documents?', all_text):
            omissions.append({
                "covenant_name": "Handover of Prior Title Deeds & Link Documents",
                "statutory_mandate": "Transfer of Property Act, 1882 — Section 55(3)",
                "severity": "MEDIUM",
                "diagnostic_observation": "No express obligation on the vendor to deliver all prior original title and revenue link deeds to the buyer upon full payment of consideration."
            })

        # 3. Apportionment of Pre-Execution Dues & Taxes
        if not re.search(r'taxes|cess|rates|revenue\s+charges', all_text):
            omissions.append({
                "covenant_name": "Tax & Municipal Outgoings Apportionment Clause",
                "statutory_mandate": "Transfer of Property Act, 1882 — Section 55(1)(g)",
                "severity": "MEDIUM",
                "diagnostic_observation": "Deed does not clearly specify that the seller remains solely liable for all municipal taxes, revenue cess, and electricity dues accrued up to the date of execution."
            })

    return omissions


def perform_full_document_audit(pdf_input: Any) -> Dict[str, Any]:
    """
    Executes a complete macro-scan of the property contract:
    - Covenant Extraction & Normalization
    - FAISS Vector Indexing
    - Calibrated Contradiction Detection
    - Contextual Ambiguity Profiling
    - Statutory Omission & Gap Analysis
    - Risk Tiering & Scorecard Computation
    """
    doc_data = extract_contract(pdf_input)
    clauses = doc_data["clauses"]
    total_clauses = len(clauses)

    if total_clauses == 0:
        return {
            "status": "error",
            "message": "No covenants could be parsed from the document.",
            "doc_data": doc_data
        }

    # Indexing
    index, embeddings, clause_to_id, clause_to_idx = index_covenants(clauses)

    # Audits (with Domain-Gated Contradiction Filter)
    contradictions = detect_contradictions(clauses, index, embeddings, clause_to_id, clause_to_idx, top_k_exclude=0)
    ambiguities = detect_all_ambiguities(clauses)
    omissions = check_statutory_omissions(clauses, doc_data["doc_type"])

    # Count high certainty and potential tension
    high_certainty_contra = [c for c in contradictions if c["confidence_tier"] == "HIGH_CERTAINTY"]
    potential_tension_contra = [c for c in contradictions if c["confidence_tier"] == "POTENTIAL_TENSION"]

    # Compute per-clause risk profile
    clause_risk_profile = {}
    for cid in clauses:
        # Check if involved in high contradiction
        involved_in_high_contra = any(c["clause_a_id"] == cid or c["clause_b_id"] == cid for c in high_certainty_contra)
        involved_in_tension = any(c["clause_a_id"] == cid or c["clause_b_id"] == cid for c in potential_tension_contra)
        has_ambiguity = any(a["clause_id"] == cid for a in ambiguities)

        if involved_in_high_contra:
            lvl = "HIGH"
        elif involved_in_tension or has_ambiguity:
            lvl = "MEDIUM"
        else:
            lvl = "LOW"
        clause_risk_profile[cid] = lvl

    high_risk_clauses = [cid for cid, r in clause_risk_profile.items() if r == "HIGH"]
    medium_risk_clauses = [cid for cid, r in clause_risk_profile.items() if r == "MEDIUM"]
    low_risk_clauses = [cid for cid, r in clause_risk_profile.items() if r == "LOW"]

    scorecard = {
        "total_clauses_scanned": total_clauses,
        "high_risk_count": len(high_risk_clauses),
        "medium_risk_count": len(medium_risk_clauses),
        "low_risk_count": len(low_risk_clauses),
        "high_certainty_contradictions": len(high_certainty_contra),
        "potential_tensions": len(potential_tension_contra),
        "ambiguity_clauses_count": len(ambiguities),
        "statutory_omissions_count": len(omissions),
        "overall_health_score": max(0, 100 - (len(high_risk_clauses) * 25 + len(potential_tension_contra) * 10 + len(ambiguities) * 8 + len(omissions) * 10))
    }

    memo_markdown = generate_due_diligence_memo_markdown(
        doc_data, scorecard, contradictions, ambiguities, omissions, clause_risk_profile
    )

    return {
        "status": "success",
        "doc_data": doc_data,
        "scorecard": scorecard,
        "contradictions": contradictions,
        "ambiguities": ambiguities,
        "omissions": omissions,
        "clause_risk_profile": clause_risk_profile,
        "memo_markdown": memo_markdown
    }


def generate_due_diligence_memo_markdown(
    doc_data: Dict[str, Any],
    scorecard: Dict[str, Any],
    contradictions: List[Dict[str, Any]],
    ambiguities: List[Dict[str, Any]],
    omissions: List[Dict[str, Any]],
    clause_risk_profile: Dict[str, str]
) -> str:
    """Formats the comprehensive Due Diligence Memorandum in clean GitHub-style Markdown."""
    lines = []
    
    # Title & Metadata
    lines.append("# DUE DILIGENCE LEGAL AUDIT MEMORANDUM")
    lines.append(f"**Document Name:** `{doc_data['filename']}`  ")
    lines.append(f"**Instrument Classification:** {doc_data['title']}  ")
    lines.append(f"**Statutory Framework:** Transfer of Property Act, 1882; Indian Contract Act, 1872; Model Tenancy Act, 2021; Registration Act, 1908  ")
    if doc_data.get("parties"):
        lines.append(f"**Identified Parties:** {'; '.join(doc_data['parties'])}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Scorecard
    lines.append("## 1. Executive Audit Scorecard & Risk Profile")
    lines.append("")
    lines.append(f"| Metric | Result | Audit Interpretation |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **Overall Health Index** | **{scorecard['overall_health_score']} / 100** | {'CRITICAL ATTENTION REQUIRED' if scorecard['overall_health_score'] < 60 else ('MODERATE RISK DETECTED' if scorecard['overall_health_score'] < 85 else 'STANDARD ALIGNMENT')} |")
    lines.append(f"| **Total Covenants Analyzed** | `{scorecard['total_clauses_scanned']}` | Complete covenant schedule extracted |")
    lines.append(f"| **High Risk Covenants** | `{scorecard['high_risk_count']}` | Clauses with direct contradiction clashes |")
    lines.append(f"| **Medium Risk Covenants** | `{scorecard['medium_risk_count']}` | Clauses with ambiguities or operational tensions |")
    lines.append(f"| **Standard / Low Risk** | `{scorecard['low_risk_count']}` | Standard aligned covenants |")
    lines.append(f"| **Calibrated Contradictions** | `{scorecard['high_certainty_contradictions']} High Certainty, {scorecard['potential_tensions']} Tension` | Verified via calibrated NLI (θ ≥ 0.85) |")
    lines.append(f"| **Contextual Ambiguities** | `{scorecard['ambiguity_clauses_count']}` | Subjective / undefined triggers |")
    lines.append(f"| **Statutory Omission Gaps** | `{scorecard['statutory_omissions_count']}` | Missing mandatory / market covenants |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 2: Contradictions Register
    lines.append("## 2. Contradictions & Conflict Register")
    if not contradictions:
        lines.append("> [!NOTE]\n> **No calibrated contradictions detected.** All covenant pairs tested remained below the calibrated tension threshold (θ < 0.85).")
    else:
        lines.append(f"A total of **{len(contradictions)} conflicting covenant pair(s)** were identified applying the benchmark-calibrated threshold.")
        lines.append("")
        for idx, item in enumerate(contradictions, 1):
            badge = "🔴 **HIGH CERTAINTY (θ ≥ 0.95)**" if item["confidence_tier"] == "HIGH_CERTAINTY" else "🟡 **POTENTIAL TENSION (0.85 ≤ θ < 0.95)**"
            lines.append(f"### Conflict #{idx}: `{item['clause_a_id']}` ⟷ `{item['clause_b_id']}`")
            lines.append(f"- **Confidence Tier:** {badge}")
            lines.append(f"- **Contradiction Score:** `{item['contradiction_score']}` (Dominant Flow: `{item['winning_direction']}`)")
            lines.append(f"- **Diagnostic Finding:** {item['diagnostic_note']}")
            lines.append("")
            lines.append(f"| Covenant | Clause Text |")
            lines.append(f"| :--- | :--- |")
            lines.append(f"| **`{item['clause_a_id']}`** | *\"{item['clause_a_text']}\"* |")
            lines.append(f"| **`{item['clause_b_id']}`** | *\"{item['clause_b_text']}\"* |")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Section 3: Contextual Ambiguity Catalog
    lines.append("## 3. Contextual Ambiguity & Vagueness Catalog")
    if not ambiguities:
        lines.append("> [!NOTE]\n> **No subjective or unbounded triggers flagged.** All clauses evaluated contain bounded metrics.")
    else:
        lines.append("The following covenants contain subjective language or undefined operational standards under Indian property law:")
        lines.append("")
        lines.append("| Clause ID | Vague Term / Trigger | Statutory Flaw & Legal Risk | Governing Statute |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for amb_item in ambiguities:
            cid = amb_item["clause_id"]
            for a in amb_item["ambiguities"]:
                lines.append(f"| **`{cid}`** | `{a['vague_term']}` | {a['flaw_description']} | *{a['statutory_reference']}* |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 4: Statutory Omission & Gap Analysis
    lines.append("## 4. Statutory Compliance & Omission Gap Analysis")
    if not omissions:
        lines.append("> [!NOTE]\n> **No statutory omission gaps detected.** Document incorporates all baseline statutory covenants.")
    else:
        lines.append("The following standard protective covenants and statutory mandates are **absent** from this draft:")
        lines.append("")
        for om in omissions:
            lines.append(f"- **{om['covenant_name']}** (`{om['severity']} PRIORITY`)")
            lines.append(f"  - **Statutory Mandate:** {om['statutory_mandate']}")
            lines.append(f"  - **Diagnostic Impact:** {om['diagnostic_observation']}")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Section 5: Full Covenants Schedule
    lines.append("## 5. Complete Covenants Schedule")
    lines.append("| Clause ID | Diagnostic Risk | Clause Text Excerpt |")
    lines.append("| :--- | :--- | :--- |")
    for cid, text in doc_data["clauses"].items():
        risk = clause_risk_profile.get(cid, "LOW")
        risk_emoji = "🔴 HIGH" if risk == "HIGH" else ("🟡 MEDIUM" if risk == "MEDIUM" else "🟢 LOW")
        excerpt = text[:120] + "..." if len(text) > 120 else text
        lines.append(f"| **`{cid}`** | {risk_emoji} | {excerpt} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 6: Professional Discretion & Zero-Remediation Notice
    lines.append("## 6. Mandatory Advocate Discretion & Zero-Remediation Notice")
    lines.append("> [!CAUTION]\n> **STRICT DIAGNOSTIC PURPOSE ONLY:**  \n"
                 "> This Due Diligence Memorandum is generated automatically by ClauseIQ as a diagnostic issue-spotting co-pilot.  \n"
                 "> - **No Legal Advice:** This memorandum does not constitute legal opinion, representation, or formal legal advice.  \n"
                 "> - **Zero-Remediation Policy:** ClauseIQ does NOT draft substitute legal covenants, prescribe mandatory revisions, or dictate commercial terms.  \n"
                 "> - **Advocate Discretion:** All findings, contradiction scores, and ambiguity flags must be independently evaluated by a qualified Advocate or legal practitioner in the relevant jurisdiction. Contractual restructuring, negotiation leverage, and legal remedies remain entirely under the professional discretion of consulting legal counsel.")

    return "\n".join(lines)


def answer_chatbot_query(
    query: str,
    audit_results: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Context-aware diagnostic chatbot grounded in the document text and Due Diligence Memorandum.
    Strictly complies with the Zero-Remediation and Advocate Discretion rules.
    """
    if not audit_results or audit_results.get("status") != "success":
        return "Please load and analyze a contract document first before asking questions."

    query_lower = query.lower().strip()
    clauses = audit_results["doc_data"]["clauses"]
    contradictions = audit_results.get("contradictions", [])
    ambiguities = audit_results.get("ambiguities", [])
    omissions = audit_results.get("omissions", [])
    doc_type = audit_results["doc_data"]["doc_type"]
    scorecard = audit_results.get("scorecard", {})

    # ZERO-REMEDIATION / DRAFTING REFUSAL CHECK
    drafting_triggers = [
        "rewrite", "rephrase", "draft", "fix this clause", "give me a replacement",
        "redraft", "suggest text", "how should i amend", "how to write", "provide a clause",
        "give me a clause", "give me the legal advice"
    ]
    if any(dt in query_lower for dt in drafting_triggers):
        return (
            "### ⚖️ Professional Advocate Discretion Notice\n\n"
            "Under ClauseIQ's **Zero-Remediation Policy**, this system is strictly a diagnostic issue-spotting tool "
            "and **does not draft substitute contractual covenants or prescribe legal remedies**.\n\n"
            "**Diagnostic Guidance for Consulting Counsel:**\n"
            "- ClauseIQ has pinpointed the legal friction and statutory reference in the Memorandum.\n"
            "- Rather than adopting an automated clause, your retained legal advocate should decide the commercial strategy "
            "(e.g., whether to strike the conflicting covenant, incorporate a 24-hour notice window under Section 15 of the Model Tenancy Act, "
            "or negotiate a mutual cap on liabilities).\n\n"
            "*All redrafting and negotiation strategy remain solely within the professional discretion of your legal advocate.*"
        )

    # 1. Specific Clause Query (e.g. "Clause 17", "explain clause 15")
    clause_match = re.search(r'\bclause[_\s]*(\d+)\b', query_lower)
    if clause_match:
        c_num = clause_match.group(1)
        cid = f"clause_{c_num}"
        if cid in clauses:
            clause_text = clauses[cid]
            # Find related contradictions
            rel_contra = [
                c for c in contradictions 
                if c["clause_a_id"] == cid or c["clause_b_id"] == cid
            ]
            # Find ambiguities
            rel_amb = [a for a in ambiguities if a["clause_id"] == cid]
            
            resp = [f"### 📋 Diagnostic Profile: `{cid}`\n"]
            resp.append(f"> **Covenant Text:**\n> \"{clause_text}\"\n")
            
            if rel_contra:
                resp.append(f"#### ⚠️ Calibrated Contradictions Flagged ({len(rel_contra)}):")
                for c in rel_contra:
                    other_id = c["clause_b_id"] if c["clause_a_id"] == cid else c["clause_a_id"]
                    tier_badge = "🔴 HIGH CERTAINTY" if c["confidence_tier"] == "HIGH_CERTAINTY" else "🟡 POTENTIAL TENSION"
                    resp.append(
                        f"- **Clash with `{other_id}`** ({tier_badge}, Score: `{c['contradiction_score']}`)\n"
                        f"  - *Conflicting Text:* \"{clauses[other_id]}\"\n"
                        f"  - *Finding:* {c['diagnostic_note']}"
                    )
                resp.append("")
            else:
                resp.append("- **Contradictions:** No calibrated contradictions detected for this clause.")

            if rel_amb:
                resp.append(f"#### 🔍 Contextual Ambiguities Flagged ({len(rel_amb[0]['ambiguities'])}):")
                for a in rel_amb[0]["ambiguities"]:
                    resp.append(
                        f"- **Vague Term:** `{a['vague_term']}`\n"
                        f"  - *Statutory Flaw:* {a['flaw_description']}\n"
                        f"  - *Statute:* {a['statutory_reference']}"
                    )
                resp.append("")
            else:
                resp.append("- **Ambiguities:** No subjective triggers flagged.")

            resp.append("\n> **Advocate Note:** Verify whether this covenant aligns with client operational requirements.")
            return "\n".join(resp)

    # 2. Contradiction Summary Query
    if any(w in query_lower for w in ["contradict", "conflict", "clash", "tension", "inconsistent"]):
        if not contradictions:
            return "No calibrated contradictions (θ ≥ 0.85) were detected across any of the covenants in this document."
        resp = [f"### ⚖️ Calibrated Contradictions in `{audit_results['doc_data']['filename']}`\n"]
        resp.append(f"Found **{len(contradictions)} conflicting covenant pair(s)**:\n")
        for idx, c in enumerate(contradictions, 1):
            badge = "🔴 HIGH CERTAINTY" if c["confidence_tier"] == "HIGH_CERTAINTY" else "🟡 POTENTIAL TENSION"
            resp.append(
                f"**{idx}. `{c['clause_a_id']}` vs `{c['clause_b_id']}`** — {badge} (Score: `{c['contradiction_score']}`)\n"
                f"- **`{c['clause_a_id']}`**: \"{c['clause_a_text'][:100]}...\"\n"
                f"- **`{c['clause_b_id']}`**: \"{c['clause_b_text'][:100]}...\"\n"
                f"- *Analysis:* {c['diagnostic_note']}\n"
            )
        return "\n".join(resp)

    # 3. Ambiguity / Vague Terms Query
    if any(w in query_lower for w in ["ambig", "vague", "unclear", "undefined", "subjective"]):
        if not ambiguities:
            return "No subjective or unbounded terms were flagged in this document."
        resp = [f"### 🔍 Ambiguous & Subjective Terms in `{audit_results['doc_data']['filename']}`\n"]
        for item in ambiguities:
            cid = item["clause_id"]
            for a in item["ambiguities"]:
                resp.append(
                    f"- **`{cid}`** contains term **`{a['vague_term']}`**:\n"
                    f"  - *Issue:* {a['flaw_description']}\n"
                    f"  - *Governing Law:* {a['statutory_reference']}\n"
                )
        return "\n".join(resp)

    # 4. Missing Covenants / Omissions Query
    if any(w in query_lower for w in ["missing", "omission", "gap", "statutory", "mta", "tenancy act", "tpa"]):
        if not omissions:
            return "No standard statutory omissions were detected in this document."
        resp = [f"### 📋 Statutory Compliance & Omission Gaps\n"]
        for om in omissions:
            resp.append(
                f"- **{om['covenant_name']}** (`{om['severity']}` Priority)\n"
                f"  - *Mandate:* {om['statutory_mandate']}\n"
                f"  - *Diagnostic Impact:* {om['diagnostic_observation']}\n"
            )
        return "\n".join(resp)

    # 5. Scorecard / Overall Health
    if any(w in query_lower for w in ["summary", "scorecard", "health", "overview", "score", "risk"]):
        return (
            f"### 📊 Contract Audit Overview\n\n"
            f"- **Overall Health Score:** `{scorecard['overall_health_score']}/100`\n"
            f"- **Total Covenants:** `{scorecard['total_clauses_scanned']}`\n"
            f"- **High Risk Clauses:** `{scorecard['high_risk_count']}`\n"
            f"- **Medium Risk Clauses:** `{scorecard['medium_risk_count']}`\n"
            f"- **Low Risk Clauses:** `{scorecard['low_risk_count']}`\n"
            f"- **Contradictions Flagged:** `{scorecard['high_certainty_contradictions']} High Certainty, {scorecard['potential_tensions']} Tension`\n"
            f"- **Contextual Ambiguities:** `{scorecard['ambiguity_clauses_count']}`\n"
            f"- **Statutory Gaps:** `{scorecard['statutory_omissions_count']}`\n\n"
            f"Refer to the Due Diligence Memorandum tabs on the left for the full breakdown."
        )

    # Generic Fallback: Search relevant clause by semantic similarity
    embed_model, _, _, _, _, _ = get_models()
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype('float32')
    index, _, _, clause_to_idx = index_covenants(clauses)
    scores, indices = index.search(q_vec, 3)
    
    resp = [f"### 🔎 Analysis for query: *\"{query}\"*\n"]
    resp.append("Here are the most relevant covenants from the active document:\n")
    clause_ids = list(clauses.keys())
    for sc, idx_val in zip(scores[0], indices[0]):
        if idx_val >= 0:
            cid = clause_ids[idx_val]
            risk = audit_results.get("clause_risk_profile", {}).get(cid, "LOW")
            resp.append(f"- **`{cid}`** (Similarity: `{float(sc):.3f}`, Risk: `{risk}`):\n  > \"{clauses[cid]}\"\n")
    
    resp.append("\n*Tip: You can ask specific questions like 'Why is Clause 17 flagged?' or 'What are the missing covenants?'*")
    return "\n".join(resp)
