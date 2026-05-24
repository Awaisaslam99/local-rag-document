import os
import re
import json
import numpy as np
import torch
import faiss
from pypdf import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import datetime
import argparse


# ---------------------------------------------------------------------------
# 1. DOCUMENT INGESTION
# ---------------------------------------------------------------------------

class DocumentIngestor:
    """
    Reads PDF and TXT files from a folder path or individual file objects
    (for Streamlit compatibility) and returns clean raw text per filename.
    """

    def __init__(self, folder_path=None):
        self.folder_path = folder_path
        self.raw_documents = {}

    def extract_text_from_pdf(self, pdf_file_or_path):
        try:
            reader = PdfReader(pdf_file_or_path)
            extracted_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

            # --- Repair character-spaced PDFs ---
            repaired_lines = []
            for line in extracted_text.splitlines():
                # Normalize internal runs of spaces for token counting only
                tokens = line.split()
                if not tokens:
                    continue
                single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
                if single_char_ratio > 0.7:
                    # Split on 2+ spaces = word boundaries, then collapse each word
                    words = re.split(r'  +', line)
                    collapsed = ["".join(w.split()) for w in words if w.strip()]
                    repaired_lines.append(" ".join(collapsed))
                else:
                    # Normal line — just collapse internal whitespace
                    repaired_lines.append(" ".join(tokens))

            # --- Split fused name+title lines ---
            split_lines = []
            for line in repaired_lines:
                fixed = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1\n\2', line)
                split_lines.append(fixed)

            return "\n".join(split_lines)

        except Exception as e:
            print(f"[ERROR] Could not read PDF: {e}")
            return ""

    def process_folder(self):
        """
        Walk the configured folder and ingest all PDF and TXT files.
        Returns dict of {filename: raw_text}.
        """
        if not self.folder_path or not os.path.exists(self.folder_path):
            print(f"[WARNING] Folder not found: {self.folder_path}")
            return self.raw_documents

        for filename in sorted(os.listdir(self.folder_path)):
            file_path = os.path.join(self.folder_path, filename)
            if not os.path.isfile(file_path):
                continue

            if filename.lower().endswith(".pdf"):
                self.raw_documents[filename] = self.extract_text_from_pdf(file_path)
            elif filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    self.raw_documents[filename] = " ".join(f.read().split())

        return self.raw_documents


# ---------------------------------------------------------------------------
# 2. DOCUMENT CLASSIFICATION + FIELD EXTRACTION
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    Classifies each document into one of:
      Invoice | Resume | Utility Bill | Other | Unclassifiable

    Uses a two-stage approach:
      Stage 1 — Scored keyword heuristic (fast, avoids loading the model
                 for obviously clear documents).
      Stage 2 — Zero-shot BART classifier for ambiguous cases.

    This is intentionally general: no hardcoded or doc-specific
    patterns. All regex patterns target structural markers that appear across
    any standard invoice, resume, or utility bill.
    """

    CATEGORIES = ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]

    # Keyword sets used for Stage 1 scoring.
    KEYWORD_SIGNALS = {
        "Invoice": [
            r"\binvoice\s*(no|number|#|date|to)\b",
            r"\bbill\s+to\b",
            r"\btotal\s+due\b",
            r"\bamount\s+due\b",
            r"\bpayment\s+terms\b",
            r"\bsubtotal\b",
            r"\bitemized\b",
        ],
        "Resume": [
            r"\bskills\b",
            r"\bexperience\b",
            r"\beducation\b",
            r"\bcertification",
            r"\bcurriculum\s+vitae\b",
            r"\b(linkedin|github)\b",
            r"\bsummary\b",
            r"\bachievements\b",
            r"\bprojects?\b",
            r"\bwork\s+history\b",
            r"\bemployment\b",
            r"\breferences?\b",
            r"\bintern(ship)?\b",
             r"\bcandidate\b",
            r"\bprofessional\s+experience\b",
            r"\btechnical\s+skills\b",
            r"\bwork\s+experience\b",
            r"\blanguages\b",       
            r"\bframeworks\b",       
            r"\bdatabases\b",        
        ],
        "Utility Bill": [
            r"\butility\b",
            r"\bkwh\b",
            r"\bkilowatt",
            r"\belectric(ity)?\b",
            r"\bwater\s+bill\b",
            r"\bgas\s+bill\b",
            r"\bmetered?\b",
            r"\bbilling\s+period\b",
            r"\bservice\s+address\b",
        ],
    }
    CONTRACT_SIGNALS = [
        r"\bagreement\b", r"\bcontract\b", r"\bwhereas\b",
        r"\bterminat", r"\bgoverning\s+law\b", r"\bindemnif"
    ]

    # Minimum score difference for Stage 1 to be considered confident
    KEYWORD_CONFIDENCE_THRESHOLD = 2

    def __init__(self):
        print("[INFO] Loading zero-shot classifier (facebook/bart-large-mnli)...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _keyword_scores(self, text_lower):
        """Return a dict of {category: score} based on keyword signal hits."""
        scores = {cat: 0 for cat in self.KEYWORD_SIGNALS}
        for category, patterns in self.KEYWORD_SIGNALS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[category] += 1
        return scores

    def classify_document(self, text):
        if not text.strip():
            return "Unclassifiable"

        text_lower = text.lower()

        contract_hits = sum(1 for p in self.CONTRACT_SIGNALS if re.search(p, text_lower))
        if contract_hits >= 3:
            return "Other"

        scores = self._keyword_scores(text_lower)

        # Also score on fully collapsed text (catches garbled multi-column PDFs)
        collapsed = " ".join(text.split()).lower()
        if collapsed != text_lower:
            collapsed_scores = self._keyword_scores(collapsed)
            for cat in scores:
                scores[cat] = max(scores[cat], collapsed_scores[cat])

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        sorted_scores = sorted(scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0
        gap = best_score - runner_up

        if best_score >= self.KEYWORD_CONFIDENCE_THRESHOLD and gap >= 1:
            return best_category

        # Tiebreaker: email + phone is a strong resume indicator
        if abs(scores.get("Resume", 0) - scores.get("Invoice", 0)) <= 2:
            has_contact = bool(
                re.search(r"[\w.\-+]+@[\w.\-]+\.\w{2,}", text) and
                re.search(r"\+?\d[\d\s\-().]{7,}", text)
            )
            if has_contact and scores.get("Resume", 0) >= 3:
                return "Resume"

        sample_text = text[:300] + " " + text[200:1700]
        result = self.classifier(sample_text, candidate_labels=self.CATEGORIES)
        return result["labels"][0]

    # ------------------------------------------------------------------
    # Field Extraction — general regex patterns
    # ------------------------------------------------------------------

    def _extract_invoice_fields(self, text):
        """
        Extract invoice fields using structural patterns that work across
        any standard invoice format — not tied to specific companies.
        """
        data = {}

        # --- Invoice Number ---
        inv_match = re.search(
            r"(?:invoice\s*(?:no\.?|number|#))[:\s\n]+([A-Z0-9]{2,}[-][A-Z0-9-]+)",
            text, re.IGNORECASE
        )
        data["invoice_number"] = inv_match.group(1).strip() if inv_match else "N/A"

        # --- Date ---
        date_match = re.search(
            r"(?:invoice\s*date|date\s*issued?|date)[:\s]*"
            r"(\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
            r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
            r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b)",
            text, re.IGNORECASE
        )
        # Fallback: first standalone date in doc if no labeled date found
        if not date_match:
            date_match = re.search(
                r"(\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
                r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
                r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b)",
                text, re.IGNORECASE
            )
        data["date"] = date_match.group(1).strip() if date_match else "N/A"

        # --- Company Name ---
        company = "N/A"

        # Pattern 1: Explicit "From:" label
        from_match = re.search(r"(?:from|issued\s+by|service\s+provider)[:\s]+([^\n,|]{3,60})", text, re.IGNORECASE)
        if from_match:
            company = from_match.group(1).strip()

        # Split document on "Bill To" / "Billed To" to isolate the issuer header
        before_billed = re.split(r"bill(?:ed)?\s+to", text, maxsplit=1, flags=re.IGNORECASE)
        # Pattern 2: Take the FIRST line before "Bill To", not the last
        # (company name is always at the top of the header block)
        if len(before_billed) > 1:
            header_lines = [l.strip() for l in before_billed[0].split("\n") if l.strip()]
            for candidate in header_lines:
                if not re.search(r'\d{3,}|@|www\.', candidate, re.IGNORECASE):
                    if re.search(r'[a-zA-Z]{3}', candidate) and len(candidate) < 80:
                        company = candidate
                        break

        # Pattern 3: Strip trailing "INVOICE" / "LIMITED" label words from first line
        if company == "N/A":
            first_lines = [l.strip() for l in text.split("\n") if l.strip()]
            if first_lines:
                candidate = re.sub(r'\s+INVOICE\s*$', '', first_lines[0], flags=re.IGNORECASE).strip()
                if re.search(r'[a-zA-Z]{3}', candidate) and len(candidate) < 80:
                    company = candidate

        data["company"] = company

        # --- Total Amount ---
        total_match = re.search(
            r"(?:total\s+due|amount\s+due|balance\s+due|total\s+amount\s+due)[:\s]*\$?\s*([\d,]+\.\d{2})",
            text, re.IGNORECASE
        )
        try:
            data["total_amount"] = float(total_match.group(1).replace(",", "")) if total_match else 0.0
        except (ValueError, AttributeError):
            data["total_amount"] = 0.0

        return data

    def _extract_resume_fields(self, text):
        """
        Extract resume fields. Works across different CV formats and layouts.
        """
        data = {}

        # --- Email ---
        email_match = re.search(
            r"[\w.\-+]+@[\w.\-]+\.(?:com|org|net|edu|gov|io|co|pk|uk|us|info|biz)[a-z]*",
            text, re.IGNORECASE
        )
        if email_match:
            # Grab the raw match and strip any trailing letters beyond the known TLD
            raw = email_match.group(0)
            # Clean: keep only up to the end of the TLD (2-6 letters after last dot)
            clean = re.match(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,6}", raw)
            data["email"] = clean.group(0) if clean else raw
        else:
            data["email"] = "N/A"

        # --- Phone ---
        phone_match = re.search(
            r"(\+?\d{1,3}[\s\-.]?)?"          
            r"(\(?\d{2,4}\)?[\s\-.]?)"        
            r"(\d{3,4}[\s\-.]?\d{3,4})",      
            text
        )
        data["phone"] = phone_match.group(0).strip() if phone_match else "N/A"

        # --- Experience Years ---
        collapsed = " ".join(text.split())

        exp_match = None

        exp_match = re.search(
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?(?:experience|exp\.?)",
            collapsed, re.IGNORECASE
        )

        if not exp_match:
            exp_match = re.search(
                r"(?:experience|exp)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
                collapsed, re.IGNORECASE
            )

        if not exp_match:
            exp_match = re.search(
                r"(\d+(?:\.\d+)?)\s*\+\s*years?",
                collapsed, re.IGNORECASE
            )

        if exp_match:
            data["experience_years"] = float(exp_match.group(1))
        else:
            # Priority 2: detect month-based durations like "4 months" / "1 month"
            current_year = datetime.datetime.now().year

            month_matches = re.findall(
                r"(\d+)\s*months?",
                collapsed, re.IGNORECASE
            )
            total_months_from_text = sum(int(m) for m in month_matches)

            # Priority 3: sum job date ranges
            year_ranges = re.findall(
                r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(\d{4})\s*[-–—]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)",
                text
            )

            experience_section = ""
            exp_section_match = re.search(
                r"(?:professional\s+experience|work\s+experience|experience|employment)[^\n]*\n(.*?)(?=\n(?:education|skills|projects|achievements|languages|references|certificates)\b|\Z)",
                text, re.IGNORECASE | re.DOTALL
            )
            if exp_section_match:
                experience_section = exp_section_match.group(1)

            # Find ranges in experience section first; fall back to all ranges
            search_text = experience_section if experience_section.strip() else text
            year_ranges = re.findall(
                r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(\d{4})\s*[-–—]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)",
                search_text
            )

            total_years = 0.0
            seen_ranges = set()
            for _, start_yr, _, end_yr in year_ranges:
                try:
                    start = int(start_yr)
                    end = current_year if re.match(r'(?i)present|current|now', end_yr) else int(end_yr)
                    key = (start, end)
                    if key in seen_ranges:
                        continue
                    seen_ranges.add(key)
                    diff = end - start
                    if 1980 <= start <= current_year and 0 < diff <= 15:
                        total_years += diff
                except ValueError:
                    continue

            # Convert month-only internships to years if no year-range total found
            if total_years == 0 and total_months_from_text > 0:
                total_years = round(total_months_from_text / 12, 1)

            # Only set a value if we actually found something; otherwise N/A
            if total_years > 0:
                data["experience_years"] = round(total_years, 1)
            elif total_months_from_text > 0:
                data["experience_years"] = round(total_months_from_text / 12, 1)
            else:
                data["experience_years"] = "N/A"

        # --- Name ---
        name = "N/A"

        HEADER_KEYWORDS = {
            "resume", "curriculum", "vitae", "summary", "profile",
            "experience", "education", "skills", "objective", "contact",
            "linkedin", "github", "portfolio", "declaration", "references",
            "projects", "achievements", "languages", "certifications",
            "internship", "career", "stack", "developer", "engineer",
            "manager", "analyst", "designer", "consultant", "specialist",
            "open", "work", "mern", "frontend", "backend", "fullstack",
        }

        def _title_case_name(raw):
            """Convert ALL-CAPS or fused name to Title Case and return clean string."""

            spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', raw)
            spaced = re.sub(r'(?<=\.)(?=[A-Z])', ' ', spaced)
            parts = []
            for token in spaced.split():
                if token.isupper() and len(token) > 1:
                    parts.append(token.capitalize())
                elif token == token.upper() and '.' in token:
                    parts.append(token)
                else:
                    parts.append(token)
            return " ".join(parts).strip()

        def _is_name_token(token):
            """
            Return True if a single token could be part of a person's name.
            Accepts: Title-case words, ALL-CAPS words (≤12 chars), initials like "M."
            Rejects: long words (>25 chars), words with digits, known tech terms.
            """
            if re.search(r'[\d@/\\#]', token):
                return False
            if len(token) > 25:
                return False
            # Initial like "M." or "J.K."
            if re.match(r'^[A-Z]\.([A-Z]\.)?$', token):
                return True
            # Title-case word: starts with capital, rest are letters/hyphens/apostrophes
            if re.match(r'^[A-Z][a-zA-Z\'\-]{1,24}$', token):
                return True

            if re.match(r'^[A-Z]{2,12}$', token):
                return True
            return False

        def _is_valid_name(candidate, keywords=HEADER_KEYWORDS):
            tokens = candidate.strip().split()
            if not (2 <= len(tokens) <= 5):
                return False
            if not all(_is_name_token(t) for t in tokens):
                return False
            if any(t.lower() in keywords for t in tokens):
                return False
            if len(candidate) > 60:
                return False
            if not any(len(t) >= 3 for t in tokens):
                return False
            return True

        # ── scan the first 15 non-empty lines ───────
        for line in text.split("\n")[:15]:
            line = line.strip()
            if not line:
                continue
            unf = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', line)
            unf = re.sub(r'(?<=\.)(?=[A-Z])', ' ', unf).strip()
            if _is_valid_name(unf):
                name = _title_case_name(line)
                break

        # ── Strategy 2: look just before the email address ─────────
        if name == "N/A" and data.get("email") and data["email"] != "N/A":
            email_pos = text.find(data["email"])
            if email_pos == -1:
                email_pos = text.find("@")
            surrounding = text[max(0, email_pos - 400): email_pos]
            # Grab last non-empty line before the email
            pre_lines = [l.strip() for l in surrounding.split("\n") if l.strip()]
            for candidate_line in reversed(pre_lines[-5:]):
                unf = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', candidate_line)
                unf = re.sub(r'(?<=\.)(?=[A-Z])', ' ', unf).strip()
                if _is_valid_name(unf):
                    name = _title_case_name(candidate_line)
                    break

        # ── broader first-25-lines scan with relaxed token count ───
        if name == "N/A":
            first_lines = [l.strip() for l in text.split("\n") if l.strip()][:25]
            for line in first_lines:
                unf = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', line)
                unf = re.sub(r'(?<=\.)(?=[A-Z])', ' ', unf).strip()
                tokens = unf.split()
                if 2 <= len(tokens) <= 5 and _is_valid_name(unf):
                    name = _title_case_name(line)
                    break

        data["name"] = name

        return data

    def _extract_utility_fields(self, text):
        """
        Extract utility bill fields. Handles alphanumeric account numbers
        and correctly picks the current-period usage (not historical meter totals).
        """
        data = {}

        # --- Account Number ---
        acc_match = re.search(
            r"(?:account\s*(?:number|no\.?|#)?)[:\s]+([A-Z0-9][A-Z0-9\-]{4,})",
            text, re.IGNORECASE
        )
        data["account_number"] = acc_match.group(1).strip() if acc_match else "N/A"

        # --- Date ---
        date_match = re.search(
            r"(?:bill\s*date|invoice\s*date|statement\s*date|date)[:\s]*"
            r"(\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
            r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b)",
            text, re.IGNORECASE
        )
        if not date_match:
            date_match = re.search(
                r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b)",
                text, re.IGNORECASE
            )
        data["date"] = date_match.group(1).strip() if date_match else "N/A"

        # --- Usage ---
        # labeled usage in the summary table
        usage_match = re.search(
            r"(?:usage|consumption|current\s+usage|units\s+consumed)"
            r"[^\d\n]{0,40}?(\d{3,5}(?:\.\d+)?)\s*kWh",
            text, re.IGNORECASE
        )

        if usage_match:
            data["usage_kwh"] = float(usage_match.group(1))   # ← this line is missing
        else:
            all_kwh = re.findall(r"(?<![,\d])(\d{1,5}(?:\.\d+)?)\s*kWh", text, re.IGNORECASE)
            candidates = [float(v) for v in all_kwh if float(v) >= 100]
            data["usage_kwh"] = candidates[0] if candidates else 0.0

        # --- Amount Due ---
        amount_match = re.search(
            r"(?:total\s+amount\s+due|amount\s+due|total\s+due|balance\s+due|total)[:\s]*\$?\s*([\d,]+\.\d{2})",
            text, re.IGNORECASE
        )
        try:
            data["amount_due"] = float(amount_match.group(1).replace(",", "")) if amount_match else 0.0
        except (ValueError, AttributeError):
            data["amount_due"] = 0.0

        return data

    def extract_fields(self, text, doc_class):
        """Route to the appropriate extractor based on document class."""
        base = {"class": doc_class}

        if doc_class == "Invoice":
            base.update(self._extract_invoice_fields(text))
        elif doc_class == "Resume":
            base.update(self._extract_resume_fields(text))
        elif doc_class == "Utility Bill":
            base.update(self._extract_utility_fields(text))
        # Other / Unclassifiable — no extraction required per task spec

        return base

    def process_all_documents(self, raw_documents):
        """
        Process all documents: classify + extract fields.
        Returns dict of {filename: extracted_data}.
        """
        final_output = {}
        for filename, text in raw_documents.items():
            print(f"[INFO] Processing: {filename}")
            doc_class = self.classify_document(text)
            extracted = self.extract_fields(text, doc_class)
            final_output[filename] = extracted
        return final_output


# ---------------------------------------------------------------------------
# 3. SEMANTIC SEARCH ENGINE
# ---------------------------------------------------------------------------

class LocalSearchEngine:
    """
    Builds a FAISS inner-product index over L2-normalized document embeddings
    using the all-MiniLM-L6-v2 SentenceTransformer model (runs fully locally).

    Cosine similarity is achieved via inner product on unit-normalized vectors.
    """

    def __init__(self, raw_documents):
        print("[INFO] Loading sentence embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.filenames = list(raw_documents.keys())
        self.corpus_texts = list(raw_documents.values())
        self.index = None

        if self.corpus_texts:
            self._build_index()

    def _build_index(self):
        self.chunks = []
        chunk_texts = []

        CHUNK_SIZE    = 10   
        OVERLAP       = 3    
        MIN_LINE_LEN  = 15   

        for fname, doc_text in zip(self.filenames, self.corpus_texts):
            # Split into clean lines, filter noise
            lines = [l.strip() for l in doc_text.splitlines()
                    if len(l.strip()) >= MIN_LINE_LEN]

            if not lines:
                self.chunks.append({"filename": fname, "text": doc_text.strip()})
                chunk_texts.append(doc_text.strip())
                continue

            # Slide a window of CHUNK_SIZE lines with OVERLAP step
            step = CHUNK_SIZE - OVERLAP
            for i in range(0, max(1, len(lines) - OVERLAP), step):
                window = lines[i : i + CHUNK_SIZE]
                chunk_text = "\n".join(window)
                if len(chunk_text) < 20:
                    continue
                self.chunks.append({"filename": fname, "text": chunk_text})
                chunk_texts.append(chunk_text)

        embeddings = self.model.encode(
            chunk_texts, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[INFO] FAISS index built: {len(self.chunks)} chunks "
            f"(window={CHUNK_SIZE} lines, overlap={OVERLAP}) "
            f"from {len(self.filenames)} docs.")

    def search(self, query, top_k=None):
        if not hasattr(self, 'chunks') or self.index is None:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        query_vec = query_vec / norm

        # Fetch more candidates so we have enough after dedup
        n_candidates = min(len(self.chunks), 50)
        scores, indices = self.index.search(query_vec, n_candidates)

        # Keep the BEST scoring chunk per file (the chunk itself IS the answer context)
        seen_files = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            fname = self.chunks[idx]["filename"]
            if fname not in seen_files or score > seen_files[fname]["score"]:
                seen_files[fname] = {
                    "filename": fname,
                    "score":    float(score),
                    "snippet":  self.chunks[idx]["text"],  
                }

        all_results = sorted(seen_files.values(), key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            return all_results[:top_k]

        # Auto threshold: only return docs with meaningful similarity
        SCORE_THRESHOLD = 0.25
        filtered = [r for r in all_results if r["score"] >= SCORE_THRESHOLD]
        return filtered[:3] if filtered else all_results[:1]  


# ---------------------------------------------------------------------------
# 4. LOCAL DOCUMENT QA
# ---------------------------------------------------------------------------

class DocumentQA:
    INTENT_MAP = [
        (r"\b(compan|employer|organization|firm|workplace|worked\s+at|work(?:ed)?\s+for)\b", ["company"]),
        (r"\b(invoice\s*(?:no|number|#|id))\b",     ["invoice_number"]),
        (r"\b(amount|total|due|paid|payment|cost)\b", ["total_amount", "amount_due"]),
        (r"\b(date|when|issued|period)\b",            ["date"]),
        (r"\b(name|who|person|candidate|applicant)\b",["name"]),
        (r"\b(email|e-mail|contact|mail)\b",          ["email"]),
        (r"\b(phone|mobile|cell)\b",                  ["phone"]),
        (r"\b(experience|years|senior|junior)\b",     ["experience_years"]),
        (r"\b(account|meter|usage|kwh|electricity)\b",["account_number", "usage_kwh", "amount_due"]),
    ]

    def __init__(self, search_engine, processed_results, raw_documents):
        self.engine   = search_engine
        self.results  = processed_results
        self.raw_docs = raw_documents

        # Load local LLM for grounded answer generation
        print("[INFO] Loading TinyLlama for QA...")
        self.llm = LocalQAPipeline()

    def _extract_employers(self, text):
        """Extract employer/company names from resume raw text."""
        employers = []
        seen = set()

        SKIP_WORDS = {
            "present", "current", "now", "remote", "full", "part", "time",
            "lahore", "karachi", "islamabad", "pakistan", "usa", "uk",
        }

        # "Job Title — Company Name" or "Job Title – Company Name"
        for match in re.finditer(
            r"(?:—|–|--)\s*([A-Z][A-Za-z0-9&.',\s]{2,50})(?:\n|$|\|)",
            text
        ):
            candidate = match.group(1).strip().rstrip(".,")
            words = candidate.lower().split()
            if any(w in SKIP_WORDS for w in words):
                continue
            # Must have at least one word that looks like a proper noun
            if re.search(r'[A-Z][a-z]{2,}', candidate) and 3 <= len(candidate) <= 60:
                key = candidate.lower()
                if key not in seen:
                    seen.add(key)
                    employers.append(candidate)

        for match in re.finditer(
            r"([A-Z][A-Za-z0-9&.',\s]{2,50}?"
            r"(?:Pvt\.?\s*Ltd|Inc\.?|Corp\.?|LLC|Technologies|Solutions|Services|Group|Systems|Agency))"
            r"(?:\s*\n|\s*\||\s*$)",
            text
        ):
            candidate = match.group(1).strip().rstrip(".,")
            key = candidate.lower()
            if key not in seen and 3 <= len(candidate) <= 70:
                seen.add(key)
                employers.append(candidate)

        return employers[:6]  
    
    def _detect_intent(self, query):
        found = []
        for pattern, fields in self.INTENT_MAP:
            if re.search(pattern, query, re.IGNORECASE):
                for f in fields:
                    if f not in found:
                        found.append(f)
        return found

    def answer(self, query):
        intent_fields = self._detect_intent(query)
        sources = []

        # ---  Detect document type from query ---
        type_hint = None
        if re.search(r'\b(worked|employer|company|experience|resume|cv|candidate|applicant|skills|project|built|developed|certification|education)\b',
                    query, re.IGNORECASE):
            type_hint = "Resume"
        elif re.search(r'\b(invoice|invoice\s*no|billing|vendor)\b', query, re.IGNORECASE):
            type_hint = "Invoice"
        elif re.search(r'\b(utility|electricity|kwh|energy|bill|account\s*no)\b', query, re.IGNORECASE):
            type_hint = "Utility Bill"

        if type_hint:
            matching_fnames = [
                fname for fname, fields in self.results.items()
                if fields.get("class") == type_hint
            ]
            if matching_fnames:
                # Build fake hits from matching docs (bypass FAISS doc selection)
                hits = [{"filename": f, "score": 1.0} for f in matching_fnames]
            else:
                # Fallback to FAISS if no typed doc found
                hits = self.engine.search(query)
        else:
            hits = self.engine.search(query)

        if not hits:
            return {"answer": "No relevant documents found.", "sources": [], "hits": []}

        # ---Build LLM context using best_snippet from RAW TEXT 
        llm_context_docs = []
        for hit in hits:
            fname     = hit["filename"]
            fields    = self.results.get(fname, {})
            doc_class = fields.get("class", "Unknown")
            raw_text  = self.raw_docs.get(fname, "")

            # Extract best context from full raw text based on query
            best_snip = self._best_snippet(raw_text, query, window=800)

            # Extract employer names for resume documents
            employers = []
            if doc_class == "Resume":
                employers = self._extract_employers(raw_text)

            llm_context_docs.append({
                "filename":  fname,
                "text":      best_snip,
                "fields":    {k: v for k, v in fields.items() if k != "class"},
                "employers": employers,
            })

            sources.append({
                "filename":    fname,
                "score":       hit["score"],
                "doc_class":   doc_class,
                "fields_used": self._from_fields(fields, intent_fields[:], fname),
                "snippet":     best_snip,
            })

        # --- Generate grounded answer using local LLM ---
        llm_answer = self.llm.generate_answer(query, llm_context_docs)
        return {"answer": llm_answer, "sources": sources, "hits": hits}

    def _from_fields(self, fields, intent_fields, fname=""):
        LABELS = {
            "name": "Name", "email": "Email", "phone": "Phone",
            "experience_years": "Experience", "company": "Company",
            "invoice_number": "Invoice No.", "date": "Date",
            "total_amount": "Total Amount", "amount_due": "Amount Due",
            "account_number": "Account No.", "usage_kwh": "Usage (kWh)",
        }
        parts = []
        doc_class = fields.get("class", "")

        # --- Resume + company/employer intent: scan raw text for employer names ---
        if doc_class == "Resume" and "company" in intent_fields:
            raw = self.raw_docs.get(fname, "")
            employers = self._extract_employers(raw)
            if employers:
                parts.append(f"Employers: {', '.join(employers)}")
            # Remove "company" so we don't look for a field that doesn't exist on resumes
            intent_fields = [f for f in intent_fields if f != "company"]

        null_vals = ("",) if intent_fields else ("N/A", "", 0, 0.0)

        check = intent_fields if intent_fields else list(LABELS.keys())
        for field in check:
            val = fields.get(field)
            if val is not None and val not in null_vals:
                label = LABELS.get(field, field.replace("_", " ").title())
                if field == "experience_years":
                    parts.append(f"{label}: {val} year(s)")
                elif field in ("total_amount", "amount_due") and isinstance(val, float):
                    parts.append(f"{label}: ${val:,.2f}")
                else:
                    parts.append(f"{label}: {val}")
            if not intent_fields and len(parts) >= 3:
                break
        return parts

    def _best_snippet(self, text, query, window=800):
        if not text:
            return "(no text available)"

        q_words = set(re.findall(r'\b\w{3,}\b', query.lower()))

        # employer/company queries — extract the full Experience section
        if any(w in q_words for w in {"company", "employer", "worked", "work", "experience"}):
            exp_match = re.search(
                r"(experience|employment|work\s*history)[^\n]*\n(.+?)(?=\n(?:education|skills|certifications|projects|achievements)\b|\Z)",
                text, re.IGNORECASE | re.DOTALL
            )
            if exp_match:
                section = exp_match.group(0).strip()
                return section[:window] + ("..." if len(section) > window else "")

        # General case: score paragraphs and return the best matching one
        paragraphs = [p.strip() for p in re.split(r'\n{2,}|\n(?=[A-Z])', text) if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [text.strip()]

        best_score, best_para = -1, ""
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(1 for w in q_words if w in para_lower)
            first_line = para.split('\n')[0].lower()
            score += sum(2 for w in q_words if w in first_line)
            if score > best_score:
                best_score, best_para = score, para

        snippet = best_para[:window]
        if len(best_para) > window:
            snippet += "..."
        return snippet

# ---------------------------------------------------------------------------
# 5. (OPTIONAL BONUS) LOCAL QA PIPELINE — TinyLlama
# ---------------------------------------------------------------------------

class LocalQAPipeline:
    """
    Optional bonus: local question-answering over retrieved document context
    using TinyLlama-1.1B-Chat (runs on CPU/GPU, no internet required).
    """

    def __init__(self):

        print("[INFO] Loading TinyLlama-1.1B-Chat (local LLM)...")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # float32 only — float16 needs CUDA
        )
        self.llm = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate_answer(self, query, retrieved_docs):
        """Answer a query grounded in retrieved document context."""
        if not retrieved_docs:
            return "No relevant documents found to answer this question."

        context_parts = []
        for doc in retrieved_docs:
            fname     = doc["filename"]
            text      = doc.get("text", doc.get("snippet", ""))
            fields    = doc.get("fields", {})
            employers = doc.get("employers", [])

            part = f"--- {fname} ---\n"
            if employers:
                part += f"Employers/Companies: {', '.join(employers)}\n"
            if fields:
                for k, v in fields.items():
                    if v not in ("N/A", "", 0, 0.0, None):
                        part += f"{k}: {v}\n"
            part += f"\nRelevant Text:\n{text}"
            context_parts.append(part)

        context = "\n\n".join(context_parts)

        prompt = f"""<|system|>
    You are a precise document analysis assistant.
    Answer the user's question using ONLY the information provided in the document context below.
    Be specific. List items clearly. Do NOT make up or infer anything not present in the context.
    If the answer is not found, say exactly: "I could not find this information in the provided documents."
    <|user|>
    Document Context:
    {context}

    Question: {query}
    <|assistant|>
    """
        output = self.llm(
            prompt,
            max_new_tokens=300,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.2
        )
        full = output[0]["generated_text"]
        return full.split("<|assistant|>\n")[-1].strip()