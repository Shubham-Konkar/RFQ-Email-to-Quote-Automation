import os, json, re, uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Paths
DATA_EMAILS = Path("data/emails")
DATA_CATALOG = Path("data/catalog/catalog.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utils
# ---------------------------

def load_emails() -> List[Dict[str, Any]]:
    """Load all .txt emails from data/emails using UTF-8 to avoid Windows cp1252 decode errors."""
    loader = DirectoryLoader(
        str(DATA_EMAILS),
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
        show_progress=True,
    )  # Forces UTF‑8 while loading text files from a directory per LangChain directory loader patterns [web:69][web:76][web:75]
    docs = loader.load()  # Loads all matching files from the directory into Document objects as recommended in the loader docs [web:69]
    emails = []
    for d in docs:
        emails.append({
            "id": Path(d.metadata.get("source", f"email-{uuid.uuid4()}")).stem,
            "content": d.page_content
        })  # Keep a simple id derived from filename for output naming as commonly done with directory loaders [web:69]
    return emails  # Returns a list of dicts ready for downstream RFQ detection and extraction consistent with the specified workflow [web:69]

def load_catalog() -> List[Dict[str, Any]]:
    with open(DATA_CATALOG, "r", encoding="utf-8") as f:
        return json.load(f)  # Loads the small JSON product catalog used for matching and pricing as described earlier [web:69]

def simple_is_rfq(text: str) -> bool:
    patt = r"\b(rfq|request\s+for\s+quote|please\s+quote|quote\s+for|quotation)\b"
    return re.search(patt, text, flags=re.IGNORECASE) is not None  # Simple keyword/rule RFQ check aligns with assignment step options [web:69]

# ---------------------------
# LLM for extraction
# ---------------------------

class LineItem(BaseModel):
    product: str = Field(description="Product name or SKU as stated in email")
    quantity: int = Field(description="Requested quantity (integer)")

class EmailExtraction(BaseModel):
    items: List[LineItem] = Field(default_factory=list, description="All requested items")
    notes: Optional[str] = Field(default=None, description="Other relevant details if present")

def build_llm():
    # Temperature 0 for deterministic extraction
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Any OpenAI chat model can be used via the LangChain OpenAI chat wrapper [web:112][web:115]

def extract_items_with_llm(llm: ChatOpenAI, email_text: str) -> EmailExtraction:
    """Use with_structured_output to reliably parse items from unstructured email text."""
    structured_llm = llm.with_structured_output(EmailExtraction)  # Uses LangChain structured outputs to get Pydantic-typed results reliably [web:97][web:93][web:91]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract requested products and integer quantities from the email."),
        ("human", "{email}")
    ])  # Guides the LLM to produce only the required fields per structured output best practices [web:97]
    chain = prompt | structured_llm  # Composes a runnable chain of prompt -> model with structure enforcement as shown in LangChain docs [web:97]
    return chain.invoke({"email": email_text})  # Executes the extraction for a single email text into the EmailExtraction model instance [web:97]

# ---------------------------
# Matching + Pricing
# ---------------------------

def match_catalog(items: List[LineItem], catalog: List[Dict[str, Any]]):
    """Fuzzy-ish match by SKU exact or name case-insensitive contains."""
    matched = []
    missing = []
    for it in items:
        q = it.product.strip().lower()
        found = None
        # exact SKU
        for row in catalog:
            if q == str(row.get("sku", "")).lower():
                found = row; break
        if not found:
            # name contains or equality
            for row in catalog:
                name = str(row.get("name", "")).lower()
                if q == name or q in name or name in q:
                    found = row; break
        if found:
            qty = max(1, int(it.quantity))
            price = float(found["unit_price"])
            matched.append({
                "sku": found["sku"],
                "name": found["name"],
                "quantity": qty,
                "unit_price": price,
                "line_total": qty * price
            })  # Builds matched lines with computed line totals for later subtotal/tax computations as typical quoting logic [web:69]
        else:
            missing.append({"requested": it.product, "quantity": it.quantity})  # Records unmatched items for the quote note as the assignment specifies [web:69]
    return matched, missing  # Returns both matched and missing sets to inform quote output and console summary per the workflow [web:69]

def compute_totals(matched: List[Dict[str, Any]], tax_rate: float = 0.08):
    subtotal = sum(x["line_total"] for x in matched)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    return round(subtotal, 2), tax, total  # Computes totals with an 8% tax example to complete the quote pricing as described earlier [web:69]

# ---------------------------
# Quote generation
# ---------------------------

def render_quote_text(email_id: str, matched: List[Dict[str, Any]], missing: List[Dict[str, Any]],
                      subtotal: float, tax: float, total: float, tax_rate: float = 0.08) -> str:
    now = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append("==== Quotation ====")
    lines.append(f"Date: {now}")
    lines.append(f"Reference: {email_id}")
    lines.append("")
    if matched:
        lines.append("Items:")
        for m in matched:
            lines.append(f"- {m['name']} ({m['sku']}) x {m['quantity']} @ ${m['unit_price']:.2f} = ${m['line_total']:.2f}")
        lines.append("")
        lines.append(f"Subtotal: ${subtotal:.2f}")
        lines.append(f"Tax ({int(tax_rate*100)}%): ${tax:.2f}")
        lines.append(f"TOTAL: ${total:.2f}")
    else:
        lines.append("No catalog items matched this request.")
    if missing:
        lines.append("")
        lines.append("Unavailable/Unmatched items:")
        for mm in missing:
            lines.append(f"- {mm['requested']} x {mm['quantity']}")
    lines.append("")
    lines.append("Thank you for your inquiry.")
    return "\n".join(lines)  # Generates a simple text quote per the assignment deliverable format to be saved to /output [web:69]

def save_quote(email_id: str, content: str) -> Path:
    filename = OUTPUT_DIR / f"{email_id}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename  # Writes the quote text file with UTF‑8 encoding to avoid downstream encoding issues on Windows editors [web:69]

# ---------------------------
# Orchestration
# ---------------------------

def process_all_emails():
    catalog = load_catalog()  # Loads the small catalog used for item matching and pricing per the sample workflow [web:69]
    llm = build_llm()  # Builds the OpenAI chat model via LangChain for structured extraction as discussed previously [web:112][web:115]
    emails = load_emails()  # Reads all .txt email files from the configured directory using the UTF‑8 loader [web:69]
    if not emails:
        print("No emails found in data/emails/")
        return
    for e in emails:
        text = e["content"]
        email_id = e["id"]
        # Step 1: classify RFQ
        is_rfq = simple_is_rfq(text)
        if not is_rfq:
            print(f"[{email_id}] No quote — not an RFQ.")
            continue
        # Step 2: extract items
        extraction = extract_items_with_llm(llm, text)
        items = extraction.items
        if not items:
            print(f"[{email_id}] No quote — RFQ detected but no items extracted.")
            continue
        # Step 3: match against catalog and price
        matched, missing = match_catalog(items, catalog)
        if not matched:
            quote_text = render_quote_text(email_id, matched, missing, 0.0, 0.0, 0.0)
            filepath = save_quote(email_id, quote_text)
            print(f"[{email_id}] No quote — items unavailable. Saved note to {filepath.name}.")
            continue
        # Step 4: totals and quote output
        subtotal, tax, total = compute_totals(matched)
        quote_text = render_quote_text(email_id, matched, missing, subtotal, tax, total)
        filepath = save_quote(email_id, quote_text)
        # Step 5: console summary
        print(f"[{email_id}] Quote saved to {filepath.name} (items: {len(matched)}, missing: {len(missing)}, total: ${total:.2f}).")

if __name__ == "__main__":
    process_all_emails()  # Entry point runs the pipeline end-to-end per the assignment’s step list using the directory loader and structured outputs [web:69][web:97]
