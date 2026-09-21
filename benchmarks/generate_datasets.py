#!/usr/bin/env python3
"""Generate classification_50.jsonl and extraction_50.jsonl with validated gold spans."""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

SHARED_LABELS = {
    "billing": "Billing, charges, invoices, refunds, payments, pricing, or subscription fees",
    "technical": "Technical problems, bugs, errors, crashes, outages, or product malfunction",
    "account": "Account access, login, password, profile, permissions, or membership",
    "shipping": "Shipping, delivery, tracking, packages, or logistics",
    "other": "Anything else not covered by billing, technical, account, or shipping",
}

# Diverse support-ticket templates per label (fill with variation).
CLASS_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged {amount} twice for invoice {inv} on {date}. Please reverse the duplicate charge.",
        "My subscription renewed at {amount} but I cancelled last week. I need a refund for the unexpected renewal.",
        "The invoice for order {inv} shows a tax line I do not understand. Can someone explain the {amount} fee?",
        "Payment failed when I tried to pay {amount} with card ending {card}. Billing portal returns error code {code}.",
        "I upgraded to the Pro plan and was billed {amount} immediately. Confirm this matches the published pricing.",
        "Please send a corrected invoice for {inv}; the company name on the PDF is wrong and finance cannot process {amount}.",
        "We were charged for seats we removed on {date}. The unused seats still appear as {amount} on the statement.",
        "Refund request: package never shipped but {amount} was captured. Transaction id {inv}.",
        "Why did my annual plan jump from the old price to {amount}? I never opted into a price increase.",
        "Accounting needs a VAT receipt for payment {inv} of {amount} processed on {date}.",
    ],
    "technical": [
        "The dashboard crashes with a blank screen after login since the {date} release. Console shows error {code}.",
        "API endpoint /v2/reports returns HTTP {code} intermittently under load. Started around {date}.",
        "Mobile app version {ver} freezes when opening attachments larger than {amount} MB.",
        "Search filters stop working after applying more than three facets. Reproducible on Chrome {ver}.",
        "Webhook deliveries are delayed by several hours. Last successful delivery was {date} with id {inv}.",
        "Export to CSV truncates rows past line {amount}. This is a regression from last month.",
        "Two-factor prompts loop forever on Safari. Error toast shows code {code}.",
        "Realtime sync disconnects every few minutes with websocket close {code}.",
        "PDF preview renders garbled text for files uploaded after {date}.",
        "Background jobs stuck in 'queued' status; job id {inv} has not progressed since {date}.",
    ],
    "account": [
        "I cannot reset my password; the email link expires immediately. Account email is associated with workspace {inv}.",
        "Please disable MFA for user id {inv} so we can recover access after a lost phone on {date}.",
        "A former employee still has admin rights. Remove access for account {inv} today.",
        "I need to change the login email on my profile from the old address to a new corporate one.",
        "Invite emails for seat {inv} never arrive. We tried resending three times since {date}.",
        "Session keeps signing me out every few minutes after enabling SSO with provider code {code}.",
        "Someone changed my username without authorization. Restore account {inv} and lock the profile.",
        "How do I transfer ownership of workspace {inv} to a new admin before {date}?",
        "I am locked out after too many failed attempts. Unlock account {inv} and reset the password.",
        "Please confirm whether account {inv} is on the Enterprise tier or still on trial.",
    ],
    "shipping": [
        "Order {inv} still shows 'label created' since {date}. Tracking never updates.",
        "The package arrived damaged; box was crushed and item {inv} is unusable. Need a replacement shipment.",
        "Carrier marked delivery on {date} but nothing was left. Address is correct for order {inv}.",
        "Can you upgrade shipping for order {inv} to overnight? We need it before {date}.",
        "Wrong item was shipped in package {inv}. We ordered SKU A but received SKU B.",
        "International shipment {inv} is stuck in customs since {date}. Any advice on the delay?",
        "Please hold shipment {inv} at the depot; I will pick it up on {date}.",
        "Tracking number for order {inv} returns not found. Confirm the label was actually purchased.",
        "Split shipment: one box arrived, the second with {amount} units is missing since {date}.",
        "Change delivery address for order {inv} before it leaves the warehouse tonight.",
    ],
    "other": [
        "Do you offer volume discounts for nonprofits? We are evaluating a multi-year agreement.",
        "I would like to schedule a product demo for our team sometime after {date}.",
        "Where can I find documentation on data residency and GDPR subprocessors?",
        "Please add our company logo to the customer showcase if that program is still open.",
        "Is there an academic license option for classroom use starting {date}?",
        "We want to provide feedback on the onboarding checklist; who owns the UX research queue?",
        "Can you clarify your SLA credits policy in writing for our legal review?",
        "Interested in partnering as a reseller in the APAC region. Point me to the right contact.",
        "How do I cite your public API in an open-source README without violating the brand guidelines?",
        "General inquiry: what is the roadmap for offline mode in the next two quarters?",
    ],
}

AMOUNTS = ["$12.99", "$49.00", "$120.50", "$7.00", "$999.00", "€34.20", "£18.75"]
DATES = ["March 3", "April 12", "May 1", "June 18", "July 22", "August 9", "September 5", "October 14"]
INVS = ["INV-1042", "INV-8891", "ORD-5501", "ORD-7712", "TKT-220", "PO-9088", "WS-441", "JOB-3321"]
CARDS = ["4421", "8890", "1203", "7744"]
CODES = ["E_PAYMENT", "502", "ERR_TIMEOUT", "WS_1006", "HTTP_429", "AUTH_423"]
VERS = ["3.2.1", "4.0.0", "2.9.8", "128.0"]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        amount=rng.choice(AMOUNTS),
        date=rng.choice(DATES),
        inv=rng.choice(INVS),
        card=rng.choice(CARDS),
        code=rng.choice(CODES),
        ver=rng.choice(VERS),
    )


def generate_classification(n: int = 50, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    labels = list(CLASS_TEMPLATES.keys())
    # Round-robin then shuffle for diversity
    plan: list[str] = []
    while len(plan) < n:
        plan.extend(labels)
    plan = plan[:n]
    rng.shuffle(plan)

    rows: list[dict] = []
    used: set[str] = set()
    for i, label in enumerate(plan, start=1):
        templates = CLASS_TEMPLATES[label]
        # try several fills to avoid near-duplicates
        text = None
        for _ in range(20):
            candidate = _fill(rng.choice(templates), rng)
            # light paraphrase-ish variation
            prefixes = ["", "Hi support — ", "Hello, ", "Urgent: ", "Follow-up: "]
            suffixes = ["", " Thanks.", " Please advise.", " This is blocking us.", ""]
            candidate = rng.choice(prefixes) + candidate + rng.choice(suffixes)
            if candidate not in used:
                text = candidate
                used.add(candidate)
                break
        assert text is not None
        rows.append(
            {
                "id": f"cls_{i:03d}_{label}",
                "text": text,
                "label": label,
                "labels": dict(SHARED_LABELS),
            }
        )
    return rows


# Extraction: multi-sentence paragraphs with an exact gold span (often a short fact).
# Mix sentence-level and sub-sentence golds.

EXTRACTION_SPECS: list[dict] = [
    # Science / history
    {
        "id": "ext_curie_physics",
        "sents": [
            "Marie Curie discovered radium and polonium.",
            "She won the Nobel Prize in Physics in 1903.",
            "Later she won a second Nobel Prize in Chemistry in 1911.",
        ],
        "question": "In what year did Marie Curie win the Nobel Prize in Physics?",
        "gold": "1903",
    },
    {
        "id": "ext_curie_chem",
        "sents": [
            "Marie Curie discovered radium and polonium.",
            "She won the Nobel Prize in Physics in 1903.",
            "Later she won a second Nobel Prize in Chemistry in 1911.",
        ],
        "question": "In what year did she win the Nobel Prize in Chemistry?",
        "gold": "1911",
    },
    {
        "id": "ext_lovelace",
        "sents": [
            "Ada Lovelace worked on Babbage's Analytical Engine.",
            "She published notes in 1843.",
            "Those notes include an early algorithm.",
        ],
        "question": "When were Ada Lovelace's notes published?",
        "gold": "1843",
    },
    {
        "id": "ext_turing",
        "sents": [
            "Alan Turing proposed an abstract computing machine in 1936.",
            "During World War II he worked at Bletchley Park.",
            "The Turing Award was established in 1966.",
        ],
        "question": "In what year did Turing propose his abstract computing machine?",
        "gold": "1936",
    },
    {
        "id": "ext_einstein",
        "sents": [
            "Albert Einstein published the special theory of relativity in 1905.",
            "He later developed general relativity.",
            "He received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        ],
        "question": "When did Einstein publish special relativity?",
        "gold": "1905",
    },
    {
        "id": "ext_darwin",
        "sents": [
            "Charles Darwin sailed on the HMS Beagle.",
            "On the Origin of Species was published in 1859.",
            "The book introduced natural selection to a wide audience.",
        ],
        "question": "When was On the Origin of Species published?",
        "gold": "1859",
    },
    {
        "id": "ext_fleming",
        "sents": [
            "Alexander Fleming noticed mold killing bacteria in a petri dish.",
            "He identified penicillin in 1928.",
            "Antibiotics later transformed medicine worldwide.",
        ],
        "question": "In what year did Fleming identify penicillin?",
        "gold": "1928",
    },
    {
        "id": "ext_newton",
        "sents": [
            "Isaac Newton formulated the laws of motion.",
            "His Principia was published in 1687.",
            "He also made major contributions to optics and mathematics.",
        ],
        "question": "When was Newton's Principia published?",
        "gold": "1687",
    },
    # Geography / orgs
    {
        "id": "ext_un_hq",
        "sents": [
            "The United Nations was founded after World War II.",
            "Its headquarters are in New York City.",
            "The General Assembly meets there each year.",
        ],
        "question": "Where are the United Nations headquarters?",
        "gold": "New York City",
    },
    {
        "id": "ext_cern",
        "sents": [
            "CERN operates the Large Hadron Collider.",
            "The laboratory sits on the border of France and Switzerland.",
            "Researchers there discovered the Higgs boson in 2012.",
        ],
        "question": "In what year was the Higgs boson discovered?",
        "gold": "2012",
    },
    {
        "id": "ext_amazon_river",
        "sents": [
            "The Amazon River is the largest river by discharge volume.",
            "It flows through Brazil and several neighboring countries.",
            "Its basin holds the world's largest tropical rainforest.",
        ],
        "question": "Which country does the Amazon River flow through among others?",
        "gold": "Brazil",
    },
    {
        "id": "ext_everest",
        "sents": [
            "Mount Everest is Earth's highest mountain above sea level.",
            "Its summit elevation is 8,849 meters.",
            "It lies on the border between Nepal and China.",
        ],
        "question": "What is the summit elevation of Mount Everest?",
        "gold": "8,849 meters",
    },
    # Business / product-ish paragraphs
    {
        "id": "ext_acme_ceo",
        "sents": [
            "Acme Robotics raised a Series B round last spring.",
            "The company appointed Priya Natarajan as CEO in 2024.",
            "Its warehouse robots now operate in twelve countries.",
        ],
        "question": "Who was appointed CEO of Acme Robotics?",
        "gold": "Priya Natarajan",
    },
    {
        "id": "ext_acme_countries",
        "sents": [
            "Acme Robotics raised a Series B round last spring.",
            "The company appointed Priya Natarajan as CEO in 2024.",
            "Its warehouse robots now operate in twelve countries.",
        ],
        "question": "In how many countries do Acme's warehouse robots operate?",
        "gold": "twelve countries",
    },
    {
        "id": "ext_northwind_rev",
        "sents": [
            "Northwind Analytics published its annual report yesterday.",
            "Revenue reached $48 million in fiscal 2025.",
            "The firm attributed growth to enterprise contracts in Europe.",
        ],
        "question": "What revenue did Northwind Analytics report for fiscal 2025?",
        "gold": "$48 million",
    },
    {
        "id": "ext_ship_eta",
        "sents": [
            "Order ORD-7712 left the Memphis warehouse on Monday.",
            "Carrier tracking shows an estimated delivery on Friday.",
            "The package weighs 4.2 kilograms.",
        ],
        "question": "When is the estimated delivery for the package?",
        "gold": "Friday",
    },
    {
        "id": "ext_ship_weight",
        "sents": [
            "Order ORD-7712 left the Memphis warehouse on Monday.",
            "Carrier tracking shows an estimated delivery on Friday.",
            "The package weighs 4.2 kilograms.",
        ],
        "question": "How much does the package weigh?",
        "gold": "4.2 kilograms",
    },
    {
        "id": "ext_invoice_amount",
        "sents": [
            "Finance issued invoice INV-1042 to Contoso Ltd.",
            "The total due is $12,450 excluding tax.",
            "Payment terms are net 30 from the invoice date.",
        ],
        "question": "What is the total due on invoice INV-1042?",
        "gold": "$12,450",
    },
    {
        "id": "ext_payment_terms",
        "sents": [
            "Finance issued invoice INV-1042 to Contoso Ltd.",
            "The total due is $12,450 excluding tax.",
            "Payment terms are net 30 from the invoice date.",
        ],
        "question": "What are the payment terms?",
        "gold": "net 30",
    },
    # Sentence-level golds (fairer to Jev closed-set)
    {
        "id": "ext_sent_mars",
        "sents": [
            "NASA's Perseverance rover landed in Jezero Crater.",
            "It collects rock samples for a future return mission.",
            "The rover also carries the Ingenuity helicopter.",
        ],
        "question": "Where did Perseverance land?",
        "gold": "NASA's Perseverance rover landed in Jezero Crater.",
    },
    {
        "id": "ext_sent_vaccine",
        "sents": [
            "mRNA vaccine platforms accelerated pandemic response.",
            "Clinical trials demonstrated high efficacy against severe disease.",
            "Cold-chain logistics remained a deployment challenge.",
        ],
        "question": "What challenge remained for deployment?",
        "gold": "Cold-chain logistics remained a deployment challenge.",
    },
    {
        "id": "ext_sent_python",
        "sents": [
            "Python is a widely used programming language.",
            "Guido van Rossum created it in the late 1980s.",
            "The language emphasizes readability and batteries-included libraries.",
        ],
        "question": "Who created Python?",
        "gold": "Guido van Rossum created it in the late 1980s.",
    },
    {
        "id": "ext_kyoto",
        "sents": [
            "The Kyoto Protocol was adopted in 1997.",
            "It set binding emission targets for developed countries.",
            "The Paris Agreement later broadened participation.",
        ],
        "question": "When was the Kyoto Protocol adopted?",
        "gold": "1997",
    },
    {
        "id": "ext_paris_agree",
        "sents": [
            "The Kyoto Protocol was adopted in 1997.",
            "It set binding emission targets for developed countries.",
            "The Paris Agreement later broadened participation.",
        ],
        "question": "Which later agreement broadened participation?",
        "gold": "The Paris Agreement",
    },
    {
        "id": "ext_dna",
        "sents": [
            "Watson and Crick proposed the double helix structure of DNA.",
            "Their paper appeared in Nature in 1953.",
            "Rosalind Franklin's X-ray work was crucial to the discovery.",
        ],
        "question": "In what year did the DNA structure paper appear in Nature?",
        "gold": "1953",
    },
    {
        "id": "ext_franklin",
        "sents": [
            "Watson and Crick proposed the double helix structure of DNA.",
            "Their paper appeared in Nature in 1953.",
            "Rosalind Franklin's X-ray work was crucial to the discovery.",
        ],
        "question": "Whose X-ray work was crucial to the DNA discovery?",
        "gold": "Rosalind Franklin",
    },
    {
        "id": "ext_tesla_coil",
        "sents": [
            "Nikola Tesla experimented with high-voltage electricity.",
            "He demonstrated the Tesla coil in the 1890s.",
            "Wireless power transmission remained one of his ambitions.",
        ],
        "question": "When did Tesla demonstrate the Tesla coil?",
        "gold": "the 1890s",
    },
    {
        "id": "ext_ibm_deepblue",
        "sents": [
            "IBM's Deep Blue faced Garry Kasparov in chess matches.",
            "In 1997 Deep Blue won a six-game rematch.",
            "The event marked a milestone in game-playing AI.",
        ],
        "question": "In what year did Deep Blue win the rematch against Kasparov?",
        "gold": "1997",
    },
    {
        "id": "ext_gps",
        "sents": [
            "The Global Positioning System relies on a constellation of satellites.",
            "Civilian GPS accuracy improved after selective availability was turned off in 2000.",
            "Smartphones now embed GPS receivers as standard hardware.",
        ],
        "question": "When was selective availability turned off?",
        "gold": "2000",
    },
    {
        "id": "ext_www",
        "sents": [
            "Tim Berners-Lee invented the World Wide Web at CERN.",
            "The first website went online in 1991.",
            "HTTP and HTML became the foundation of the modern web.",
        ],
        "question": "When did the first website go online?",
        "gold": "1991",
    },
    {
        "id": "ext_bernerson",
        "sents": [
            "Tim Berners-Lee invented the World Wide Web at CERN.",
            "The first website went online in 1991.",
            "HTTP and HTML became the foundation of the modern web.",
        ],
        "question": "Who invented the World Wide Web?",
        "gold": "Tim Berners-Lee",
    },
    {
        "id": "ext_photosynthesis",
        "sents": [
            "Photosynthesis converts light energy into chemical energy.",
            "Chlorophyll absorbs mainly blue and red wavelengths.",
            "Oxygen is released as a byproduct of the light reactions.",
        ],
        "question": "What wavelengths does chlorophyll mainly absorb?",
        "gold": "blue and red",
    },
    {
        "id": "ext_oxygen_byproduct",
        "sents": [
            "Photosynthesis converts light energy into chemical energy.",
            "Chlorophyll absorbs mainly blue and red wavelengths.",
            "Oxygen is released as a byproduct of the light reactions.",
        ],
        "question": "What is released as a byproduct of the light reactions?",
        "gold": "Oxygen",
    },
    {
        "id": "ext_battery",
        "sents": [
            "Lithium-ion batteries power most modern electric vehicles.",
            "Energy density typically ranges from 150 to 250 Wh/kg for automotive packs.",
            "Thermal management is critical for safety and longevity.",
        ],
        "question": "What energy density range is typical for automotive lithium-ion packs?",
        "gold": "150 to 250 Wh/kg",
    },
    {
        "id": "ext_chip",
        "sents": [
            "The foundry produced a 5-nanometer process node at scale.",
            "Yield improved to 82 percent in the second quarter.",
            "Customers include major smartphone and GPU vendors.",
        ],
        "question": "What yield did the foundry report in the second quarter?",
        "gold": "82 percent",
    },
    {
        "id": "ext_process_node",
        "sents": [
            "The foundry produced a 5-nanometer process node at scale.",
            "Yield improved to 82 percent in the second quarter.",
            "Customers include major smartphone and GPU vendors.",
        ],
        "question": "What process node did the foundry produce at scale?",
        "gold": "5-nanometer",
    },
    {
        "id": "ext_library",
        "sents": [
            "The city opened a new central library on Oak Street.",
            "It houses more than 400,000 physical volumes.",
            "Late fees were abolished in 2023.",
        ],
        "question": "How many physical volumes does the library house?",
        "gold": "400,000",
    },
    {
        "id": "ext_late_fees",
        "sents": [
            "The city opened a new central library on Oak Street.",
            "It houses more than 400,000 physical volumes.",
            "Late fees were abolished in 2023.",
        ],
        "question": "When were late fees abolished?",
        "gold": "2023",
    },
    {
        "id": "ext_stadium",
        "sents": [
            "Riverside Stadium seats 52,000 spectators.",
            "The renovation finished in 2019.",
            "A retractable roof was added during that project.",
        ],
        "question": "How many spectators does Riverside Stadium seat?",
        "gold": "52,000",
    },
    {
        "id": "ext_roof",
        "sents": [
            "Riverside Stadium seats 52,000 spectators.",
            "The renovation finished in 2019.",
            "A retractable roof was added during that project.",
        ],
        "question": "When did the renovation finish?",
        "gold": "2019",
    },
    {
        "id": "ext_volcano",
        "sents": [
            "Mount Fuji is an active stratovolcano in Japan.",
            "Its last confirmed eruption was in 1707.",
            "The peak is a popular destination for climbers in summer.",
        ],
        "question": "When was Mount Fuji's last confirmed eruption?",
        "gold": "1707",
    },
    {
        "id": "ext_fuji_country",
        "sents": [
            "Mount Fuji is an active stratovolcano in Japan.",
            "Its last confirmed eruption was in 1707.",
            "The peak is a popular destination for climbers in summer.",
        ],
        "question": "In which country is Mount Fuji located?",
        "gold": "Japan",
    },
    {
        "id": "ext_contract",
        "sents": [
            "Brightline Logistics signed a three-year contract with Harbor Foods.",
            "The agreement covers refrigerated transport across five states.",
            "Service begins on November 1.",
        ],
        "question": "When does service begin under the Brightline contract?",
        "gold": "November 1",
    },
    {
        "id": "ext_contract_states",
        "sents": [
            "Brightline Logistics signed a three-year contract with Harbor Foods.",
            "The agreement covers refrigerated transport across five states.",
            "Service begins on November 1.",
        ],
        "question": "How many states does the refrigerated transport agreement cover?",
        "gold": "five states",
    },
    {
        "id": "ext_sent_climate",
        "sents": [
            "Global mean surface temperature has risen since the industrial era.",
            "The IPCC synthesizes climate science for policymakers.",
            "Mitigation pathways include electrification and efficiency gains.",
        ],
        "question": "What organization synthesizes climate science for policymakers?",
        "gold": "The IPCC synthesizes climate science for policymakers.",
    },
    {
        "id": "ext_moon_landing",
        "sents": [
            "Apollo 11 landed on the Moon in 1969.",
            "Neil Armstrong and Buzz Aldrin walked on the lunar surface.",
            "Michael Collins remained in lunar orbit aboard the command module.",
        ],
        "question": "In what year did Apollo 11 land on the Moon?",
        "gold": "1969",
    },
    {
        "id": "ext_armstrong",
        "sents": [
            "Apollo 11 landed on the Moon in 1969.",
            "Neil Armstrong and Buzz Aldrin walked on the lunar surface.",
            "Michael Collins remained in lunar orbit aboard the command module.",
        ],
        "question": "Who remained in lunar orbit aboard the command module?",
        "gold": "Michael Collins",
    },
    {
        "id": "ext_shakespeare",
        "sents": [
            "William Shakespeare wrote tragedies, comedies, and histories.",
            "Hamlet was likely written around 1600.",
            "The Globe Theatre staged many of his plays in London.",
        ],
        "question": "Around when was Hamlet likely written?",
        "gold": "1600",
    },
    {
        "id": "ext_globe",
        "sents": [
            "William Shakespeare wrote tragedies, comedies, and histories.",
            "Hamlet was likely written around 1600.",
            "The Globe Theatre staged many of his plays in London.",
        ],
        "question": "Where did the Globe Theatre stage many of Shakespeare's plays?",
        "gold": "London",
    },
    {
        "id": "ext_sent_ocean",
        "sents": [
            "The Pacific Ocean is the largest of Earth's oceanic divisions.",
            "It covers more area than all of Earth's land combined.",
            "The Mariana Trench contains the deepest known points.",
        ],
        "question": "Which ocean is Earth's largest oceanic division?",
        "gold": "The Pacific Ocean is the largest of Earth's oceanic divisions.",
    },
]


def build_extraction_row(spec: dict) -> dict:
    paragraph = " ".join(spec["sents"])
    gold = spec["gold"]
    start = paragraph.find(gold)
    if start < 0:
        raise ValueError(f"{spec['id']}: gold {gold!r} not found in paragraph")
    end = start + len(gold)
    if paragraph[start:end] != gold:
        raise ValueError(f"{spec['id']}: span mismatch")
    # ensure unique occurrence for unambiguous offsets
    if paragraph.find(gold, start + 1) >= 0:
        # still OK if identical substring repeats; use first occurrence intentionally
        pass
    return {
        "id": spec["id"],
        "paragraph": paragraph,
        "question": spec["question"],
        "gold": gold,
        "gold_start": start,
        "gold_end": end,
    }


def generate_extraction() -> list[dict]:
    if len(EXTRACTION_SPECS) < 50:
        raise RuntimeError(f"Need 50 specs, have {len(EXTRACTION_SPECS)}")
    rows = [build_extraction_row(s) for s in EXTRACTION_SPECS[:50]]
    # validate all
    for r in rows:
        assert r["paragraph"][r["gold_start"] : r["gold_end"]] == r["gold"], r["id"]
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cls = generate_classification(50)
    ext = generate_extraction()
    assert len(cls) == 50
    assert len(ext) == 50
    write_jsonl(DATA / "classification_50.jsonl", cls)
    write_jsonl(DATA / "extraction_50.jsonl", ext)
    print(f"Wrote {DATA / 'classification_50.jsonl'} ({len(cls)})")
    print(f"Wrote {DATA / 'extraction_50.jsonl'} ({len(ext)})")
    # label balance
    from collections import Counter
    print("classification labels:", dict(Counter(r["label"] for r in cls)))


if __name__ == "__main__":
    main()
