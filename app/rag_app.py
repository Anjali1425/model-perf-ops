import os
from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import observe, get_client

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Hardcoded doc store — no vector DB needed for this demo
DOCS = [
    "Return policy: items can be returned within 30 days with a receipt.",
    "Shipping: standard shipping takes 5-7 business days.",
    "Expedited shipping: 2-day shipping is available for an additional $15.",
    "Payment methods: we accept Visa, Mastercard, AmEx, and PayPal.",
    "Order cancellation: orders can be cancelled within 24 hours of placement.",
    "Damaged items: if your item arrives damaged, contact support within 7 days.",
    "Gift cards: gift cards are non-refundable and do not expire.",
    "Store hours: customer support is available Monday–Friday, 9am–6pm EST.",
    "International shipping: we ship to Canada and the UK; duties are the buyer's responsibility.",
    "Loyalty program: earn 1 point per dollar spent; 100 points = $1 reward.",
]


def retrieve(query, docs=DOCS, top_k=2):
    """Keyword-overlap retrieval. Simple and sufficient for a small demo."""
    scored = sorted(
        docs,
        key=lambda d: -sum(w in d.lower() for w in query.lower().split())
    )
    return scored[:top_k]


@observe(as_type="generation")
def answer(query):
    context = "\n".join(retrieve(query))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )
    get_client().update_current_generation(
        model="claude-haiku-4-5-20251001",
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "total": response.usage.input_tokens + response.usage.output_tokens,
        },
    )
    return response.content[0].text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask the RAG app one or more questions.")
    parser.add_argument("question", nargs="?", help="A single question to ask")
    parser.add_argument("--file", help="Path to a text file with one question per line")
    parser.add_argument("-i", "--interactive", action="store_true", help="Ask questions one at a time")
    args = parser.parse_args()

    if args.interactive:
        print("Type a question and press Enter. Type 'quit' or Ctrl+D to stop.\n")
        while True:
            try:
                q = input("Q: ").strip()
            except EOFError:
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            print(f"A: {answer(q)}\n")
    else:
        if args.question:
            questions = [args.question]
        else:
            path = args.file or os.path.join(os.path.dirname(__file__), "questions.txt")
            with open(path) as f:
                questions = [line.strip() for line in f if line.strip()]

        for q in questions:
            print(f"Q: {q}")
            print(f"A: {answer(q)}\n")
