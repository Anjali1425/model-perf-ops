import os
from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import observe

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


@observe()
def answer(query):
    context = "\n".join(retrieve(query))
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )
    return response.content[0].text


if __name__ == "__main__":
    test_query = "How long do I have to return something?"
    print(f"Q: {test_query}")
    print(f"A: {answer(test_query)}")
