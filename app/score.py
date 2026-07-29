"""
RAGAS evaluation script for the RAG app.

Builds an eval dataset from real retrieve() and answer() calls, then scores it
with real RAGAS metrics. Judge LLM is Claude (needs ANTHROPIC_API_KEY in .env);
embeddings run locally via sentence-transformers, so no embedding API key is needed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from anthropic import Anthropic
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from rag_app import retrieve, answer

# Eval set: 10 questions with ground-truth answers
EVAL_QUESTIONS = [
    ("How long do I have to return something?",        "Items can be returned within 30 days with a receipt."),
    ("How long does standard shipping take?",          "Standard shipping takes 5-7 business days."),
    ("Is expedited shipping available?",               "Yes, 2-day shipping is available for an additional $15."),
    ("What payment methods do you accept?",            "We accept Visa, Mastercard, AmEx, and PayPal."),
    ("Can I cancel my order?",                         "Orders can be cancelled within 24 hours of placement."),
    ("What do I do if my item arrives damaged?",       "Contact support within 7 days of receiving a damaged item."),
    ("Are gift cards refundable?",                     "Gift cards are non-refundable and do not expire."),
    ("When is customer support available?",            "Customer support is available Monday–Friday, 9am–6pm EST."),
    ("Do you ship internationally?",                   "We ship to Canada and the UK; duties are the buyer's responsibility."),
    ("How does the loyalty program work?",             "Earn 1 point per dollar spent; 100 points equals $1 in rewards."),
]


def build_eval_dataset():
    """Build the RAGAS dataset using real app outputs."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for question, ground_truth in EVAL_QUESTIONS:
        retrieved = retrieve(question)
        response = answer(question)

        questions.append(question)
        answers.append(response)
        contexts.append(retrieved)          # list of strings per question
        ground_truths.append(ground_truth)

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation(dataset):
    """Score the dataset with RAGAS, judged by Claude with local embeddings."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to .env before running.")

    llm = llm_factory(
        "claude-haiku-4-5-20251001",
        provider="anthropic",
        client=Anthropic(api_key=api_key),
    )
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    return result.to_pandas()[["faithfulness", "answer_relevancy", "context_precision"]].mean()


if __name__ == "__main__":
    print("Building eval dataset...")
    dataset = build_eval_dataset()
    print(f"Eval set: {len(dataset)} questions\n")

    print("Running evaluation...")
    scores = run_evaluation(dataset)

    print("\n=== RAGAS Scores ===")
    for metric, score in scores.items():
        status = "OK" if score >= 0.85 else "BELOW THRESHOLD"
        print(f"  {metric:<25} {score:.2f}  [{status}]")
