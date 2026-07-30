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
from langchain_community.embeddings import HuggingFaceEmbeddings as LangchainHFEmbeddings
from langfuse import get_client
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from rag_app import retrieve, answer

METRICS = [faithfulness, answer_relevancy, context_precision]

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
    """Build the RAGAS dataset using real app outputs. Also returns each
    question's Langfuse trace ID so scores can be attached back to it."""
    questions, answers, contexts, ground_truths, trace_ids = [], [], [], [], []
    langfuse = get_client()

    for question, ground_truth in EVAL_QUESTIONS:
        with langfuse.start_as_current_span(name="eval_question"):
            retrieved = retrieve(question)
            response = answer(question)
            trace_ids.append(langfuse.get_current_trace_id())

        questions.append(question)
        answers.append(response)
        contexts.append(retrieved)          # list of strings per question
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })
    return dataset, trace_ids


def run_evaluation(dataset, trace_ids):
    """Score the dataset with RAGAS, judged by Claude with local embeddings,
    then push each metric back to its matching Langfuse trace as a score."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to .env before running.")

    llm = llm_factory(
        "claude-haiku-4-5-20251001",
        provider="anthropic",
        client=Anthropic(api_key=api_key),
    )
    # claude-haiku-4-5 rejects requests that set both temperature and top_p;
    # ragas's instructor adapter defaults to setting both, so drop top_p.
    llm.model_args.pop("top_p", None)
    embeddings = LangchainEmbeddingsWrapper(
        LangchainHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()

    langfuse = get_client()
    metric_names = [m.name for m in METRICS]
    for trace_id, (_, row) in zip(trace_ids, df.iterrows()):
        for metric in metric_names:
            langfuse.create_score(
                trace_id=trace_id,
                name=metric,
                value=float(row[metric]),
                data_type="NUMERIC",
            )
    langfuse.flush()

    return df[metric_names].mean()


if __name__ == "__main__":
    print("Building eval dataset...")
    dataset, trace_ids = build_eval_dataset()
    print(f"Eval set: {len(dataset)} questions\n")

    print("Running evaluation...")
    scores = run_evaluation(dataset, trace_ids)

    print("\n=== RAGAS Scores ===")
    for metric, score in scores.items():
        status = "OK" if score >= 0.85 else "BELOW THRESHOLD"
        print(f"  {metric:<25} {score:.2f}  [{status}]")
