import ollama


MODEL_NAME = "qwen3:4b"


def generate_quality_explanation(
    quality_report,
    rag_result
):
    """
    Generate a natural-language quality explanation
    using a local Ollama LLM and retrieved foundry knowledge.
    """

    retrieved_knowledge = "\n\n".join(
        [
            (
                f"Source: {item['source']}\n"
                f"Similarity: {item['similarity']:.4f}\n"
                f"Content:\n{item['content']}"
            )
            for item in rag_result["results"]
        ]
    )

    prompt = f"""
You are an AI assistant for an industrial foundry
quality inspection system.

Your task is to explain the supplied machine-learning
results clearly and suggest practical investigation steps.

Use ONLY the supplied quality report and retrieved
knowledge.

Do not invent measurements, defects, causes, or facts.

Important rules:

- ML predictions are predictions, not guaranteed truth.
- SHAP shows model feature contributions, not causation.
- Grad-CAM highlights image regions influential to the
  model, not definitive physical defects.
- If the process and visual models disagree, explain the
  disagreement rather than choosing one model as correct.
- Recommendations are investigation steps, not guaranteed
  fixes.

QUALITY REPORT:

{quality_report}


RETRIEVED FOUNDRY KNOWLEDGE:

{retrieved_knowledge}


Give the response in exactly this structure:

Summary:
Give a short explanation of the current situation.

Analysis:
Explain the process prediction, visual prediction,
model agreement/disagreement, confidence, and important
SHAP/Grad-CAM evidence.

Recommended Investigation:
Give 3 to 5 practical investigation steps based on the
retrieved knowledge.

Limitations:
Mention important uncertainty or limitations.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]