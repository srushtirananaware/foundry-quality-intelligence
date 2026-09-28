from pathlib import Path
import sys


# Project root and src path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))


# ML / CV modules
from models.process_xgboost import predict_quality
from models.process_image import predict_image
from models.process_explainability import explain_process_prediction
from gradcam import generate_gradcam


# RAG modules
from rag.retriever import (
    load_embedding_model,
    load_knowledge_index,
    search_knowledge,
    build_quality_query,
)

from llm.quality_explainer import generate_quality_explanation

def make_decision(process_assessment, visual_assessment):

    quality = process_assessment["predicted_quality"]
    process_probabilities = process_assessment["probabilities"]

    visual_prediction = visual_assessment["prediction"]
    visual_confidence = visual_assessment["confidence"]

    process_confidence = max(process_probabilities.values())

    # Temporary prototype rule
    process_concern = quality in [1, 2]
    visual_concern = visual_prediction == "Defective"

    if process_concern and visual_concern:
        status = "HIGH CONCERN"
        explanation = (
            "Both the process model and visual inspection indicate "
            "a potential quality issue."
        )

    elif process_concern and not visual_concern:
        status = "MODEL DISAGREEMENT"
        explanation = (
            "The process model indicates a lower quality class, "
            "while the visual inspection predicts OK. "
            "Manual review is recommended."
        )

    elif not process_concern and visual_concern:
        status = "MODEL DISAGREEMENT"
        explanation = (
            "The process model indicates a better quality class, "
            "while the visual inspection predicts a defect. "
            "Manual review is recommended."
        )

    else:
        status = "NO IMMEDIATE CONCERN"
        explanation = (
            "Both the process model and visual inspection indicate "
            "no immediate quality concern."
        )

    if process_confidence < 0.60 or visual_confidence < 0.60:
        status = "LOW CONFIDENCE"
        explanation = (
            "One or more model predictions have relatively low confidence. "
            "Manual inspection is recommended."
        )

    return {
        "status": status,
        "explanation": explanation,
        "process_confidence": process_confidence,
        "visual_confidence": visual_confidence,
    }


def generate_quality_report(
    process_assessment,
    visual_assessment,
    decision
):
    """
    Convert model outputs into a structured quality report.
    """

    report = {
        "process": {
            "predicted_quality": process_assessment["predicted_quality"],
            "confidence": max(
                process_assessment["probabilities"].values()
            ),
            "probabilities": process_assessment["probabilities"],
            "key_factors": process_assessment["key_factors"],
        },

        "visual": {
            "prediction": visual_assessment["prediction"],
            "confidence": visual_assessment["confidence"],
            "defective_probability": visual_assessment[
                "defective_probability"
            ],
            "ok_probability": visual_assessment[
                "ok_probability"
            ],
            "gradcam_path": visual_assessment["gradcam_path"],
        },

        "decision": {
            "status": decision["status"],
            "process_confidence": decision["process_confidence"],
            "visual_confidence": decision["visual_confidence"],
            "explanation": decision["explanation"],
        },
    }

    return report


def retrieve_quality_knowledge(quality_report, top_k=3):
    """
    Retrieve relevant foundry knowledge based on the
    current quality assessment.
    """

    query = build_quality_query(quality_report)

    model = load_embedding_model()

    chunks, embeddings = load_knowledge_index()

    if chunks is None or embeddings is None:
        raise FileNotFoundError(
            "RAG knowledge index not found. "
            "Run src/rag/retriever.py first."
        )

    results = search_knowledge(
        query,
        chunks,
        embeddings,
        model,
        top_k=top_k
    )

    return {
        "query": query,
        "results": results,
    }


def analyze_quality(process_values, image_path):
    """
    Run both the process-quality and casting-image models,
    then interpret their outputs using the decision layer.
    """

    # --------------------------------------------------
    # Process quality prediction
    # --------------------------------------------------

    predicted_quality, quality_probabilities = predict_quality(
        process_values
    )

    process_explanations = explain_process_prediction(
        process_values,
        top_n=5
    )

    process_assessment = {
        "predicted_quality": predicted_quality,
        "probabilities": quality_probabilities,
        "key_factors": process_explanations,
    }

    # --------------------------------------------------
    # Image defect prediction
    # --------------------------------------------------

    image_result = predict_image(image_path)

    # --------------------------------------------------
    # Grad-CAM explanation
    # --------------------------------------------------

    gradcam_result = generate_gradcam(image_path)

    visual_assessment = {
        "prediction": image_result["prediction"],
        "confidence": image_result["confidence"],
        "defective_probability": image_result["defective_probability"],
        "ok_probability": image_result["ok_probability"],
        "gradcam_path": str(gradcam_result["gradcam_path"]),
    }

    # --------------------------------------------------
    # Combined decision
    # --------------------------------------------------

    decision = make_decision(
        process_assessment,
        visual_assessment
    )

    # --------------------------------------------------
    # Structured quality report
    # --------------------------------------------------

    quality_report = generate_quality_report(
        process_assessment,
        visual_assessment,
        decision
    )

    # --------------------------------------------------
    # RAG knowledge retrieval
    # --------------------------------------------------

    rag_result = retrieve_quality_knowledge(
        quality_report,
        top_k=3
    )

    llm_explanation = generate_quality_explanation(
    quality_report,
    rag_result
    )

    result = {
        "process_assessment": process_assessment,
        "visual_assessment": visual_assessment,
        "decision": decision,
        "quality_report": quality_report,
        "rag": rag_result,
        "llm_explanation": llm_explanation,
    }

    return result


if __name__ == "__main__":

    # --------------------------------------------------
    # Sample process parameters
    # --------------------------------------------------

    sample_process_values = [
        106.0,   # Melt temperature
        81.2,    # Mold temperature
        7.0,     # Time to fill
        3.2,     # Plasticizing time
        75.0,    # Cycle time
        900.0,   # Closing force
        920.0,   # Clamping force peak
        117.0,   # Torque peak
        105.0,   # Torque mean
        146.2,   # Back pressure
        910.0,   # Injection pressure
        8.8,     # Screw position
        18.75,   # Shot volume
    ]

    # --------------------------------------------------
    # Sample casting image
    # --------------------------------------------------

    sample_image = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "casting_data"
        / "casting_data"
        / "test"
        / "def_front"
        / "cast_def_0_1059.jpeg"
    )

    # --------------------------------------------------
    # Run quality intelligence
    # --------------------------------------------------

    result = analyze_quality(
        sample_process_values,
        sample_image
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("=== FOUNDRY QUALITY INTELLIGENCE ===")
    print()

    # --------------------------------------------------
    # PROCESS ASSESSMENT
    # --------------------------------------------------

    print("PROCESS ASSESSMENT")
    print("------------------")

    print(
        "Predicted Quality:",
        result["process_assessment"]["predicted_quality"]
    )

    print("\nQuality Probabilities:")

    for quality, probability in result[
        "process_assessment"
    ]["probabilities"].items():

        print(f"{quality}: {probability:.2%}")

    print()

    print("KEY PROCESS FACTORS")
    print("-------------------")

    for index, factor in enumerate(
        result["process_assessment"]["key_factors"],
        start=1
    ):

        direction = (
            "supports prediction"
            if factor["shap_value"] > 0
            else "pushes against prediction"
        )

        print(
            f"{index}. {factor['feature']} "
            f"({direction})"
        )

    # --------------------------------------------------
    # VISUAL ASSESSMENT
    # --------------------------------------------------

    print()
    print("VISUAL ASSESSMENT")
    print("-----------------")

    print(
        "Prediction:",
        result["visual_assessment"]["prediction"]
    )

    print(
        f"Confidence: "
        f"{result['visual_assessment']['confidence']:.2%}"
    )

    print(
        f"Defective probability: "
        f"{result['visual_assessment']['defective_probability']:.2%}"
    )

    print(
        f"OK probability: "
        f"{result['visual_assessment']['ok_probability']:.2%}"
    )

    print(
        "Grad-CAM explanation:",
        result["visual_assessment"]["gradcam_path"]
    )

    # --------------------------------------------------
    # QUALITY DECISION
    # --------------------------------------------------

    print()
    print("QUALITY DECISION")
    print("----------------")

    print(
        "Status:",
        result["decision"]["status"]
    )

    print(
        "Process Model Confidence:",
        f'{result["decision"]["process_confidence"]:.2%}'
    )

    print(
        "Visual Model Confidence:",
        f'{result["decision"]["visual_confidence"]:.2%}'
    )

    print(
        "Explanation:",
        result["decision"]["explanation"]
    )

    # --------------------------------------------------
    # STRUCTURED QUALITY REPORT
    # --------------------------------------------------

    report = result["quality_report"]

    print()
    print("STRUCTURED QUALITY REPORT")
    print("-------------------------")

    print(
        f"Process Quality: "
        f"Quality {report['process']['predicted_quality']}"
    )

    print(
        f"Process Confidence: "
        f"{report['process']['confidence']:.2%}"
    )

    print(
        f"Visual Result: "
        f"{report['visual']['prediction']}"
    )

    print(
        f"Visual Confidence: "
        f"{report['visual']['confidence']:.2%}"
    )

    print(
        f"Overall Status: "
        f"{report['decision']['status']}"
    )

    print()
    print("Key Process Factors:")

    for factor in report["process"]["key_factors"]:
        print(f"• {factor['feature']}")

    print()
    print(
        "Visual Explanation:",
        report["visual"]["gradcam_path"]
    )

    # --------------------------------------------------
    # RAG KNOWLEDGE
    # --------------------------------------------------

    print()
    print("RETRIEVED FOUNDRY KNOWLEDGE")
    print("---------------------------")

    print()
    print("RAG Query:")
    print(result["rag"]["query"])

    print()

    for index, knowledge in enumerate(
        result["rag"]["results"],
        start=1
    ):

        print(
            f"{index}. "
            f"{knowledge['source']} "
            f"(chunk {knowledge['chunk_id']})"
        )

        print(
            f"Similarity: "
            f"{knowledge['similarity']:.4f}"
        )

        print(knowledge["content"])

        print("-" * 60)

    # --------------------------------------------------
# LLM QUALITY EXPLANATION
# --------------------------------------------------

print()
print("AI QUALITY EXPLANATION")
print("----------------------")

print(
    result["llm_explanation"]
)