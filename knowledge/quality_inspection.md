# Quality Inspection

## Overview

Quality inspection is the process of evaluating a manufactured component
against defined quality requirements.

In an intelligent manufacturing system, inspection can combine process data,
visual inspection, historical information, and expert knowledge.

## Visual Inspection

Visual inspection can be used to identify visible surface conditions such as:

- Cracks
- Porosity
- Surface marks
- Incomplete filling
- Cold shut indications
- Surface irregularities
- Other visible abnormalities

Visual inspection provides evidence about the physical appearance of the
component.

## Image Classification

An image classification model can classify a casting image into predefined
categories.

For the current Foundry Quality Intelligence system, the computer vision
model uses two classes:

- OK
- Defective

The model produces a prediction together with probabilities for the two
classes.

A high probability indicates the model's confidence in its prediction, but
does not guarantee that the prediction is correct.

## Process-Based Inspection

Process-based quality analysis uses manufacturing parameters to identify
patterns associated with different quality classes.

The current process model uses multiple process parameters simultaneously.

The model predicts one of four quality classes:

- Quality 1
- Quality 2
- Quality 3
- Quality 4

The meaning of these quality classes depends on the dataset and quality
definition used to train the model.

## Combining Process and Visual Inspection

Process-based predictions and visual inspection can provide complementary
information.

Possible situations include:

### Agreement

The process model and visual model indicate consistent results.

This provides two sources of evidence supporting the same inspection
outcome.

### Disagreement

The process model and visual model produce different results.

For example:

- Process model indicates a lower quality class
- Visual model predicts OK

Such disagreement should be treated as a signal for further investigation
rather than automatically selecting one model as correct.

## Model Confidence

Confidence values should be considered when interpreting predictions.

A prediction with high confidence can still be incorrect.

A prediction with relatively low confidence may indicate that the input is
more difficult for the model to classify.

Confidence should therefore be considered together with:

- Model performance on the test dataset
- Input quality
- Process conditions
- Visual evidence
- Historical observations

## Explainable AI

Explainable AI methods can help users understand model predictions.

### SHAP

SHAP can identify process features that contributed to a model prediction.

For the process model, SHAP explanations can highlight important parameters
such as:

- Cycle time
- Injection pressure
- Plasticizing time
- Time to fill
- Mold temperature
- Closing force
- Clamping force

SHAP values describe the contribution of features to a model output. They
should not automatically be interpreted as proof of physical causation.

### Grad-CAM

Grad-CAM can highlight image regions that influenced a computer vision
model's prediction.

The highlighted regions represent areas that were important to the model's
decision.

Grad-CAM should be treated as a model explanation rather than a definitive
identification of a physical defect.

## Inspection Workflow

A combined intelligent inspection workflow can follow these stages:

1. Collect process parameters.
2. Collect the casting image.
3. Run the process-quality model.
4. Run the visual inspection model.
5. Generate SHAP explanations for process predictions.
6. Generate Grad-CAM explanations for image predictions.
7. Compare the model outputs.
8. Identify agreement or disagreement.
9. Retrieve relevant technical knowledge.
10. Present the findings for human review.

## Human Review

Automated predictions should support human decision-making rather than
replace appropriate inspection procedures.

When model outputs disagree, when confidence is low, or when the result is
outside the conditions represented in the training data, additional
inspection or expert review may be appropriate.