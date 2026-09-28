# Foundry Troubleshooting Guide

## Overview

Troubleshooting involves investigating abnormal process behaviour or
quality results and identifying possible contributing factors.

The purpose of troubleshooting is to organize evidence and guide further
investigation. A suspected factor should not automatically be treated as
the confirmed root cause.

## Process Quality Concerns

When the process model predicts a lower quality class, the following
information can be reviewed:

- Process parameter values
- Model confidence
- Important SHAP features
- Historical process behaviour
- Visual inspection results
- Material conditions
- Machine conditions

The process prediction should be interpreted in the context of the data
used to train the model.

## Visual Defect Concerns

When the image model predicts a defective casting, inspection can consider:

- Predicted class
- Defective probability
- Model confidence
- Grad-CAM highlighted regions
- Physical appearance of the casting
- Defect location and shape

The Grad-CAM result should be treated as supporting evidence about the
model's decision, not as definitive proof of a physical defect.

## Model Disagreement

Model disagreement occurs when the process model and visual model provide
different indications.

For example:

- Process model indicates a lower quality class
- Visual model predicts OK

Or:

- Process model indicates a better quality class
- Visual model predicts Defective

When disagreement occurs, recommended investigation steps include:

1. Review the process parameters.
2. Review the important SHAP features.
3. Inspect the casting image.
4. Review the Grad-CAM explanation.
5. Compare the result with historical process behaviour.
6. Perform additional physical inspection when appropriate.

The system should not automatically assume that one model is correct and the
other is incorrect.

## Low Confidence Predictions

When model confidence is relatively low, the result should be treated with
additional caution.

Possible investigation steps include:

- Verify the input values.
- Check whether the image quality is suitable.
- Compare the input with known training examples.
- Review model performance on similar samples.
- Request additional inspection.

## Process Parameter Investigation

When a process parameter appears as an important SHAP feature, it can be
selected for further investigation.

Important features currently observed in the process model include:

- Cycle time
- Injection pressure
- Plasticizing time
- Time to fill
- Mold temperature
- Closing force
- Clamping force

The presence of a feature in a SHAP explanation indicates that it
contributed to the model prediction. It does not establish that the feature
is the physical cause of the quality outcome.

## Investigation Workflow

A general troubleshooting workflow can be:

1. Identify the model prediction.
2. Check model confidence.
3. Review process parameters.
4. Review SHAP explanations.
5. Review visual prediction.
6. Review Grad-CAM explanation.
7. Retrieve relevant technical information.
8. Compare multiple sources of evidence.
9. Perform additional inspection if required.
10. Record the findings for future analysis.

## Human-in-the-Loop Approach

The intelligent inspection system should support engineers and quality
personnel by organizing information and highlighting relevant evidence.

Final decisions about production quality, process changes, or corrective
actions should consider appropriate human expertise and inspection
procedures.

## Important Limitation

Machine learning predictions describe patterns learned from the available
training data.

If a new process condition, material, machine configuration, or defect type
is substantially different from the training data, model performance may
differ from the observed test performance.

Predictions should therefore be interpreted within the scope of the training
data and validated when used in a real production environment.