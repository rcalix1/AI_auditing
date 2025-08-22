# AI Auditing, Eplainable AI, and Ethics

* AIBT
* AIBT: Explainable AI and Ethical Considerations - such as SHAP
* AI auditing, risk, and compliance

# 🧠 AI Auditing, Explainability, and Ethics

**90-Minute Masterclass for Business Students**
This session introduces the foundations of AI auditing, from ethical frameworks and explainability tools to real-world audit case studies and constraint-based input testing using Neural Input Optimization (NIO).

---

## 1. Introduction to AI Auditing

AI auditing refers to the process of assessing machine learning systems for fairness, transparency, accuracy, and compliance. As AI becomes embedded in hiring, finance, and law enforcement, there’s growing demand for independent validation of model behavior. This section highlights why auditing is becoming essential in business, legal, and ethical contexts.

---

## 2. Ethics and Frameworks

AI auditing is grounded in ethical principles such as fairness, accountability, and transparency. Government and industry bodies have published frameworks to help organizations implement responsible AI practices. These include principles like explainability, consent, and robustness across the model lifecycle.

---

## 3. Explainable AI (XAI) and Tools

Explainable AI helps us understand why models make specific decisions by attributing importance to different input features. SHAP (SHapley Additive exPlanations) is a leading method that calculates the contribution of each input to the model’s output. Business users gain transparency into decisions affecting credit, hiring, or customer segmentation.

---

## 4. Bias and Fairness Auditing

Fairness audits examine whether a model’s outputs treat different demographic groups equitably. Tools like audit-AI and Aequitas allow you to measure metrics such as demographic parity, equal opportunity, and disparate impact. These tools help identify whether a model could perpetuate discrimination or structural bias.

---

## 5. Legal and Regulatory Trends

As AI regulation matures, law firms and corporate risk managers are beginning to audit AI systems for compliance. Legal frameworks are emerging to mandate transparency, document explainability, and define responsible data use. Businesses that audit their models proactively can avoid penalties and build stakeholder trust.

---

## 6. Case Study: RoBERTaXLM Audit

This case study involved a formal audit of a large language model to uncover latent bias in its token predictions and representations. Auditors applied statistical and visualization techniques to reveal NER biases. This demonstrates the value of independent model audits in high-risk NLP systems.

---

## 7. Auditing in Industry: KPMG & Deloitte Examples

Leading audit firms have integrated AI tools into financial reporting, compliance checks, and fraud detection. These firms now offer services to test AI models for transparency and bias as part of traditional audits. For business leaders, AI auditing is increasingly part of financial governance.

---

## 8. AI Audit Competitions and Challenges

Academic and industry-backed competitions have begun rewarding teams that uncover flaws in AI systems. These challenges show how community-driven auditing can surface problems missed during internal development. They also promote open benchmarking of auditing tools and techniques.

---

## 9. Metrics for AI Auditing

Auditing relies on measurable indicators such as KL Divergence (distribution shifts), Shapley values (feature impact), and fairness ratios. These metrics quantify aspects of transparency, robustness, and equity in AI systems. Interpreting these results is key for turning audits into actionable recommendations.

---

## 10. NIO (Neural Input Optimization) for Auditing

NIO is a constraint-based method that inverts the AI model by optimizing inputs to achieve specific outputs. This is powerful for red-teaming and audit scenarios, such as generating inputs that satisfy policy constraints but expose weaknesses or loopholes. NIO offers a novel way to test model robustness and alignment with business rules.

---

## 11. Live Demo Suggestions (Optional)

Demos can help clarify how auditing tools work in practice. SHAP can visualize feature contributions for credit scoring, while Aequitas can show group fairness metrics. A NIO demo could generate unexpected but policy-compliant inputs to test system boundaries.

---

## 12. Final Takeaways

AI auditing is a growing field blending ethics, law, data science, and business strategy. Organizations that embrace auditing will improve trust, reduce risk, and gain competitive advantage. The future of responsible AI depends on transparency, accountability, and thoughtful governance.

---

## 13. 🔧 ML/AI Auditing Demos with Python

This section introduces three practical auditing tools you can use to evaluate the transparency, fairness, and ethics of AI systems. Each tool is demonstrated with code and includes clear business-use relevance.

---

## ✅ 1. SHAP: Feature Attribution for Explainability

**What it does:** Quantifies how much each input feature contributes to a model's prediction.

**Use case:** Helps explain predictions in finance, healthcare, or HR settings (e.g., why a customer was denied a loan).

**Key Benefits:**
- Breaks down predictions into understandable components
- Makes black-box models more transparent
- Visual and business-friendly

```python
import shap
import xgboost
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# Load sample data and train model
data = load_boston()
X, y = data.data, data.target
model = xgboost.XGBRegressor().fit(X, y)

# SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Plot summary
shap.summary_plot(shap_values, X, feature_names=data.feature_names)
```

---

## 🔎 2. BERT Toxicity Scorer (Language Fairness)

**What it does:** Uses a pretrained BERT model to assess whether a sentence is toxic, hateful, or abusive.

**Use case:** Applied in content moderation, internal policy audits, or social media sentiment analysis.

**Key Benefits:**
- Fast, off-the-shelf scoring for harmful content
- Demonstrates NLP auditing at the content level
- Scalable for business applications

```python
from transformers import pipeline

# Load toxicity classifier
toxicity = pipeline("text-classification", model="unitary/toxic-bert")

# Example sentences
sentences = [
    "I hate you.",
    "Have a great day!",
    "That was a stupid decision."
]

# Score for toxicity
for text in sentences:
    result = toxicity(text)[0]
    print(f"{text} => {result['label']} ({result['score']:.2f})")
```

---

## ⚖️ 3. Group Fairness: Disparity Ratio

**What it does:** Compares outcomes (e.g., approvals) between demographic groups to measure disparity.

**Use case:** Audit hiring, lending, or recommender systems to ensure equal treatment across  protected attributes.

**Key Benefits:**
- Simple and interpretable
- Highlights potential bias
- Can be integrated into compliance reports

```python
import pandas as pd

# Mock predictions
data = pd.DataFrame({
    'gender': ['male', 'female', 'male', 'female', 'female', 'male'],
    'approved': [1, 0, 1, 0, 1, 1]
})

# Group rates
rate_male = data[data.gender == 'male']['approved'].mean()
rate_female = data[data.gender == 'female']['approved'].mean()

# Disparity ratio (female vs male)
ratio = rate_female / rate_male
print(f"Approval rate (female/male): {ratio:.2f}")
```

---

## 📄 Summary

These tools form a practical foundation for AI auditing in business contexts:
- **SHAP**: Interprets model behavior feature-by-feature
- **Toxicity Scorer**: Flags harmful or biased content
- **Fairness Ratio**: Quantifies demographic disparities

These can be applied independently or as part of a larger audit framework to support transparency, accountability, and regulatory compliance.

---


## ARTIFICIAL INTELLIGENCE ETHICS FRAMEWORK FOR THE INTELLIGENCE COMMUNITY

* https://www.intelligence.gov/artificial-intelligence-ethics-framework-for-the-intelligence-community

## Audits

* RobertaXLM audit
* SkyScan audit

## RobertaXLM Audit

* https://assets.iqt.org/pdfs/IQTLabs_RoBERTaAudit_Dec2022_final.pdf/web/viewer.html
* https://www.iqt.org/library/iqt-labs-releases-audit-report-of-roberta-an-large-language-model
* https://ieeexplore.ieee.org/document/10020403

## ISACA

* https://ec.europa.eu/futurium/en/system/files/ged/auditing-artificial-intelligence.pdf

## Other

* https://www.thomsonreuters.com/en-us/posts/technology/auditing-ai-transparency/

## Stanford

* https://hai.stanford.edu/policy/ai-audit-challenge
* https://hai.stanford.edu/ai-audit-challenge-2023-finalists

## KPMG

* https://kpmg.com/xx/en/our-insights/ai-and-technology/ai-in-financial-reporting-and-audit.html

## AI audit metrics

* SHapley
* https://www.edpb.europa.eu/system/files/2024-06/ai-auditing_checklist-for-ai-auditing-scores_edpb-spe-programme_en.pdf
* https://oecd.ai/en/catalogue/metrics?page=1
* https://pypi.org/project/audit-AI/
* https://github.com/pymetrics/audit-ai
* https://github.com/dssg/aequitas
* https://www.google.com/search?q=Python+ai+auditing+kl+divergence&client=firefox-b-1-d&sca_esv=e78c9a996bbe5715&biw=1920&bih=947&ei=K7AaZ-SSNpLcptQPpfSt6Ac&ved=0ahUKEwik1KbV76eJAxUSrokEHSV6C30Q4dUDCBE&uact=5&oq=Python+ai+auditing+kl+divergence&gs_lp=Egxnd3Mtd2l6LXNlcnAiIFB5dGhvbiBhaSBhdWRpdGluZyBrbCBkaXZlcmdlbmNlMgUQIRigATIFECEYoAEyBRAhGKABMgUQIRigAUjWMFCrGViMLHACeACQAQCYAY0BoAHJCqoBAzkuNbgBA8gBAPgBAZgCD6ACyQrCAgoQABiwAxjWBBhHwgIFECEYqwKYAwCIBgGQBgiSBwQxMC41oAfLMw&sclient=gws-wiz-serp
* https://www.google.com/search?q=Python+ai+auditing+shapley&client=firefox-b-1-d&sca_esv=e78c9a996bbe5715&biw=1920&bih=947&ei=8bAaZ5S9NqyoptQPwMiwyAc&ved=0ahUKEwjU-duz8KeJAxUslIkEHUAkDHkQ4dUDCBE&uact=5&oq=Python+ai+auditing+shapley&gs_lp=Egxnd3Mtd2l6LXNlcnAiGlB5dGhvbiBhaSBhdWRpdGluZyBzaGFwbGV5MgUQIRigATIFECEYoAEyBRAhGKABMgUQIRigATIFECEYoAFIsBNQ0gVYsRFwAXgBkAEAmAGOAaAB7gWqAQM0LjO4AQPIAQD4AQGYAgigAogGwgIKEAAYsAMY1gQYR8ICBRAhGKsCwgIHECEYoAEYCpgDAIgGAZAGCJIHAzUuM6AH8Rg&sclient=gws-wiz-serp

## Law Firms 

* https://www.luminos.law

## RLHF

* How to reduce Bias
* Bias BERT scorer
* https://github.com/rcalix1/TransferLearning/blob/main/RLHF/ITS530-DavidHigley-gpt2-phish-spam.ipynb


