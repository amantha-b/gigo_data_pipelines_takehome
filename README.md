# GIGO in Practice – Data Quality Pipeline Teaching Package

**Author:** Amantha Bhaskarabhatla  
**NUID:** 002300618  

This repository is an educational package for teaching **GIGO (Garbage In, Garbage Out)** through a hands-on **data-quality pipeline in Python**.

The focus is not on fancy models, but on something more fundamental:

> If your data is garbage, your analysis and models will also be garbage.



## 1. Concept Overview

This project turns the idea of **GIGO** into a concrete, reproducible workflow that students can run, extend, and critique.

We work with a synthetic but realistic **transaction dataset** that includes:

- `customer_id` – integer customer ID  
- `age` – customer age  
- `country` – country code (`US`, `UK`, `IN`, `DE`, `CA`)  
- `product_category` – e.g., `Electronics`, `Clothing`, `Grocery`, `Beauty`  
- `transaction_amount` – purchase amount  

From there, we deliberately inject “garbage” into the data:

- Missing values  
- Impossible ages (negative, extremely large)  
- Invalid country and product codes  
- Negative and extreme transaction amounts  
- Duplicate rows  

The teaching goal is to show how **data validation + cleaning** changes a simple business metric:

> **Average transaction amount by country – before vs after cleaning** → a concrete GIGO moment.

Along the way, students are exposed to:

- Data quality dimensions (completeness, validity, uniqueness, reasonableness)  
- Data contracts / validation rules  
- Simple anomaly detection / “suspicious” flags  
- The mindset of **computational skepticism** toward data


## 2. Learning Objectives

After working through this package, students will be able to:

- Explain what **GIGO** means in the context of data science and ML.
- Describe basic **data quality dimensions**:
  - Completeness, validity, uniqueness, reasonableness.
- Design a simple **data contract** (validation rules) for a tabular dataset.
- Implement a small **data-quality report** that surfaces missing and invalid values.
- Build a **cleaning pipeline** that:
  - Removes duplicates  
  - Fixes invalid ranges and categories  
  - Handles outliers and missing data
- Quantify how GIGO affects a **business metric** by comparing:
  - Average transaction amount by country **before vs after cleaning**
- Practice **computational skepticism** by questioning assumptions baked into datasets and validation rules.


## 3. Installation

### 3.1. Clone the repository

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

Using pip and requirements.txt

python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


The core dependencies used in this project are:

- python (3.10+)

- pandas

- numpy

- matplotlib

- jupyterlab / notebook (for running the notebooks locally)

On Google Colab, most of these are already available; you can simply upload the notebooks and run.


Repo Structure

gigo/
├── data/
│   └── transactions_dirty.csv          # Example corrupted dataset
├── docu pdf/
│   ├── gigo_amantha.pdf               # Tutorial documentation PDF / concept writeup
│   ├── pedagogical_report_amantha.pdf # Pedagogical report (teaching reflection)
│   └── self_assessment_amantha.pdf    # Self-assessment / percentile rationale
├── notebooks/
│   ├── gigo_amantha.ipynb             # Full teaching notebook (with theory + code)
│   └── gigo_template_amantha.ipynb    # Starter notebook with TODOs for learners
├── gigo_pipeline_amantha.py           # Main pipeline script (end-to-end implementation)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file

4. Usage Examples
4.1. Run the Main Implementation Script

If you want to see the full GIGO pipeline in one go, from the gigo/ folder:

python gigo_pipeline_amantha.py


This script will:

- Generate a synthetic transaction dataset.

- Corrupt it with realistic data issues (missing values, invalid categories, outliers, duplicates).

- Print a data-quality report BEFORE cleaning (per-column missing/invalid percentages).

- Apply a rule-based cleaning pipeline.

- Print a data-quality report AFTER cleaning.

- Compute and display average transaction amount by country before vs after cleaning.

- Use this as the reference implementation that matches the logic in the teaching notebook.

2. Full Teaching Notebook – notebooks/gigo_amantha.ipynb

This is the main tutorial / teaching notebook.

Open it:

notebooks/gigo_amantha.ipynb

The notebook includes:

- Conceptual explanation of GIGO and data quality.

- Synthetic data creation and controlled corruption.

- Definition of a data contract via validation rules.

- A reusable data_quality_report(...) function.

- A cleaning pipeline that: Treats improbable values (e.g., impossible ages, out-of-range amounts); Handles invalid categories and missing values; Manages duplicates.

- Visual and numeric comparison of metrics before vs after cleaning.

- Reflection prompts and discussion of how hidden data issues distort downstream analysis.

- Progressive exercises (easy → medium → hard) for learners to extend the pipeline.

4.3. Starter Notebook for Learners – notebooks/gigo_template_amantha.ipynb

This notebook mirrors the structure of the full one, but includes TODOs and missing code blocks that learners need to fill in.

Typical tasks include:

- Completing the validation_rules / data contract.

- Implementing the data_quality_report(df, rules) helper function.

- Implementing parts of the cleaning logic.

- Computing and comparing per-country metrics before vs after cleaning.

- Answering reflection questions and exercise prompts at the end.

This is designed as a guided practice version of the full tutorial.

4.4 Example Dataset – data/transactions_dirty.csv

Once generated by the notebook or script, the main example dataset lives at:

data/transactions_dirty.csv

5. Documentation PDFs

Inside docu pdf/ you’ll find supporting written materials:

gigo_amantha.pdf -- Tutorial-style documentation that explains the concept, context, and implementation steps.

pedagogical_report_amantha.pdf -- A 6–10 page report describing teaching philosophy, concept deep dive, implementation analysis, and assessment strategy.

self_assessment_amantha.pdf -- A short self-evaluation discussing where this work sits among peer projects and why.

These PDFs are meant to complement the notebook and video for a complete teaching package.

6. Video Walkthrough

I've created an unlisted video on Youtube with the link: https://youtu.be/AdSkgQC8uVM

7. Acknowledgements

INFO 7390: Advanced Data Science and Architecture (Northeastern University) – for framing GIGO, data quality, and computational skepticism.

Open-source tools such as pandas, NumPy, and matplotlib, which power the data manipulation and visualization in this teaching package.

