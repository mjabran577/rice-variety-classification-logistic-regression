# Rice Variety Classification Using Logistic Regression

A machine-learning portfolio project that classifies **Cammeo** and **Osmancik** rice varieties from physical grain measurements using Logistic Regression.

## Project overview

The analysis uses the **Rice (Cammeo and Osmancik)** dataset from the UCI Machine Learning Repository. It contains **3,810 observations** and seven numerical grain features:

- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- Extent

The target variable contains two rice varieties: **Cammeo** and **Osmancik**.

## Workflow

1. Load and inspect the dataset
2. Explore class distribution and feature relationships
3. Encode the target variable
4. Split the data into 70% training and 30% testing data
5. Train a Logistic Regression classifier
6. Evaluate the model using accuracy, confusion matrix, ROC-AUC, ROC curve, and 5-fold cross-validation

For reproducibility, the train-test split uses `random_state=732084`, matching the original university analysis.

## Results

| Metric | Result |
|---|---:|
| Test accuracy | **93.53%** |
| ROC-AUC | **0.9816** |
| Mean 5-fold cross-validation accuracy | **93.02%** |

These results show that the measured grain characteristics provide strong separation between the two rice varieties.

## Visual results

### Class distribution
![Class distribution](figures/class_distribution.png)

### Area by rice variety
![Area boxplot](figures/area_boxplot.png)

### Correlation heatmap
![Correlation heatmap](figures/correlation_heatmap.png)

### Confusion matrix
![Confusion matrix](figures/confusion_matrix.png)

### ROC curve
![ROC curve](figures/roc_curve.png)

## Why this matters

A similar classification workflow could support:

- automated rice sorting
- food quality control
- agricultural research
- reduction of manual inspection effort
- more consistent classification in processing facilities

## Repository structure

```text
.
├── README.md
├── rice_classification.py
├── generate_report.py
├── requirements.txt
├── data/
│   └── README.md
├── figures/
│   ├── class_distribution.png
│   ├── area_boxplot.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── notebooks/
│   └── Rice.ipynb
├── report/
│   └── Rice_Executive_Summary_BT732084.pdf
├── .github/workflows/
│   └── build-assets.yml
├── .gitignore
└── LICENSE
```

## Notebook and report

- [`notebooks/Rice.ipynb`](notebooks/Rice.ipynb) — cleaned, step-by-step Jupyter notebook based on the original university analysis.
- [`report/Rice_Executive_Summary_BT732084.pdf`](report/Rice_Executive_Summary_BT732084.pdf) — concise portfolio report with methodology, results, figures, discussion, and applications.

## Run the project

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Download the **Rice (Cammeo and Osmancik)** dataset from the UCI Machine Learning Repository and save the ARFF file as:

```text
data/Rice_Cammeo_Osmancik.arff
```

Then run:

```bash
python rice_classification.py
```

The script trains the model, prints the evaluation metrics, and saves the main visualizations in the `figures/` directory.

To rebuild the PDF report after generating the figures, install ReportLab and run:

```bash
pip install reportlab
python generate_report.py
```

The included GitHub Actions workflow can also rebuild the figures and PDF report automatically from the UCI source dataset.

## Tools

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- SciPy
- Jupyter Notebook
- GitHub Actions

## Author

**Muhammad Jabran**  
M.Sc. Food System Sciences, University of Bayreuth

## License

This project code is released under the MIT License. The original dataset remains subject to the terms of its source repository.
