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
├── requirements.txt
├── data/
│   └── README.md
├── figures/
│   └── README.md
├── .gitignore
└── LICENSE
```

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

## Tools

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- SciPy

## Author

**Muhammad Jabran**  
M.Sc. Food System Sciences, University of Bayreuth

## License

This project code is released under the MIT License. The original dataset remains subject to the terms of its source repository.
