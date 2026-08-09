"""Generate a concise portfolio PDF report from the project results and figures."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("report/Rice_Executive_Summary_BT732084.pdf")
FIGURES = Path("figures")


def add_figure(story, filename: str, caption: str, width: float = 14 * cm):
    path = FIGURES / filename
    if path.exists():
        img = Image(str(path))
        img._restrictSize(width, 10 * cm)
        story.extend([img, Paragraph(caption, styles["Caption"]), Spacer(1, 0.3 * cm)])


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        name="Caption",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=8,
    )
)

OUTPUT.parent.mkdir(exist_ok=True)
doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=1.8 * cm,
    leftMargin=1.8 * cm,
    topMargin=1.6 * cm,
    bottomMargin=1.6 * cm,
)

story = [
    Paragraph("Rice Variety Classification Using Logistic Regression", styles["TitleCenter"]),
    Paragraph("Muhammad Jabran | Machine Learning Portfolio Project", styles["Normal"]),
    Spacer(1, 0.4 * cm),
    Paragraph("Objective", styles["Heading2"]),
    Paragraph(
        "The aim of this project was to classify rice grains as either Cammeo or Osmancik using machine learning techniques. A Logistic Regression model was developed and evaluated to determine how accurately rice varieties can be identified based on physical grain characteristics.",
        styles["BodyText"],
    ),
    Paragraph("Dataset", styles["Heading2"]),
    Paragraph(
        "The analysis used the Rice (Cammeo and Osmancik) dataset from the UCI Machine Learning Repository. It contains 3,810 observations and seven numerical features: Area, Perimeter, Major Axis Length, Minor Axis Length, Eccentricity, Convex Area, and Extent. The dataset contains 2,180 Osmancik samples and 1,630 Cammeo samples.",
        styles["BodyText"],
    ),
]

add_figure(story, "class_distribution.png", "Figure 1. Class distribution of Cammeo and Osmancik rice samples.")

story.extend(
    [
        Paragraph("Exploratory Data Analysis", styles["Heading2"]),
        Paragraph(
            "The exploratory analysis examined descriptive statistics and visualizations. The Area feature showed noticeable differences between the two varieties. The correlation analysis showed strong positive relationships among several size-related measurements, particularly Area, Convex Area, Perimeter, and Major Axis Length.",
            styles["BodyText"],
        ),
    ]
)
add_figure(story, "area_boxplot.png", "Figure 2. Distribution of grain Area by rice variety.")
add_figure(story, "correlation_heatmap.png", "Figure 3. Correlation matrix of the numerical grain measurements.")

story.extend(
    [
        Paragraph("Methodology", styles["Heading2"]),
        Paragraph(
            "The target variable was encoded with Cammeo as 0 and Osmancik as 1. The dataset was divided into 70% training data and 30% test data. Following the original assignment, BT-ID 732084 was used as the random state. A Logistic Regression model with max_iter=1000 was evaluated using test accuracy, a confusion matrix, ROC-AUC, an ROC curve, and 5-fold cross-validation.",
            styles["BodyText"],
        ),
        Paragraph("Results", styles["Heading2"]),
    ]
)

result_data = [
    ["Metric", "Result"],
    ["Test accuracy", "93.53%"],
    ["ROC-AUC", "0.9816"],
    ["Mean 5-fold CV accuracy", "93.02%"],
]
result_table = Table(result_data, colWidths=[8 * cm, 5 * cm])
result_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAAAAA")),
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]
    )
)
story.extend([result_table, Spacer(1, 0.4 * cm)])
add_figure(story, "confusion_matrix.png", "Figure 4. Confusion matrix on the test set.")
add_figure(story, "roc_curve.png", "Figure 5. Receiver Operating Characteristic curve for the Logistic Regression model.")

story.extend(
    [
        Paragraph("Discussion", styles["Heading2"]),
        Paragraph(
            "The high test accuracy, strong ROC-AUC, and stable cross-validation performance show that the selected grain measurements contain sufficient information to distinguish the two rice varieties effectively. A limitation is that the dataset contains only two varieties, while real-world classification systems may need to handle additional varieties and greater measurement variability.",
            styles["BodyText"],
        ),
        Paragraph("Business Applications", styles["Heading2"]),
        Paragraph(
            "This type of model could support automated rice sorting, food quality control, agricultural research, reduced manual inspection effort, and more consistent classification in processing facilities.",
            styles["BodyText"],
        ),
        Paragraph("Conclusion", styles["Heading2"]),
        Paragraph(
            "The project demonstrates how a simple, interpretable machine-learning method can achieve strong performance in a food-quality classification task and illustrates the practical value of data analytics in agricultural and food-system applications.",
            styles["BodyText"],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Source: UCI Machine Learning Repository — Rice (Cammeo and Osmancik) dataset.",
            styles["Caption"],
        ),
    ]
)

doc.build(story)
print(f"Created {OUTPUT}")
