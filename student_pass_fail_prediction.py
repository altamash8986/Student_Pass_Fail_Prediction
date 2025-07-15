import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load dataset
dataset = pd.read_csv("student_pass_fail.csv")

# Features and target
x = dataset.iloc[:, :-1]
y = dataset["Pass"]


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)


# Standard Scaler

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)


# Train Random Forest Classifier  Model
Ran_model = RandomForestClassifier(random_state=42)
Ran_model.fit(x_train_scaler, y_train)

# Train Logistic Regression Model
L_model = LogisticRegression(max_iter=1000, class_weight="balanced")
L_model.fit(x_train_scaler, y_train)

# Train XG Boost Classifier Model
X_model = XGBClassifier(random_state=42)
X_model.fit(x_train_scaler, y_train)

# Accuracy
Ran_test_accuracy = Ran_model.score(x_test_scaler, y_test)
L_test_accuracy = L_model.score(x_test_scaler, y_test)
X_test_accuracy = X_model.score(x_test_scaler, y_test)


# Prediction function
def predict_pass_fail(
    model_choice,
    hours_studied,
    attendance,
    previous_score,
    assignments_submitted,
    extra_classes,
):
    input_data = pd.DataFrame(
        [
            [
                hours_studied,
                attendance,
                previous_score,
                assignments_submitted,
                extra_classes,
            ]
        ],
        columns=[
            "Hours_Studied",
            "Attendance",
            "Previous_Score",
            "Assignments_Submitted",
            "Extra_Classes",
        ],
    )

    input_scaler = scaler.transform(input_data)

    if model_choice == "Random Forest":
        prediction = Ran_model.predict(input_scaler)[0]
        confidence = Ran_model.predict_proba(input_scaler)[0][1]
        accuracy = Ran_test_accuracy

    elif model_choice == "Logistic Regression":
        prediction = L_model.predict(input_scaler)[0]
        confidence = L_model.predict_proba(input_scaler)[0][1]
        accuracy = L_test_accuracy

    elif model_choice == "XG Boost Classifier":
        prediction = X_model.predict(input_scaler)[0]
        confidence = X_model.predict_proba(input_scaler)[0][1]
        accuracy = X_test_accuracy

    result = (
        " 🎊 Congrats, You Passed " if prediction == 1 else " ❌ Better Luck Next Time"
    )

    chances = "Chances of Pass " if prediction == 1 else "Chances of Fail "

    return (
        result,
        f"Model Used: {model_choice}",
        f"Testing Accuracy: {accuracy*100:.2f} %",
        f"Confidence: {confidence*100:.2f} % --> {chances}",
    )


interface = gr.Interface(
    fn=predict_pass_fail,
    inputs=[
        gr.Radio(
            [
                "Random Forest",
                "Logistic Regression",
                "XG Boost Classifier",
            ],
            label="Choose ML Model",
            value="Random Forest",
        ),
        gr.Slider(label="Hours Studied", minimum=1, maximum=20, value=3),
        gr.Slider(label="Attendance (out of 100)", minimum=0, maximum=100, value=45),
        gr.Slider(label="Previous Score", minimum=0, maximum=100, value=33),
        gr.Radio([1, 0], label="Assignments Submitted (1=Yes, 0=No)", value=0),
        gr.Radio([1, 0], label="Attended Extra Classes (1=Yes, 0=No)", value=1),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Model Selected"),
        gr.Textbox(label="Testing Accuracy"),
        gr.Textbox(label="Chances of (Pass/Fail)"),
    ],
    title="Student Pass/Fail Predictor",
    description="Predict if a student will pass or fail based on study hours, attendance, previous scores, and more.",
    article="MADE BY 'MOHD ALTAMASH'",
)

interface.launch(share=True)
