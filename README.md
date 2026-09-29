# Economic Growth — GDP Analysis and Prediction

Python machine-learning project for analysing country-level economic data and predicting **GDP ($ per capita)**, with a Flask web application for prediction.

## Project workflow

- Import and inspect `world.csv`
- Check missing values
- Apply group-based missing-value imputation using `Region`
- Visualize data with Matplotlib and Seaborn
- Encode categorical variables with `LabelEncoder`
- Split data with `train_test_split`
- Train Linear Regression, Decision Tree Regressor, and Random Forest Regressor
- Evaluate models using R² and RMSE
- Save the Random Forest model
- Deploy predictions through Flask

## Dataset

The supplied project material describes `world.csv` as a 227-row, 20-column dataset containing fields such as:

`Country`, `Region`, `Population`, `Area (sq. mi.)`, `Pop. Density (per sq. mi.)`, `Coastline (coast/area ratio)`, `Net migration`, `Infant mortality (per 1000 births)`, `GDP ($ per capita)`, `Literacy (%)`, `Phones (per 1000)`, `Arable (%)`, `Crops (%)`, `Other (%)`, `Climate`, `Birthrate`, `Deathrate`, `Agriculture`, `Industry`, `Service`.

Place the original dataset at:

```text
data/raw/world.csv
```

The original `world.csv` and trained `gdp.pkl` shown in the supplied project screenshots were not attached as actual files, so this repository does not fabricate them.

## Missing-value handling

The project material specifies group-based imputation using `Region`:

- numerical features → group median
- categorical feature such as `Climate` → group mode

A fallback median/mode is used if a group has no usable value.

## Models and evaluation

Models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Metrics:

- R² score
- RMSE (Root Mean Squared Error)

The supplied project material states that Random Forest had lower RMSE than the other models and was selected for deployment. Exact historical scores are not invented here; running the pipeline on the original dataset produces the current metrics.

## Flask application

`Flask_app/app.py`:

- initializes Flask
- loads the trained model
- accepts prediction inputs
- calls the model
- renders the prediction result

Run:

```bash
cd Flask_app
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## Project structure

```text
Economic-Growth/
├── data/
│   └── raw/
│       └── world.csv
├── models/
│   └── gdp.pkl
├── notebooks/
│   └── GDP_Analysis.ipynb
├── reports/
│   └── model_metrics.csv
├── src/
│   └── train_model.py
├── visualizations/
├── Flask_app/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── home.html
│   │   └── gdp_pred.html
│   └── static/css/style.css
├── docs/
│   ├── Project_Workflow.md
│   └── Submission_Checklist.md
├── screenshots/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Train

Place the original `world.csv` in `data/raw/`, then:

```bash
python src/train_model.py
```

Generated files:

```text
models/gdp.pkl
reports/model_metrics.csv
```

## Author

**Venegalla Venkata Laxminath**


## GitHub Pages demo

A static demonstration page is included at `docs/index.html`.

After pushing the repository to GitHub, enable **Settings → Pages → Deploy from branch → main → /docs**.

The resulting demo URL will be:

`https://YOUR-GITHUB-USERNAME.github.io/Economic-Growth/`

This GitHub Pages page is a static presentation demo. The actual ML prediction requires the Flask application to run on a Python-capable server.
