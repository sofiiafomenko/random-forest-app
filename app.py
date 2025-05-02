from metrics import (
    accuracy_score_manual as accuracy_score,
    precision_score_manual as precision_score,
    recall_score_manual as recall_score,
    confusion_matrix_manual as confusion_matrix,
    mean_absolute_error_manual as mean_absolute_error,
    mean_squared_error_manual as mean_squared_error,
    r2_score_manual as r2_score
)

from random_forest_model import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, jsonify
import pandas as pd
import io
import base64
import numpy as np

# Prepnúť Matplotlib na neinteraktívny backend
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Hlavná stránka aplikácie


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Stránka s informáciami o aplikácii


@app.route('/info', methods=['GET'])
def info():
    return render_template('info.html')

# Hlavná funkcia na predikciu


@app.route('/predict', methods=['POST'])
def predict():
    # Čítanie ďalších parametrov formulára: 'classification' alebo 'regression'
    selected_task = request.form.get('task')
    analyze_metrics = request.form.get('analyze_metrics') == "on"
    visualize = request.form.get('visualize') == "on"
    show_summary = request.form.get('show_summary') == "on"

    # Pokus o načítanie počtu riadkov na zobrazenie
    try:
        display_rows = int(request.form.get('display_rows', 10))
    except:
        display_rows = 10
    try:
        head_rows = int(request.form.get('head_rows', 5))
    except:
        head_rows = 5

    # Čítanie hyperparametrov
    try:
        n_trees = int(request.form.get('n_trees', 10))
    except:
        n_trees = 10
    max_depth_input = request.form.get('max_depth', '')
    max_depth = int(max_depth_input) if max_depth_input.strip() != '' else None
    try:
        min_samples_split = int(request.form.get('min_samples_split', 2))
    except:
        min_samples_split = 2
    loss = request.form.get('loss', 'gini')
    balance_class_weights = request.form.get('balance_class_weights') == "on"

    # Kontrola, či bol nahraný súbor
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'File not provided'})

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': 'Error reading CSV: ' + str(e)})

    # Kontrola minimálneho počtu stĺpcov v CSV
    if df.shape[1] < 4:
        return jsonify({'error': 'CSV must contain at least 4 columns'})

    # --------------------- #
    # SPRACOVANIE KATEGÓRIÍ #
    # --------------------- #

    # Uloženie pôvodného DataFrame pre ďalšie použitie
    original_df = df.copy()

    # Určenie cieľového stĺpca (posledný stĺpec)
    target_col = df.columns[-1]

    # Konverzia kategórií (okrem cieľového stĺpca) na číselné kódy
    feature_columns = df.columns[:-1]

    for col in feature_columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            df[col] = df[col].astype('category').cat.codes

    # Ak ide o klasifikáciu, spracovať aj cieľový stĺpec
    if selected_task == 'classification':
        if df[target_col].dtype == 'object' or str(df[target_col].dtype) == 'category':
            df[target_col] = df[target_col].astype('category')
            target_categories = df[target_col].cat.categories
            df[target_col] = df[target_col].cat.codes
        else:
            target_categories = None
    else:
        target_categories = None

    from sklearn.model_selection import train_test_split

    # Rozdelenie na X (vstupy) a y (cieľové hodnoty)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Rozdelenie na trénovaciu a testovaciu množinu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Výpočet informácií o datasete
    dataset_rows = df.shape[0]
    dataset_columns = df.shape[1]
    dataset_info = f"Dataset contains {dataset_rows} rows and {dataset_columns} attributes."

    # Vytvorenie modelu
    if selected_task == 'regression':
        model = RandomForestRegressor(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            loss=loss
        )
        task = 'regression'
    else:
        model = RandomForestClassifier(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            loss=loss,
            balance_class_weights=balance_class_weights
        )
        task = 'classification'

        # Trénovanie modelu
    model.fit(X_train, y_train)

    # Predikcia výsledkov na testovacej množine
    preds = model.predict(X_test)

    # Ak ide o klasifikáciu a cieľ bol kategóriou, dekódovať predikcie
    if task == 'classification' and target_categories is not None:
        decoded_preds = [target_categories[int(p)] for p in preds]
        test_df = pd.DataFrame(X_test, columns=feature_columns)
        test_df['True'] = y_test
        test_df['Prediction'] = decoded_preds
    else:
       # Vytvorenie DataFrame pre testovaciu množinu
        test_df = pd.DataFrame(X_test, columns=feature_columns)
        test_df['True'] = y_test
        test_df['Prediction'] = preds

    # Výpočet metrík na testovacej množine
    metrics = {}
    metric_explanations = {}
    vis_imgs = []          # zoznam grafov ako base64 obrázky
    vis_explanations = []  # zoznam vysvetlení pre grafy

    if analyze_metrics:
        if task == 'classification':
            metrics['Accuracy'] = accuracy_score(y_test, preds)
            metrics['Precision'] = precision_score(
                y_test, preds, average='weighted', zero_division=0
            )
            metrics['Recall'] = recall_score(
                y_test, preds, average='weighted', zero_division=0
            )
            metrics['F1'] = 2 * (metrics['Precision'] * metrics['Recall']) / \
                (metrics['Precision'] + metrics['Recall'])
            metric_explanations = {
                'Accuracy': 'The proportion of correct predictions among all predictions.',
                'Precision': 'The ratio of true positives to the total predicted positives.',
                'Recall': 'The ratio of true positives to the actual positives.',
                'F1': 'The harmonic mean of precision and recall.'
            }

        else:
            metrics['MAE'] = mean_absolute_error(y_test, preds)
            metrics['MSE'] = mean_squared_error(y_test, preds)
            metrics['R2'] = r2_score(y_test, preds)
            metric_explanations = {
                'MAE': 'Mean Absolute Error (MAE): The average absolute error.',
                'MSE': 'Mean Squared Error (MSE): The average squared error.',
                'R2': 'R² (Coefficient of Determination): How well the model explains the data.'
            }

    # Jednoduché hodnotenie výsledkov
    result_assessment = ""
    if analyze_metrics:
        if task == 'classification':
            acc = metrics.get('Accuracy', 0)
            if acc >= 0.8:
                result_assessment = "Classification quality is excellent."
            elif acc >= 0.6:
                result_assessment = "Classification quality is average."
            else:
                result_assessment = "Classification quality is low and needs improvement."
        else:  # regression
            r2_val = metrics.get('R2', 0)
            if r2_val >= 0.7:
                result_assessment = "The regression model shows good results."
            elif r2_val >= 0.5:
                result_assessment = "The regression model shows average results."
            else:
                result_assessment = "The regression model needs improvement."

    # Generovanie vizualizácií
    if visualize:
        if task == 'classification':
            # Matica zámien
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Confusion matrix shows how many objects of each class were predicted correctly/incorrectly."
            )
            plt.close(fig)

            # Histogram skutočných hodnôt
            true_series = pd.Series(y_test)
            fig, ax = plt.subplots()
            counts_true = true_series.value_counts()
            counts_true.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Distribution of True Labels (numeric)')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Bar chart showing the distribution of true classes (in numeric form)."
            )
            plt.close(fig)

            # Histogram predikovaných hodnôt
            pred_series = pd.Series(preds)
            fig, ax = plt.subplots()
            counts_pred = pred_series.value_counts()
            counts_pred.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('Distribution of Predicted Labels (numeric)')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Bar chart showing the distribution of predicted classes (in numeric form)."
            )
            plt.close(fig)

            # Porovnanie rozdelení
            fig, ax = plt.subplots()
            counts_true.sort_index().plot(kind='bar', ax=ax, position=0,
                                          width=0.4, label='True', color='skyblue')
            counts_pred.sort_index().plot(kind='bar', ax=ax, position=1, width=0.4,
                                          label='Predicted', color='lightgreen')
            ax.set_title('Comparison of distributions (numeric class codes)')
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Comparison chart showing how well the predicted distribution matches the true distribution."
            )
            plt.close(fig)

            # Koláčový graf skutočných hodnôt
            fig, ax = plt.subplots()
            counts_true.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_ylabel('')
            ax.set_title('Distribution of True Labels (Pie Chart)')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Pie chart showing the percentage distribution of true labels."
            )
            plt.close(fig)

        else:  # regression
            # Graf rozptylu (Actual vs Predicted)
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [
                    y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Scatter plot shows the relationship between predictions and actual values."
            )
            plt.close(fig)

            # Graf reziduí
            residuals = y_test - preds
            fig, ax = plt.subplots()
            ax.scatter(preds, residuals, alpha=0.6, color='orange')
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Predicted')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Residual plot shows the errors (difference between actual and predicted values)."
            )
            plt.close(fig)

            # Histogram reziduí
            fig, ax = plt.subplots()
            ax.hist(residuals, bins=20, color='purple', alpha=0.7)
            ax.set_title('Histogram of Residuals')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Histogram of residuals helps to assess the error distribution."
            )
            plt.close(fig)

            # Boxplot porovnania skutočných a predikovaných hodnôt
            fig, ax = plt.subplots()
            ax.boxplot([y_test, preds], labels=['Actual', 'Predicted'])
            ax.set_title('Comparison (Boxplot)')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Boxplot compares the distribution of actual and predicted values."
            )
            plt.close(fig)

            # Histogram skutočných hodnôt
            fig, ax = plt.subplots()
            ax.hist(y_test, bins=20, color='green', alpha=0.7)
            ax.set_title('Distribution of Actual Values (Target)')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            vis_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            vis_explanations.append(
                "Histogram shows the original distribution of the target variable."
            )
            plt.close(fig)

        # Dodatočná heatmapa, ak je počet atribútov <= 60
        if original_df.shape[1] <= 60:
            numeric_df = original_df.select_dtypes(include=[np.number])
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(25, 20))
                hm = sns.heatmap(
                    numeric_df.corr(),
                    annot=True,
                    # šírka anotácií vo vnútri buniek
                    annot_kws={"fontsize": 25},
                    cmap='coolwarm',
                    ax=ax
                )
                # Zvýšenie veľkosti písma na farebnej lište (colorbar)
                cbar = hm.collections[0].colorbar
                cbar.ax.tick_params(labelsize=25)
                ax.set_title(
                    'Heatmap of correlations between numeric features', fontsize=24)
                ax.tick_params(axis='both', which='major', labelsize=20)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                vis_imgs.append(base64.b64encode(
                    buf.getvalue()).decode('utf-8'))
                vis_explanations.append(
                    "Heatmap for analyzing relationships between numeric variables.")
                plt.close(fig)

    predictions_table = original_df.head(
        display_rows).to_html(classes='data', index=False)
    predictions_explanation = f"First {display_rows} rows with added 'Prediction' column.<br>{dataset_info}"
    head_table = original_df.head(head_rows).to_html(
        classes='data', index=False)
    head_explanation = f"First {head_rows} rows of the original dataset (without transformation)."
    summary_table = None
    summary_explanation = ""
    if show_summary:
        summary_table = original_df.describe(
            include='all').to_html(classes='data', index=True)
        summary_explanation = "Basic statistics of the original DataFrame."

    return render_template(
        'results.html',
        predictions_table=predictions_table,
        predictions_explanation=predictions_explanation,
        head_table=head_table,
        head_explanation=head_explanation,
        summary_table=summary_table,
        summary_explanation=summary_explanation,
        task=task,
        metrics=metrics,
        metric_explanations=metric_explanations,
        visualizations=vis_imgs,
        vis_explanations=vis_explanations,
        display_rows=display_rows,
        head_rows=head_rows,
        result_assessment=result_assessment,
        dataset_name=file.filename,
        dataset_info=dataset_info
    )


# Spustenie aplikácie
if __name__ == '__main__':
    app.run(debug=True)
