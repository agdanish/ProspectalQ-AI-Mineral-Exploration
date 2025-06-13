import os
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from joblib import Parallel, delayed
from tqdm import tqdm
import json

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Load feature mapping and inverse mapping
with open("4_model_training/feature_mapping_v6.json") as f:
    feature_mapping = json.load(f)
inv_feature_mapping = {v: k for k, v in feature_mapping.items()}

def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)

def load_data_and_model():
    model = lgb.Booster(model_file=model_path)
    full_dataset = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    renamed_dataset = full_dataset.rename(columns={k: v for k, v in feature_mapping.items() if k in full_dataset.columns})
    mapped_feature_columns = list(feature_mapping.values())
    X = renamed_dataset[mapped_feature_columns].replace(-9999, np.nan)
    return model, renamed_dataset, X, labels_df

def generate_shap_explanation(grid_id, X, explainer, model, grid_indices, lookup_dict, output_dir):
    try:
        grid_output_dir = os.path.join(output_dir, grid_id)
        ensure_dir_exists(grid_output_dir)
        idx = grid_indices[grid_id]
        grid_features = X.iloc[[idx]]

        try:
            shap_values = explainer.shap_values(grid_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        except:
            shap_object = explainer(grid_features)
            shap_values = shap_object.values

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)

        top10_features = feature_importance.head(10)

        extra_info = lookup_dict.get(grid_id, {})
        explanation_text = f"SHAP Explanation for Grid_ID: {grid_id}\n"
        explanation_text += f"Predicted Probability: {extra_info.get('predicted_prob', 'N/A')}\n"
        explanation_text += f"Label: {extra_info.get('label', 'N/A')}\n"
        explanation_text += f"DEM Elevation: {extra_info.get('dem_elevation', 'N/A')}\n"
        explanation_text += f"Prospectivity Level: {extra_info.get('prospectivity_level', 'N/A')}\n\n"
        explanation_text += "Top 10 Contributing Features:\n"

        for i, (feature, importance) in enumerate(zip(top10_features['feature'], top10_features['importance'])):
            value = grid_features[feature].values[0]
            direction = "INCREASES" if shap_values[0][feature_importance.index[i]] > 0 else "DECREASES"
            real_name = inv_feature_mapping.get(feature, feature)
            explanation_text += f"{i+1}. {real_name} = {value:.6f} ({direction} prediction by {importance:.6f})\n"

        with open(os.path.join(grid_output_dir, "shap_explanation.txt"), "w") as f:
            f.write(explanation_text)

        feature_idx = top10_features.index.tolist()
        feature_names = [inv_feature_mapping.get(X.columns[i], X.columns[i]) for i in feature_idx]
        feature_shap_values = [shap_values[0][i] for i in feature_idx]

        plt.figure(figsize=(14, 8))
        left_margin = min(0.28 + (max(len(name) for name in feature_names) * 0.005), 0.4)
        pos_color = '#D12929'
        neg_color = '#2C7BB6'
        colors = [pos_color if val > 0 else neg_color for val in feature_shap_values]
        plt.subplots_adjust(left=left_margin, right=0.95, top=0.95, bottom=0.1)

        ax = plt.gca()
        ax.barh(np.arange(len(feature_names)), feature_shap_values, color=colors, height=0.7, alpha=0.9)
        plt.title(f"SHAP Values for Grid_ID: {grid_id}", fontsize=14, pad=10)
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
        plt.ylabel('Feature', fontsize=11)
        plt.yticks(np.arange(len(feature_names)), feature_names, fontsize=9)
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='#555555', linestyle='-', linewidth=0.8, alpha=0.5)
        max_abs = max(abs(min(feature_shap_values)), abs(max(feature_shap_values)))
        plt.xlim(min(-0.1, -max_abs*1.1), max(0.1, max_abs*1.1))

        for i, v in enumerate(feature_shap_values):
            ha = 'left' if v >= 0 else 'right'
            offset = max_abs * 0.02
            x_pos = v + offset if v >= 0 else v - offset
            plt.text(x_pos, i, f"{v:.4f}", va='center', ha=ha, fontsize=9, color='#333333')

        for i, feature in enumerate(top10_features['feature']):
            real_name = inv_feature_mapping.get(feature, feature)
            value = grid_features[feature].values[0]
            label_obj = plt.gca().get_yticklabels()[i]
            label_obj.set_text(f"{real_name} = {value:.4f}")

        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')

        plot_path = os.path.join(grid_output_dir, "shap_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'grid_id': grid_id,
            'explanation_text': explanation_text,
            'plot_path': plot_path,
            'success': True
        }

    except Exception as e:
        return {
            'grid_id': grid_id,
            'explanation_text': f"Error processing Grid_ID {grid_id}: {str(e)}",
            'plot_path': None,
            'success': False
        }

def create_summary_pdf(results, output_pdf_path):
    print(f"Creating professional summary PDF: {output_pdf_path}...")
    try:
        pdfmetrics.registerFont(TTFont('Times-Roman', 'Times-Roman.ttf'))
        pdfmetrics.registerFont(TTFont('Times-Bold', 'Times-Bold.ttf'))
        font_family = 'Times-Roman'
        bold_font = 'Times-Bold'
    except:
        print("Times New Roman font not available, using default fonts")
        font_family = 'Times-Roman'
        bold_font = 'Times-Bold'

    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MainTitle', fontName=bold_font, fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=24, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='GridHeader', fontName=bold_font, fontSize=16, leading=19, spaceAfter=14, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='NormalText', fontName=font_family, fontSize=11, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name='BoldText', fontName=bold_font, fontSize=11, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name='SectionHeading', fontName=bold_font, fontSize=12, leading=14, spaceAfter=8, textColor=colors.darkslategray))

    story = [Paragraph("SHAP Explainability Summary Report", styles['MainTitle']), Spacer(1, 0.3 * inch)]
    story.append(Paragraph("""This report provides SHAP (SHapley Additive exPlanations) value analysis for the top prospective Grid_IDs identified in the mineral targeting solution. Each entry includes the key contributing features and their impact on the prediction, visualized through SHAP plots.""", styles['NormalText']))
    story.append(Spacer(1, 0.4 * inch))

    sorted_results = sorted(results, key=lambda x: x['grid_id'])
    count = 0

    for result in sorted_results:
        if not result['success']:
            continue
        if count > 0:
            story.append(PageBreak())
        count += 1

        story.append(Paragraph(f"Grid_ID: {result['grid_id']}", styles['GridHeader']))
        section = None
        for line in result['explanation_text'].split('\n'):
            line = line.strip()
            if not line:
                continue
            if "Top 10 Contributing Features:" in line:
                section = "features"
                story.append(Paragraph(line, styles['SectionHeading']))
                continue
            if section == "features":
                story.append(Paragraph(line, styles['NormalText']))
            else:
                if ':' in line:
                    key, value = line.split(':', 1)
                    story.append(Paragraph(f"<b>{key}:</b>{value}", styles['NormalText']))
                else:
                    story.append(Paragraph(line, styles['NormalText']))

        story.append(Spacer(1, 0.25 * inch))
        if result['plot_path'] and os.path.exists(result['plot_path']):
            img = PILImage.open(result['plot_path'])
            width = 6.5 * inch
            height = width * (img.height / img.width)
            story.append(Image(result['plot_path'], width=width, height=height))
            story.append(Paragraph(f"<i>Figure 1: SHAP values showing feature contributions for Grid_ID {result['grid_id']}</i>", styles['NormalText']))

    doc.build(story)
    print(f"Professional PDF created successfully: {output_pdf_path}")

def main():
    global model_path, features_path, labels_path
    model_path = "4_model_training/final_lgbm_model_v6.txt"
    features_path = "2_data_processed/FINAL_features_v5.csv"
    labels_path = "2_data_processed/labels_top30000_v6.csv"

    output_dir = os.path.join("5_outputs_maps/SHAP", "SHAP_plots")
    ensure_dir_exists(output_dir)

    model, full_dataset, X, labels_df = load_data_and_model()
    filtered_dataset = full_dataset[full_dataset['Grid_ID'].isin(labels_df['Grid_ID'])]
    X = X.loc[filtered_dataset.index]

    grid_indices = {row['Grid_ID']: idx for idx, row in filtered_dataset.reset_index().iterrows()}
    lookup_dict = {
        row['Grid_ID']: {
            'predicted_prob': row.get('Predicted_Prob'),
            'label': row.get('Label'),
            'dem_elevation': row.get('DEM_Elevation_Mean'),
            'prospectivity_level': row.get('Prospectivity_Level')
        } for _, row in labels_df.iterrows()
    }

    print("Initializing SHAP explainer...")
    try:
        explainer = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent")
    except:
        explainer = shap.Explainer(model)

    grid_ids = list(grid_indices.keys())
    print(f"Generating SHAP explanations for {len(grid_ids)} Grid_IDs using parallel processing...")
    n_jobs = min(10, os.cpu_count() or 1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_shap_explanation)(grid_id, X, explainer, model, grid_indices, lookup_dict, output_dir)
        for grid_id in tqdm(grid_ids, desc="Processing Grid_IDs")
    )

    output_pdf_path = os.path.join("5_outputs_maps/SHAP", "SHAP_Explainability_Summary.pdf")
    create_summary_pdf(results, output_pdf_path)

    print("SHAP explanation generation complete!")
    print(f"Successfully processed {sum(1 for r in results if r['success'])} out of {len(grid_ids)} Grid_IDs")
    print(f"Summary PDF saved to: {output_pdf_path}")
    print(f"Individual SHAP outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()