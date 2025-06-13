#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GSI Hackathon - SHAP Explainability Generator for LightGBM Model
----------------------------------------------------------------
This script generates SHAP value explanations for the top 500 prospective Grid_IDs
identified in the mineral targeting solution, using parallel processing for efficiency.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import warnings
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from PIL import Image as PILImage
import io

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data_and_model():
    """Load the model, dataset, feature mapping, and labels."""
    print("Loading data and model...")
    
    # Load the trained LightGBM model
    model = lgb.Booster(model_file="4_model_training/final_lgbm_model_v6.txt")
    
    # Get the expected feature count from model
    num_features = model.num_feature()
    print(f"Model expects {num_features} features")
    
    # Load the full dataset with all features
    full_dataset = pd.read_csv("2_data_processed/FINAL_features_v5.csv")
    
    # Load the labels with top 500 Grid_IDs
    labels_df = pd.read_csv("2_data_processed/labels_top500_v6.csv")
    
    # Load the feature columns mapping if available
    try:
        with open("4_model_training/feature_mapping_v6.json", "r") as f:
            feature_columns = json.load(f)
            # If it's a list, use it directly; if it's a dict, extract the values
            if isinstance(feature_columns, dict):
                feature_columns = list(feature_columns.values())
            elif not isinstance(feature_columns, list):
                print("Warning: Unexpected format in JSON file, using numeric columns instead")
                feature_columns = None
    except Exception as e:
        print(f"Warning: Error loading JSON file: {str(e)}")
        print("Will use numeric columns from dataset")
        feature_columns = None
    
    return model, full_dataset, feature_columns, labels_df, num_features


def filter_and_prepare_data(full_dataset, feature_columns, labels_df, num_features):
    """Filter dataset to include only top 500 Grid_IDs and required features."""
    print("Preparing data for SHAP analysis...")
    
    # Create a lookup dictionary for image_filename to other data
    lookup_dict = {}
    
    # Print available columns for debugging
    print(f"Available columns in labels_df: {labels_df.columns.tolist()}")
    
    for _, row in labels_df.iterrows():
        lookup_dict[row['image_filename']] = {
            'predicted_prob': row['Predicted_Prob'] if 'Predicted_Prob' in labels_df.columns else None,
            'label': row['Label'] if 'Label' in labels_df.columns else None,
            'dem_elevation': row['DEM_Elevation_Mean'] if 'DEM_Elevation_Mean' in labels_df.columns else None,
            'prospectivity_level': row['Prospectivity_Level'] if 'Prospectivity_Level' in labels_df.columns else None,
        }
    
    # Filter the dataset to include only relevant Grid_IDs
    filtered_dataset = full_dataset[full_dataset['Grid_ID'].isin(labels_df['image_filename'])]
    
    # Store Grid_ID column separately before removing it from features
    grid_id_column = filtered_dataset['Grid_ID'].copy()
    
    # Get all numeric columns
    numeric_columns = filtered_dataset.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-feature columns if they're in the numeric columns
    for col in ['Grid_ID', 'Prospectivity_Level', 'Label']:
        if col in numeric_columns:
            numeric_columns.remove(col)
            
    # Use specified feature columns if provided, otherwise use all numeric columns
    if feature_columns is not None:
        try:
            # Exclude non-numeric columns
            numeric_features = [col for col in feature_columns 
                               if col in numeric_columns]
            X = filtered_dataset[numeric_features]
        except KeyError as e:
            print(f"Error: Some features in JSON not found in dataset: {str(e)}")
            print("Falling back to all numeric columns")
            X = filtered_dataset[numeric_columns]
    else:
        # Use all numeric columns
        X = filtered_dataset[numeric_columns]
    
    # Match feature count with what the model expects
    print(f"Current feature count: {X.shape[1]}, Model expects: {num_features}")
    
    if X.shape[1] > num_features:
        print(f"Too many features ({X.shape[1]}), trimming to first {num_features} features")
        X = X.iloc[:, :num_features]
    elif X.shape[1] < num_features:
        print(f"Warning: Too few features ({X.shape[1]}), model expects {num_features}")
        # Add dummy features to match the expected count
        for i in range(X.shape[1], num_features):
            X[f'dummy_feature_{i}'] = 0.0
            
    # Verify the feature count now matches
    print(f"Final feature matrix shape: {X.shape}")
    
    # Map Grid_IDs to indices for reference
    grid_indices = dict(zip(grid_id_column, range(len(grid_id_column))))
    
    return filtered_dataset, X, grid_indices, lookup_dict


def generate_shap_explanation(grid_id, X, explainer, model, grid_indices, lookup_dict, output_dir):
    """Generate SHAP values and explanation for a single Grid_ID."""
    try:
        # Create output directory for this Grid_ID
        grid_output_dir = os.path.join(output_dir, grid_id)
        ensure_dir_exists(grid_output_dir)
        
        # Get the row index for this Grid_ID
        idx = grid_indices[grid_id]
        
        # Get the feature values for this Grid_ID
        grid_features = X.iloc[[idx]]
        
        # Calculate SHAP values for this instance
        try:
            # First try the classic method without pred_params
            shap_values = explainer.shap_values(grid_features)
            
            # If shap_values is a list (for multi-class), take the positive class (class 1)
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_values = shap_values[1]  # Class 1 (positive class) SHAP values
                else:
                    shap_values = shap_values[0]
        except Exception as e:
            print(f"Classic SHAP calculation failed: {str(e)}")
            print("Trying alternative SHAP calculation method...")
            
            # Try alternative calculation method
            shap_object = explainer(grid_features)
            shap_values = shap_object.values
        
        # Get the feature importance order
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)
        
        # Get top 10 features
        top10_features = feature_importance.head(10)
        
        # Generate explanation text
        extra_info = lookup_dict.get(grid_id, {})
        pred_prob = extra_info.get('predicted_prob', 'N/A')
        label = extra_info.get('label', 'N/A')
        dem_elevation = extra_info.get('dem_elevation', 'N/A')
        prospectivity = extra_info.get('prospectivity_level', 'N/A')
        
        explanation_text = f"SHAP Explanation for Grid_ID: {grid_id}\n"
        explanation_text += f"Predicted Probability: {pred_prob}\n"
        explanation_text += f"Label: {label}\n"
        explanation_text += f"DEM Elevation: {dem_elevation}\n"
        explanation_text += f"Prospectivity Level: {prospectivity}\n\n"
        explanation_text += "Top 10 Contributing Features:\n"
        
        for i, (feature, importance) in enumerate(zip(top10_features['feature'], top10_features['importance'])):
            value = grid_features[feature].values[0]
            direction = "INCREASES" if shap_values[0][feature_importance.index[i]] > 0 else "DECREASES"
            explanation_text += f"{i+1}. {feature} = {value:.6f} ({direction} prediction by {importance:.6f})\n"
        
        # Save explanation text to file
        with open(os.path.join(grid_output_dir, "shap_explanation.txt"), "w") as f:
            f.write(explanation_text)
        
        # ---- PROFESSIONAL SHAP PLOT GENERATION ----
        # Extract SHAP values for the top features
        feature_idx = top10_features.index.tolist()
        feature_names = top10_features['feature'].tolist()
        feature_shap_values = [shap_values[0][i] for i in feature_idx]
        
        # Set up figure with proper dimensions to avoid wasted space
        plt.figure(figsize=(14, 8))
        
        # Get the maximum width needed for feature names to set proper left margin
        max_name_len = max([len(name) for name in feature_names])
        left_margin = min(0.28 + (max_name_len * 0.005), 0.4)  # Dynamic margin based on name length
        
        # Define professional color palette
        pos_color = '#D12929'  # Professional red for positive values
        neg_color = '#2C7BB6'  # Professional blue for negative values
        colors = [pos_color if val > 0 else neg_color for val in feature_shap_values]
        
        # Set up the plot area with appropriate margins
        plt.subplots_adjust(left=left_margin, right=0.95, top=0.95, bottom=0.1)
        
        # Create horizontal bar plot
        ax = plt.gca()
        bars = ax.barh(np.arange(len(feature_names)), feature_shap_values, color=colors, height=0.7, alpha=0.9)
        
        # Set clear plot title
        plt.title(f"SHAP Values for Grid_ID: {grid_id}", fontsize=14, pad=10)
        
        # Set clean, professional axis labels
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=11, labelpad=10)
        plt.ylabel('Feature', fontsize=11, labelpad=10)
        
        # Format y-axis ticks (feature names)
        plt.yticks(np.arange(len(feature_names)), feature_names, fontsize=9)
        
        # Add subtle grid lines for readability without cluttering
        plt.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
        
        # Add zero reference line for clarity
        plt.axvline(x=0, color='#555555', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Format x-axis with proper scaling
        max_abs_value = max(abs(min(feature_shap_values)), abs(max(feature_shap_values)))
        plt.xlim(min(-0.1, -max_abs_value*1.1), max(0.1, max_abs_value*1.1))
        
        # Add value labels on the bars
        for i, v in enumerate(feature_shap_values):
            # Position depends on the value sign
            if v >= 0:
                ha = 'left'
                x_pos = v + (max_abs_value * 0.02)  # Small offset from the end of the bar
            else:
                ha = 'right'
                x_pos = v - (max_abs_value * 0.02)  # Small offset from the end of the bar
                
            plt.text(x_pos, i, f"{v:.4f}", va='center', ha=ha, fontsize=9, 
                     color='#333333', fontweight='normal')
        
        # Display feature values next to feature names
        for i, feature in enumerate(top10_features['feature']):
            value = grid_features[feature].values[0]
            # Add feature value to the y-tick label
            current_label = plt.gca().get_yticklabels()[i].get_text()
            plt.gca().get_yticklabels()[i].set_text(f"{current_label} = {value:.4f}")
        
        # Add subtle border around plot for professional look
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            
        # Save the plot with high resolution
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
        error_msg = f"Error processing Grid_ID {grid_id}: {str(e)}"
        print(error_msg)
        return {
            'grid_id': grid_id,
            'explanation_text': error_msg,
            'plot_path': None,
            'success': False
        }

def create_summary_pdf(results, output_pdf_path):
    """Create a professional PDF summary of all SHAP explanations."""
    print(f"Creating professional summary PDF: {output_pdf_path}...")
    
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Register Times New Roman font if available
    try:
        pdfmetrics.registerFont(TTFont('Times-Roman', 'Times-Roman.ttf'))
        pdfmetrics.registerFont(TTFont('Times-Bold', 'Times-Bold.ttf'))
        font_family = 'Times-Roman'
        bold_font = 'Times-Bold'
    except:
        print("Times New Roman font not available, using default fonts")
        font_family = 'Times-Roman'  # ReportLab's built-in Times-Roman
        bold_font = 'Times-Bold'
    
    doc = SimpleDocTemplate(
        output_pdf_path, 
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Create custom styles with Times New Roman font
    
    # Main Title style
    styles.add(ParagraphStyle(
        name='MainTitle',
        fontName=bold_font,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=24,
        textColor=colors.darkblue
    ))
    
    # Grid Header style
    styles.add(ParagraphStyle(
        name='GridHeader',
        fontName=bold_font,
        fontSize=16,
        leading=19,
        spaceAfter=14,
        textColor=colors.darkblue
    ))
    
    # Normal paragraph style with Times New Roman
    styles.add(ParagraphStyle(
        name='NormalText',
        fontName=font_family,
        fontSize=11,
        leading=13,  # Increased line spacing to 1.15
        spaceAfter=6
    ))
    
    # Bold text style for key information
    styles.add(ParagraphStyle(
        name='BoldText',
        fontName=bold_font,
        fontSize=11,
        leading=13,  # Increased line spacing to 1.15
        spaceAfter=6
    ))
    
    # Style for section headings
    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontName=bold_font,
        fontSize=12,
        leading=14,
        spaceAfter=8,
        textColor=colors.darkslategray
    ))
    
    story = []
    
    # Add main document title
    main_title = Paragraph("SHAP Explainability Summary Report", styles['MainTitle'])
    story.append(main_title)
    story.append(Spacer(1, 0.3 * inch))
    
    # Add introduction text
    intro_text = """This report provides SHAP (SHapley Additive exPlanations) value analysis for the top prospective Grid_IDs 
    identified in the mineral targeting solution. Each entry includes the key contributing features and their 
    impact on the prediction, visualized through SHAP plots."""
    
    intro = Paragraph(intro_text, styles['NormalText'])
    story.append(intro)
    story.append(Spacer(1, 0.4 * inch))
    
    # Sort results by grid_id to ensure consistent ordering
    sorted_results = sorted(results, key=lambda x: x['grid_id'])
    
    # Counter for page breaks
    count = 0
    
    for result in sorted_results:
        if not result['success']:
            continue
        
        # Add page break after first page and then every entry
        if count > 0:
            story.append(PageBreak())
        count += 1
            
        # Add Grid_ID header
        grid_header = Paragraph(f"Grid_ID: {result['grid_id']}", styles['GridHeader'])
        story.append(grid_header)
        
        # Parse explanation text for better formatting
        explanation_lines = result['explanation_text'].split('\n')
        
        # Process the explanation text by sections
        section = None
        for line in explanation_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a header line
            if "SHAP Explanation for Grid_ID:" in line:
                continue  # Skip this line as we already have a header
            elif "Top 10 Contributing Features:" in line:
                section = "features"
                section_header = Paragraph(line, styles['SectionHeading'])
                story.append(section_header)
                continue
            
            # Handle regular data lines
            if section == "features":
                # This is a feature line, format it nicely
                p = Paragraph(line, styles['NormalText'])
                story.append(p)
            else:
                # This is metadata, format as key-value pairs
                if ":" in line:
                    key, value = line.split(":", 1)
                    p = Paragraph(f"<b>{key}:</b>{value}", styles['NormalText'])
                    story.append(p)
                else:
                    p = Paragraph(line, styles['NormalText'])
                    story.append(p)
        
        story.append(Spacer(1, 0.25 * inch))
        
        # Add SHAP plot with caption
        if result['plot_path'] and os.path.exists(result['plot_path']):
            img = PILImage.open(result['plot_path'])
            img_width, img_height = img.size
            
            # Calculate aspect ratio
            aspect_ratio = img_height / img_width
            
            # Set width to 6.5 inches (letter width - margins)
            width = 6.5 * inch
            height = width * aspect_ratio
            
            # Add the image to the PDF
            img_obj = Image(result['plot_path'], width=width, height=height)
            story.append(img_obj)
            
            # Add caption for the figure
            caption = Paragraph(f"<i>Figure 1: SHAP values showing feature contributions for Grid_ID {result['grid_id']}</i>", 
                               styles['NormalText'])
            story.append(caption)
    
    # Build the PDF
    doc.build(story)
    print(f"Professional PDF created successfully: {output_pdf_path}")

def main():
    """Main function to orchestrate the SHAP explanation generation process."""
    # Set up output directory
    output_dir = os.path.join("5_outputs_maps/SHAP_XAi", "SHAP_plots")
    ensure_dir_exists(output_dir)
    
    # Load data and model
    model, full_dataset, feature_columns, labels_df, num_features = load_data_and_model()
    
    # Filter and prepare data
    filtered_dataset, X, grid_indices, lookup_dict = filter_and_prepare_data(
        full_dataset, feature_columns, labels_df, num_features
    )
    
    # Initialize SHAP explainer
    print("Initializing SHAP explainer...")
    try:
        # Initialize without the problematic parameters
        explainer = shap.TreeExplainer(
            model, 
            model_output="raw",
            feature_perturbation="tree_path_dependent"
        )
    except Exception as e:
        print(f"Error initializing TreeExplainer: {str(e)}")
        print("Trying alternative initialization...")
        # Try alternative initialization for LightGBM models
        explainer = shap.Explainer(model)
    
    # Get unique Grid_IDs for processing
    grid_ids = list(grid_indices.keys())
    print(f"Generating SHAP explanations for {len(grid_ids)} Grid_IDs using parallel processing...")
    
    # Process Grid_IDs in parallel with tqdm progress bar
    # Reduce n_jobs to prevent memory issues
    n_jobs = min(10, os.cpu_count() or 1)
    print(f"Using {n_jobs} parallel jobs")
    
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(generate_shap_explanation)(
            grid_id, X, explainer, model, grid_indices, lookup_dict, output_dir
        ) for grid_id in tqdm(grid_ids, desc="Processing Grid_IDs")
    )
    
    # Create summary PDF
    output_pdf_path = os.path.join("5_outputs_maps/SHAP_XAi", "SHAP_Explainability_Summary.pdf")
    create_summary_pdf(results, output_pdf_path)
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    print(f"SHAP explanation generation complete!")
    print(f"Successfully processed {success_count} out of {len(grid_ids)} Grid_IDs")
    print(f"Summary PDF saved to: {output_pdf_path}")
    print(f"Individual SHAP outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
