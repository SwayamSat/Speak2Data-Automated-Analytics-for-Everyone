"""
Speak2Data: Automated Analytics for Everyone
Main Streamlit application for natural language data analysis.
"""

import os
import warnings
import logging

# Suppress gRPC/ALTS warnings before importing other modules
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List
import sys

# Import our custom modules
from db_module import DatabaseManager
from nlp_module import NLPProcessor
from sql_generator import SQLGenerator
from ml_pipeline_simple import SimpleMLPipeline as MLPipeline
from utils import DataProcessor, VisualizationGenerator, StreamlitHelpers, ErrorHandler

# Page configuration
st.set_page_config(
    page_title="Speak2Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Minimal CSS Design System
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables - Professional Color Palette */
    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --primary-light: #3b82f6;
        --secondary: #64748b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --background: #ffffff;
        --surface: #f8fafc;
        --border: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --radius: 8px;
    }
    
    /* Global Typography */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        text-align: left;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 1rem;
        font-weight: 400;
        color: var(--text-secondary);
        text-align: left;
        margin-bottom: 2.5rem;
        line-height: 1.6;
    }
    
    /* Section Headers */
    h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
    
    h3 {
        font-size: 1.25rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: var(--surface) !important;
    }
    
    /* Sidebar Headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Sidebar Text - General text elements */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span:not(.stButton span):not(button span),
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown div {
        color: var(--text-primary) !important;
    }
    
    /* Ensure text elements inherit proper color */
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] .block-container {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar Caption Text */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption * {
        color: var(--text-secondary) !important;
    }
    
    /* Sidebar Buttons - Must come after general text styles */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius);
        padding: 0.75rem 1rem;
        font-weight: 400;
        font-size: 0.875rem;
        text-align: left;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] .stButton > button span {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--primary) !important;
        border-color: var(--primary) !important;
        transform: translateX(4px);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover span {
        color: white !important;
    }
    
    /* Sidebar Expanders */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: var(--text-primary) !important;
        font-weight: 500;
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent * {
        color: var(--text-primary) !important;
    }
    
    /* Expander when closed - ensure text is visible */
    [data-testid="stSidebar"] .streamlit-expanderHeader:not([aria-expanded="true"]) {
        color: var(--text-primary) !important;
    }
    
    /* Expander when open - ensure proper contrast */
    [data-testid="stSidebar"] .streamlit-expanderHeader[aria-expanded="true"] {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    /* Expander content text */
    [data-testid="stSidebar"] .streamlit-expanderContent p,
    [data-testid="stSidebar"] .streamlit-expanderContent span,
    [data-testid="stSidebar"] .streamlit-expanderContent div {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    /* Sidebar text elements that might have default white color */
    [data-testid="stSidebar"] .css-10trblm,
    [data-testid="stSidebar"] [class*="css-"] {
        color: var(--text-primary) !important;
    }
    
    /* Force all sidebar text to be dark, but exclude buttons */
    [data-testid="stSidebar"] *:not(.stButton):not(.stButton *):not(button):not(button *) {
        color: var(--text-primary) !important;
    }
    
    /* Override any Streamlit default colors */
    [data-testid="stSidebar"] .element-container *,
    [data-testid="stSidebar"] .block-container *,
    [data-testid="stSidebar"] .stMarkdown * {
        color: var(--text-primary) !important;
    }
    
    /* Specific expander styling overrides */
    [data-testid="stSidebar"] .streamlit-expander {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expander .streamlit-expanderHeader {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expander .streamlit-expanderContent {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    /* Override any white text in expanders, but exclude buttons */
    [data-testid="stSidebar"] .streamlit-expander *:not(.stButton):not(.stButton *):not(button):not(button *) {
        color: var(--text-primary) !important;
    }
    
    /* Specific fix for expander headers that might be white */
    [data-testid="stSidebar"] .streamlit-expanderHeader,
    [data-testid="stSidebar"] .streamlit-expanderHeader div,
    [data-testid="stSidebar"] .streamlit-expanderHeader span {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    /* Remove all hover effects from expander headers */
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover,
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover div,
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover span {
        color: var(--text-primary) !important;
        background-color: transparent !important;
        transform: none !important;
    }
    
    /* Keep expander headers consistent in all states */
    [data-testid="stSidebar"] .streamlit-expanderHeader:focus,
    [data-testid="stSidebar"] .streamlit-expanderHeader:active,
    [data-testid="stSidebar"] .streamlit-expanderHeader[aria-expanded="true"] {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    /* Override any Streamlit default hover/active states - keep consistent */
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover *,
    [data-testid="stSidebar"] .streamlit-expanderHeader:focus *,
    [data-testid="stSidebar"] .streamlit-expanderHeader:active * {
        color: var(--text-primary) !important;
    }
    
    /* Disable all transitions and hover effects */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        transition: none !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Remove all hover effects from database schema expanders specifically */
    [data-testid="stSidebar"] .streamlit-expander:hover {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expander:hover .streamlit-expanderHeader {
        background-color: transparent !important;
        color: var(--text-primary) !important;
    }
    
    /* Override any Streamlit default expander hover states */
    [data-testid="stSidebar"] .streamlit-expander *:hover {
        background-color: transparent !important;
        color: var(--text-primary) !important;
    }
    
    /* Ensure no visual changes on any expander interaction */
    [data-testid="stSidebar"] .streamlit-expanderHeader:focus,
    [data-testid="stSidebar"] .streamlit-expanderHeader:active,
    [data-testid="stSidebar"] .streamlit-expanderHeader:visited {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Ensure expander content is always visible */
    [data-testid="stSidebar"] .streamlit-expanderContent,
    [data-testid="stSidebar"] .streamlit-expanderContent div,
    [data-testid="stSidebar"] .streamlit-expanderContent span,
    [data-testid="stSidebar"] .streamlit-expanderContent p {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: 0.625rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.875rem;
        font-size: 0.9375rem;
        transition: all 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Info/Success/Error/Warning Boxes */
    .stAlert {
        border-radius: var(--radius);
        border-left: 3px solid;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    div[data-baseweb="notification"] {
        border-radius: var(--radius);
    }
    
    /* Code Blocks */
    .stCodeBlock {
        border-radius: var(--radius);
        border: 1px solid var(--border);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Selectbox Styling */
    .stSelectbox label {
        font-weight: 500;
        color: var(--text-primary);
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: var(--radius);
        overflow: hidden;
    }
    
    /* Sidebar Dataframe Styling */
    [data-testid="stSidebar"] .dataframe {
        font-size: 0.8rem;
        border: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .dataframe th {
        background-color: var(--surface);
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stSidebar"] .dataframe td {
        color: var(--text-primary);
        background-color: var(--background);
    }
    
    [data-testid="stSidebar"] .dataframe tr:nth-child(even) td {
        background-color: var(--surface);
    }
    
    /* ML Section Container */
    .ml-section {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Card Styling */
    .info-card {
        background-color: var(--background);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }
    
    /* Footer */
    footer {
        visibility: hidden;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--surface);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary);
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploader"] .uploadedFile {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stFileUploader"] .uploadedFile * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stFileUploader"] .uploadedFile:hover {
        background-color: var(--surface) !important;
        border-color: var(--primary) !important;
    }
    
    /* File uploader button styling */
    [data-testid="stFileUploader"] button {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background-color: var(--primary-dark) !important;
    }
    
    /* Sidebar file uploader specific styling */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * {
        color: var(--text-primary) !important;
    }
    
    /* Override any Streamlit default dark styling for file uploader */
    [data-testid="stSidebar"] [data-baseweb="file-uploader"],
    [data-testid="stSidebar"] [data-baseweb="file-uploader"] * {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }
    
    /* File uploader drag and drop area */
    [data-testid="stFileUploader"] [data-baseweb="base-input"] {
        background-color: var(--background) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stFileUploader"] [data-baseweb="base-input"]::placeholder {
        color: var(--text-secondary) !important;
    }
    
    /* Query Output Styling */
    .stCodeBlock {
        background-color: var(--background) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    
    .stCodeBlock code {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }
    
    /* Dataframe output styling */
    .dataframe {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }
    
    .dataframe th {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    
    .dataframe td {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    
    .dataframe tr:nth-child(even) td {
        background-color: var(--surface) !important;
    }
    
    /* Metric display styling */
    [data-testid="stMetric"] {
        background-color: var(--background) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    /* Alert boxes styling */
    .stAlert {
        background-color: var(--background) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
    }
    
    .stSuccess {
        background-color: #d4edda !important;
        border-color: #c3e6cb !important;
        color: #155724 !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        border-color: #f5c6cb !important;
        color: #721c24 !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        border-color: #ffeaa7 !important;
        color: #856404 !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        border-color: #bee5eb !important;
        color: #0c5460 !important;
    }
    
    /* CRITICAL: Ensure info box text is visible with proper contrast */
    .stInfo,
    .stInfo *,
    .stInfo p,
    .stInfo span,
    .stInfo div,
    .stInfo strong,
    .stInfo em,
    .stInfo markdown,
    .stInfo [class*="markdown"],
    .stInfo [data-testid*="markdown"],
    div[data-baseweb="notification"].stInfo,
    div[data-baseweb="notification"].stInfo *,
    div[data-baseweb="notification"].stInfo p,
    div[data-baseweb="notification"].stInfo span,
    div[data-baseweb="notification"].stInfo div,
    .stInfo .stMarkdown,
    .stInfo .stMarkdown *,
    .stInfo .stMarkdown p,
    .stInfo .stMarkdown span,
    .stInfo .stMarkdown strong,
    [class*="stInfo"] *,
    [class*="stInfo"] p,
    [class*="stInfo"] span,
    [class*="stInfo"] strong {
        color: #0c5460 !important;
    }
    
    /* Database Management Section */
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Force all sidebar elements to have proper colors */
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] .element-container * {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    
    /* Sidebar info boxes */
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stError {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }
    
    /* Query results styling */
    .main .stCodeBlock {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        color: #212529 !important;
    }
    
    .main .stCodeBlock code {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* IMPORTANT: Only set text colors where we KNOW the background is white/light */
    /* NEVER override text in colored alert/info boxes */
    
    /* Basic headings in main content (white background) */
    .main h1:not(.stAlert h1):not(.stInfo h1):not(.stSuccess h1):not(.stWarning h1):not(.stError h1),
    .main h2:not(.stAlert h2):not(.stInfo h2):not(.stSuccess h2):not(.stWarning h2):not(.stError h2),
    .main h3:not(.stAlert h3):not(.stInfo h3):not(.stSuccess h3):not(.stWarning h3):not(.stError h3) {
        color: var(--text-primary) !important;
    }
    
    /* Plain paragraphs in main content (white background) */
    /* IMPORTANT: Exclude all alert/info box paragraphs to preserve their text colors */
    .main p:not(.stAlert p):not(.stInfo p):not(.stSuccess p):not(.stWarning p):not(.stError p):not([class*="stInfo"] p):not([class*="stSuccess"] p):not([class*="stWarning"] p):not([class*="stError"] p) {
        color: var(--text-primary) !important;
    }
    
    /* Explicitly style info box content for visibility - higher specificity */
    div.stInfo,
    div.stInfo *,
    div.stInfo p,
    div.stInfo span,
    div.stInfo div:not([class*="button"]),
    div.stInfo strong,
    div.stInfo em,
    div.stInfo .stMarkdown,
    div.stInfo .stMarkdown *,
    div.stInfo .stMarkdown p,
    div.stInfo .stMarkdown span,
    div.stInfo .stMarkdown strong,
    [class*="stInfo"] p,
    [class*="stInfo"] span,
    [class*="stInfo"] strong,
    [class*="stInfo"] * {
        color: #0c5460 !important;
    }
    
    /* Additional selectors for Streamlit's internal structure - MUST come after all generic rules */
    [data-baseweb="notification"],
    [data-baseweb="notification"] p,
    [data-baseweb="notification"] span,
    [data-baseweb="notification"] div,
    [data-baseweb="notification"] strong,
    [data-baseweb="notification"] em,
    [data-baseweb="notification"] * {
        color: inherit !important;
    }
    
    /* Final override: Force info box text color regardless of nesting level */
    .main .stInfo,
    .main .stInfo *,
    .main .stInfo p,
    .main .stInfo span,
    .main .stInfo div,
    .main .stInfo strong,
    .main .stInfo em,
    .main .stInfo .stMarkdown,
    .main .stInfo .stMarkdown *,
    .main .stInfo .stMarkdown p,
    .main .stInfo .stMarkdown span,
    .main .stInfo .stMarkdown strong,
    .main [class*="stInfo"],
    .main [class*="stInfo"] *,
    .main [class*="stInfo"] p,
    .main [class*="stInfo"] span,
    .main [class*="stInfo"] strong,
    .main div[data-baseweb="notification"],
    .main div[data-baseweb="notification"] *,
    .main div[data-baseweb="notification"] p,
    .main div[data-baseweb="notification"] span,
    .main div[data-baseweb="notification"] strong {
        color: #0c5460 !important;
    }
    
    /* Form labels (always on white background) */
    .main .stTextInput > label,
    .main .stTextArea > label,
    .main .stSelectbox > label,
    .main .stNumberInput > label {
        color: var(--text-primary) !important;
    }
    
    /* Input fields (white background) */
    .main .stTextInput input,
    .main .stTextArea textarea,
    .main .stNumberInput input {
        color: var(--text-primary) !important;
        background-color: #ffffff !important;
    }
    
    /* Selectbox (white background) */
    .main .stSelectbox select {
        color: var(--text-primary) !important;
        background-color: #ffffff !important;
    }
    
    /* Caption text (always light gray) */
    .stCaption,
    [data-testid="stCaption"] {
        color: var(--text-secondary) !important;
    }
    
    /* Code blocks (light gray background) */
    .main .stCodeBlock code {
        color: #212529 !important;
        background-color: #f8f9fa !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: var(--background) !important;
    }
    
    /* CRITICAL: Ensure sidebar expander hover fixes are preserved and not overridden by any generic rules */
    /* These must come after generic rules to override them */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        transition: none !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover *,
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover div,
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover span {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        transform: none !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expander:hover {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expander:hover .streamlit-expanderHeader,
    [data-testid="stSidebar"] .streamlit-expander:hover .streamlit-expanderHeader * {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expander *:hover {
        background-color: transparent !important;
        color: var(--text-primary) !important;
    }
    
    /* Buttons should have white text on primary background (main area) */
    .main .stButton > button {
        color: white !important;
        background-color: var(--primary) !important;
    }
    
    .main .stButton > button span {
        color: white !important;
    }
    
    /* Sidebar buttons should have dark text */
    [data-testid="stSidebar"] .stButton > button {
        color: var(--text-primary) !important;
        background-color: var(--background) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button span {
        color: var(--text-primary) !important;
    }
    
    /* Primary buttons (type="primary") always white text */
    button[kind="primary"],
    .stButton > button[type="button"][kind="primary"] {
        color: white !important;
    }
    
    button[kind="primary"] span,
    .stButton > button[type="button"][kind="primary"] span {
        color: white !important;
    }
    
    /* Additional headings (white background only) */
    .main h4:not(.stAlert h4):not(.stInfo h4):not(.stSuccess h4):not(.stWarning h4):not(.stError h4),
    .main h5:not(.stAlert h5):not(.stInfo h5):not(.stSuccess h5):not(.stWarning h5):not(.stError h5),
    .main h6:not(.stAlert h6):not(.stInfo h6):not(.stSuccess h6):not(.stWarning h6):not(.stError h6) {
        color: var(--text-primary) !important;
    }
    
    /* Subheader text (gray for subtlety) */
    .main .sub-header {
        color: var(--text-secondary) !important;
    }
    
    /* Input placeholders */
    .main input::placeholder,
    .main textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.7 !important;
    }
    
    /* Dataframes/tables (white background) */
    .main .dataframe th,
    .main .dataframe td {
        color: var(--text-primary) !important;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Highlight Box Styling */
    .highlight-box {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(37, 99, 235, 0.03) 100%);
        border: 2px solid var(--primary);
        border-radius: var(--radius);
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.15);
    }
    
    .highlight-box h3 {
        color: var(--primary) !important;
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Upload Section Container */
    #upload-section-container {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(37, 99, 235, 0.03) 100%);
        border: 2px solid var(--primary);
        border-radius: var(--radius);
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.15);
    }
    
    /* Style elements that follow the highlight box to appear inside it */
    #upload-section-container ~ * {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(37, 99, 235, 0.02) 100%) !important;
        border-left: 2px solid var(--primary) !important;
        border-right: 2px solid var(--primary) !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    
    /* Create closing border effect */
    #upload-section-container ~ *:last-of-type::after {
        content: '';
        display: block;
        height: 0;
        border-bottom: 2px solid var(--primary);
        margin-top: 1rem;
        border-radius: 0 0 var(--radius) var(--radius);
    }
    
    /* FINAL OVERRIDE: Force info box text visibility - MUST be last to override all other rules */
    .main .element-container .stInfo,
    .main .element-container .stInfo *,
    .main .element-container .stInfo p,
    .main .element-container .stInfo span,
    .main .element-container .stInfo div,
    .main .element-container .stInfo strong,
    .main .element-container .stInfo em,
    .main .element-container .stInfo .stMarkdown,
    .main .element-container .stInfo .stMarkdown *,
    .main .element-container .stInfo .stMarkdown p,
    .main .element-container .stInfo .stMarkdown span,
    .main .element-container .stInfo .stMarkdown strong,
    .main [class*="stInfo"],
    .main [class*="stInfo"] *,
    .main [class*="stInfo"] p,
    .main [class*="stInfo"] span,
    .main [class*="stInfo"] strong,
    .main div[data-baseweb="notification"].stInfo,
    .main div[data-baseweb="notification"].stInfo *,
    .main div[data-baseweb="notification"].stInfo p,
    .main div[data-baseweb="notification"].stInfo span,
    .main div[data-baseweb="notification"].stInfo strong,
    .main div[data-baseweb="notification"].stInfo div {
        color: #0c5460 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'nlp_processor' not in st.session_state:
    st.session_state.nlp_processor = None
if 'sql_generator' not in st.session_state:
    st.session_state.sql_generator = None
if 'ml_pipeline' not in st.session_state:
    st.session_state.ml_pipeline = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'custom_db_uploaded' not in st.session_state:
    st.session_state.custom_db_uploaded = False
if 'custom_db_name' not in st.session_state:
    st.session_state.custom_db_name = None
if 'query_suggestions' not in st.session_state:
    st.session_state.query_suggestions = []
if 'schema_hash' not in st.session_state:
    st.session_state.schema_hash = None
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None

def initialize_components():
    """Initialize all components of the application."""
    try:
        # Initialize database manager
        if st.session_state.db_manager is None:
            with st.spinner("Initializing database..."):
                st.session_state.db_manager = DatabaseManager()
        
        # Get current database schema
        if st.session_state.db_manager:
            current_schema = st.session_state.db_manager.get_table_schema()
            schema_info = {"tables": current_schema}
            
            # Initialize or update NLP processor with current schema
            if st.session_state.nlp_processor is None:
                with st.spinner("Initializing NLP processor..."):
                    st.session_state.nlp_processor = NLPProcessor(schema_info=schema_info)
            else:
                # Update schema if database changed
                st.session_state.nlp_processor.update_schema(current_schema)
            
            # Initialize SQL generator
            if st.session_state.sql_generator is None:
                st.session_state.sql_generator = SQLGenerator(st.session_state.nlp_processor)
            else:
                # Update schema in SQL generator
                st.session_state.sql_generator.schema_info = st.session_state.nlp_processor.schema_info
        
        # Initialize ML pipeline
        if st.session_state.ml_pipeline is None:
            st.session_state.ml_pipeline = MLPipeline()
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">Speak2Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform natural language into actionable insights with automated SQL generation and machine learning analysis</p>', unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with options and information."""
    with st.sidebar:
        st.markdown("## Navigation")
        st.markdown("---")
        
        # Database Upload Section - Highlight Box
        st.markdown("""
        <div class="highlight-box">
            <h3>🗄️ Database Management</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader for custom database or data files
        uploaded_file = st.file_uploader(
            "Upload Database or Data File",
            type=['db', 'sqlite', 'sqlite3', 'csv', 'xlsx', 'xls', 'parquet'],
            help="Upload SQLite database (.db, .sqlite) or data files (.csv, .xlsx, .parquet) - will be automatically imported",
            key="db_uploader"
        )
        
        # Handle database or data file upload
        if uploaded_file is not None:
            # Check if this is a new file (not already processed)
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.get('last_uploaded_file_id') != current_file_id:
                try:
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    suffix = f'.{file_extension}'
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_path = tmp_file.name
                    
                    # Initialize database manager based on file type
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Load file
                    status_text.text(f"📂 Loading file '{uploaded_file.name}'...")
                    progress_bar.progress(20)
                    
                    # Determine file type and import accordingly
                    if file_extension in ['db', 'sqlite', 'sqlite3']:
                        st.session_state.db_manager = DatabaseManager(custom_db_path=temp_file_path)
                    elif file_extension == 'csv':
                        st.session_state.db_manager = DatabaseManager.create_from_csv(temp_file_path)
                    elif file_extension in ['xlsx', 'xls']:
                        st.session_state.db_manager = DatabaseManager.create_from_excel(temp_file_path)
                    elif file_extension == 'parquet':
                        st.session_state.db_manager = DatabaseManager.create_from_parquet(temp_file_path)
                    else:
                        st.error(f"❌ Unsupported file type: {file_extension}")
                        progress_bar.empty()
                        status_text.empty()
                    
                    if file_extension in ['db', 'sqlite', 'sqlite3', 'csv', 'xlsx', 'xls', 'parquet']:
                        st.session_state.custom_db_uploaded = True
                        st.session_state.custom_db_name = uploaded_file.name
                        st.session_state.last_uploaded_file_id = current_file_id
                        
                        # Step 2: Get schema
                        status_text.text("🔍 Analyzing database schema...")
                        progress_bar.progress(40)
                        
                        # Clear cached suggestions to regenerate for new database
                        st.session_state.query_suggestions = []
                        st.session_state.schema_hash = None
                        
                        # Reset NLP processor and SQL generator to force reinitialization with new schema
                        st.session_state.nlp_processor = None
                        st.session_state.sql_generator = None
                        
                        # Get schema from new database
                        if st.session_state.db_manager:
                            current_schema = st.session_state.db_manager.get_table_schema()
                            if current_schema:
                                schema_info = {"tables": current_schema}
                                
                                # Step 3: Initialize NLP processor
                                status_text.text("🤖 Initializing AI components...")
                                progress_bar.progress(60)
                                
                                # Initialize NLP processor with new schema
                                st.session_state.nlp_processor = NLPProcessor(schema_info=schema_info)
                                
                                # Initialize SQL generator
                                st.session_state.sql_generator = SQLGenerator(st.session_state.nlp_processor)
                                
                                # Step 4: Generate query suggestions
                                status_text.text("💡 Generating AI-powered query suggestions...")
                                progress_bar.progress(80)
                                
                                # Generate new query suggestions immediately
                                try:
                                    if st.session_state.nlp_processor:
                                        st.session_state.query_suggestions = st.session_state.nlp_processor._generate_fallback_suggestions(
                                            current_schema, num_suggestions=6
                                        )
                                    else:
                                        st.session_state.query_suggestions = _generate_basic_suggestions(current_schema)
                                    st.session_state.schema_hash = hash(str(sorted(current_schema.items())))
                                    
                                    progress_bar.progress(100)
                                    status_text.text("✅ Database loaded and ready!")
                                    
                                    st.success(f"✅ Database '{uploaded_file.name}' loaded successfully!")
                                    st.info(f"📊 Found {len(current_schema)} tables: {', '.join(list(current_schema.keys())[:5])}{'...' if len(current_schema) > 5 else ''}")
                                except Exception as e:
                                    # Use basic suggestions if fallback fails
                                    st.session_state.query_suggestions = _generate_basic_suggestions(current_schema)
                                    st.session_state.schema_hash = hash(str(sorted(current_schema.items())))
                            else:
                                st.warning("⚠️ Database loaded but no tables found.")
                        else:
                            st.error("❌ Failed to load database.")
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Only rerun once after successful load
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error loading database: {str(e)}")
                    st.info("Using default database instead.")
        
        # Reset to default database option
        if st.session_state.get('custom_db_uploaded', False):
            if st.button("🔄 Reset to Default Database", use_container_width=True):
                st.session_state.db_manager = DatabaseManager()
                st.session_state.custom_db_uploaded = False
                st.session_state.custom_db_name = None
                
                # Clear cached suggestions to regenerate for default database
                st.session_state.query_suggestions = []
                st.session_state.schema_hash = None
                
                # Reset NLP processor and SQL generator to force reinitialization with default schema
                st.session_state.nlp_processor = None
                st.session_state.sql_generator = None
                
                # Reinitialize components with default database
                initialize_components()
                st.rerun()
        
        # Show current database info
        if st.session_state.get('custom_db_uploaded', False):
            st.info(f"📁 Using: {st.session_state.custom_db_name}")
        else:
            st.info("📁 Using: Default business_data.db")
        
        st.markdown("---")
        
        # Database information
        if st.session_state.db_manager:
            st.markdown("### Database Schema")
            schema = st.session_state.db_manager.get_table_schema()
            for table, columns in schema.items():
                with st.expander(f"{table} ({len(columns)} columns)", expanded=False):
                    # Create a DataFrame for better table display
                    import pandas as pd
                    schema_df = pd.DataFrame({
                        'Column Name': columns,
                        'Data Type': ['TEXT'] * len(columns)  # Default type, could be enhanced
                    })
                    st.dataframe(
                        schema_df,
                        use_container_width=True,
                        hide_index=True,
                        height=min(300, (len(columns) + 1) * 35)  # Dynamic height based on rows
                    )
            st.markdown("---")
        
        # Query history moved to main area
        
        # Query history
        if st.session_state.query_history:
            st.markdown("### Recent Queries")
            for i, query in enumerate(st.session_state.query_history[-5:]):
                if st.button(f"{query[:45]}..." if len(query) > 45 else query, key=f"history_{i}", use_container_width=True):
                    st.session_state.sample_query = query
                    st.rerun()

def generate_basic_explanation(results_df: pd.DataFrame) -> str:
    """Generate a basic explanation when API is unavailable."""
    if results_df.empty:
        return "No data found matching your criteria."
    
    # Get basic stats
    row_count = len(results_df)
    col_count = len(results_df.columns)
    
    # Find numeric columns for analysis
    numeric_cols = [col for col in results_df.columns if pd.api.types.is_numeric_dtype(results_df[col])]
    
    if numeric_cols:
        # Calculate totals and averages for numeric columns
        total_values = {}
        avg_values = {}
        for col in numeric_cols:
            total_values[col] = results_df[col].sum()
            avg_values[col] = results_df[col].mean()
        
        # Find the column with highest total
        max_col = max(total_values.keys(), key=lambda k: total_values[k])
        
        explanation = f"**Data Summary:** Found {row_count:,} records with {col_count} columns. "
        explanation += f"The {max_col} column shows the highest total value of {total_values[max_col]:,.0f}, "
        explanation += f"with an average of {avg_values[max_col]:,.0f} per record. "
        
        if len(numeric_cols) > 1:
            other_cols = [col for col in numeric_cols if col != max_col]
            explanation += f"Other key metrics include {', '.join(other_cols[:2])}."
    else:
        # For non-numeric data
        explanation = f"**Data Summary:** Retrieved {row_count:,} records with {col_count} columns: {', '.join(results_df.columns[:3])}"
        if col_count > 3:
            explanation += f" and {col_count-3} more columns."
    
    return explanation

def process_query(user_query: str):
    """Process user query and generate results."""
    try:
        # Validate input
        if not user_query or not user_query.strip():
            st.warning("Please enter a valid question.")
            return
        
        # Store the user query for explanation
        st.session_state.last_query = user_query.strip()
        
        # Clear sample query after processing
        if 'sample_query' in st.session_state:
            st.session_state.sample_query = None
        
        # Add to query history
        if user_query not in st.session_state.query_history:
            st.session_state.query_history.append(user_query)
        
        # Update schema before generating query to ensure we use the latest database schema
        if st.session_state.db_manager:
            current_schema = st.session_state.db_manager.get_table_schema()
            schema_info = {"tables": current_schema}
            
            # Ensure NLP processor has the latest schema
            if st.session_state.nlp_processor:
                st.session_state.nlp_processor.update_schema(current_schema)
                # Update SQL generator's schema reference
                st.session_state.sql_generator.schema_info = st.session_state.nlp_processor.schema_info
        
        # Generate SQL query
        with st.spinner("Analyzing your question..."):
            try:
                # Ensure SQL generator is initialized
                if not st.session_state.sql_generator:
                    st.error("SQL Generator not initialized. Please refresh the page.")
                    return
                
                query_result = st.session_state.sql_generator.generate_query(user_query)
                
                # Check if query_result is valid
                if not query_result:
                    st.error("Failed to generate query. Please try again.")
                    return
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"Query Analysis Error: {error_msg}")
                
                # Show helpful debugging info
                with st.expander("🔍 Troubleshooting", expanded=True):
                    st.write("**Error Details:**")
                    st.code(error_msg, language=None)
                    
                    # Show current schema
                    if st.session_state.db_manager:
                        try:
                            schema = st.session_state.db_manager.get_table_schema()
                            st.write("**Current Database Schema:**")
                            for table, cols in list(schema.items())[:3]:
                                st.write(f"- **{table}**: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
                        except:
                            st.write("Could not retrieve schema information")
                
                return
        
        # Debug information
        with st.expander("Debug Information", expanded=False):
            st.write("**Parsed Query:**")
            st.json(query_result.get("parsed_query", {}))
            st.write("**Generated SQL:**")
            st.code(query_result.get("sql_query", "No SQL generated"))
            st.write("**Is Valid:**")
            st.write(query_result.get("is_valid", False))
        
        # Check if query has errors
        if not query_result.get("is_valid", False):
            error_msg = query_result.get('error', 'Invalid query generated')
            st.error(f"Query Analysis Error: {error_msg}")
            
            # Try to show what was generated anyway
            if query_result.get("sql_query"):
                st.info("Generated SQL (may have issues):")
                st.code(query_result["sql_query"], language="sql")
                
                # Offer to try executing anyway
                if st.button("⚠️ Try executing anyway (may fail)", key="force_execute"):
                    # Continue to execution with potentially invalid query
                    pass
                else:
                    return
            else:
                # If no SQL was generated, show schema info
                with st.expander("📋 Current Database Schema", expanded=True):
                    if st.session_state.db_manager:
                        try:
                            schema = st.session_state.db_manager.get_table_schema()
                            if schema:
                                for table_name, columns in schema.items():
                                    st.markdown(f"**{table_name}**")
                                    st.code(", ".join(columns), language=None)
                                    st.markdown("---")
                            else:
                                st.warning("No tables found in database")
                        except Exception as e:
                            st.error(f"Error retrieving schema: {str(e)}")
                return
        
        # Display query information
        st.subheader("Query Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Generated SQL:**")
            st.code(query_result["sql_query"], language="sql")
        
        with col2:
            st.write("**Query Type:**")
            st.info(query_result["query_type"])
            
            st.write("**Tables Used:**")
            st.info(", ".join(query_result["tables_used"]))
        
        # Execute SQL query
        with st.spinner("Executing query..."):
            try:
                # Validate SQL query exists
                if not query_result.get("sql_query"):
                    st.error("No SQL query was generated. Please try rephrasing your question.")
                    return
                
                sql_query = query_result["sql_query"]
                
                # Validate database manager is available
                if not st.session_state.db_manager:
                    st.error("Database manager not initialized. Please refresh the page.")
                    return
                
                # Show the generated SQL for debugging
                with st.expander("Generated SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
                    
                    # Also show available tables/columns for reference
                    try:
                        schema = st.session_state.db_manager.get_table_schema()
                        if schema:
                            st.markdown("**Available Tables and Columns:**")
                            for table, cols in list(schema.items())[:5]:
                                st.text(f"{table}: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
                    except:
                        pass
                
                # Execute the query
                results_df = st.session_state.db_manager.execute_query(sql_query)
                st.session_state.current_results = results_df
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"Database Error: {error_msg}")
                
                # Show helpful suggestions with actual schema
                try:
                    schema = st.session_state.db_manager.get_table_schema()
                    if schema:
                        st.warning("**Please check your question and use tables/columns that exist in the database:**")
                        
                        # Show schema information in an expandable section
                        with st.expander("📋 Available Database Schema", expanded=True):
                            for table_name, columns in schema.items():
                                st.markdown(f"**{table_name}**")
                                st.code(", ".join(columns), language=None)
                                st.markdown("---")
                    else:
                        st.warning("No tables found in the database. Please check your database file.")
                except Exception as schema_err:
                    st.warning(f"Could not retrieve database schema: {str(schema_err)}")
                
                # Show the SQL that failed
                if query_result.get("sql_query"):
                    with st.expander("❌ Failed SQL Query", expanded=False):
                        st.code(query_result["sql_query"], language="sql")
                
                return
        
        # Check if results are empty
        if results_df.empty:
            st.info("No results found for your query.")
            st.session_state.current_results = None
            return
        
        # Store results in session state for display in main function
        st.success(f"Query executed successfully. Found {len(results_df)} rows.")
        st.rerun()  # Refresh to show results in main function
    
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")

def _generate_basic_suggestions(schema: Dict[str, List[str]]) -> List[str]:
    """Generate basic query suggestions based on database schema.
    
    Args:
        schema: Dictionary mapping table names to their column lists
        
    Returns:
        List of suggested natural language queries
    """
    suggestions = []
    tables = list(schema.keys())
    
    if not tables:
        return [
            "Show me all data",
            "What are the top items?",
            "Show me data breakdown by category",
            "What trends can you identify?",
            "Compare performance across different segments",
            "Analyze data patterns"
        ]
    
    # Get first few tables
    for table in tables[:3]:
        columns = schema[table]
        
        # Find numeric columns
        numeric_cols = [col for col in columns if any(term in col.lower() for term in 
            ['amount', 'price', 'cost', 'quantity', 'total', 'count', 'value', 'revenue', 'sales'])]
        
        # Find categorical columns
        categorical_cols = [col for col in columns if any(term in col.lower() for term in 
            ['category', 'type', 'status', 'segment', 'name', 'city', 'state', 'region', 'class'])]
        
        # Find date columns
        date_cols = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
        
        # Generate suggestions based on columns found
        if numeric_cols and categorical_cols:
            suggestions.append(f"Show me the total {numeric_cols[0]} by {categorical_cols[0]}")
            suggestions.append(f"What are the top 10 items by {numeric_cols[0]}?")
        
        if categorical_cols and len(categorical_cols) > 1:
            suggestions.append(f"Show me breakdown by {categorical_cols[0]}")
        
        if date_cols and numeric_cols:
            suggestions.append(f"Show me {numeric_cols[0]} trends over time")
        
        if numeric_cols:
            suggestions.append(f"What is the average {numeric_cols[0]}?")
        
        if len(tables) > 1:
            suggestions.append(f"Compare data across {', '.join(tables[:2])}")
    
    # Fill remaining slots with generic suggestions
    generic_suggestions = [
        "What are the top performing items?",
        "Show me data breakdown by category",
        "What trends can you identify?",
        "Compare performance across different segments",
        "Analyze data patterns",
        "What predictions can we make?"
    ]
    
    # Add generic suggestions to fill up to 6 total
    for gen_sug in generic_suggestions:
        if gen_sug not in suggestions and len(suggestions) < 6:
            suggestions.append(gen_sug)
    
    return suggestions[:6]

def main():
    """Main application function."""
    # Display header
    display_header()
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.subheader("Query Input")
    
    # Generate default query suggestions based on database schema
    if st.session_state.db_manager:
        # Get current schema
        current_schema = st.session_state.db_manager.get_table_schema()
        if current_schema:
            schema_hash = hash(str(sorted(current_schema.items())))
            
            # Regenerate suggestions if schema changed or if not cached
            if (st.session_state.get('schema_hash') != schema_hash or 
                not st.session_state.query_suggestions or 
                len(st.session_state.query_suggestions) == 0):
                # Generate default suggestions based on schema
                try:
                    if st.session_state.nlp_processor:
                        st.session_state.query_suggestions = st.session_state.nlp_processor._generate_fallback_suggestions(
                            current_schema, num_suggestions=6
                        )
                    else:
                        # If NLP processor not available, use basic suggestions
                        st.session_state.query_suggestions = _generate_basic_suggestions(current_schema)
                    st.session_state.schema_hash = schema_hash
                except Exception as e:
                    # Use basic suggestions if fallback fails
                    st.session_state.query_suggestions = _generate_basic_suggestions(current_schema)
                    st.session_state.schema_hash = schema_hash
    
    # Display query suggestions
    if st.session_state.query_suggestions and len(st.session_state.query_suggestions) > 0:
        st.markdown("#### 💡 Suggested Queries")
        st.caption("Click any suggestion to use it in your query")
        
        # Display suggestions in a grid layout
        cols = st.columns(3)
        for idx, suggestion in enumerate(st.session_state.query_suggestions):
            with cols[idx % 3]:
                if st.button(
                    suggestion, 
                    key=f"suggest_{idx}", 
                    use_container_width=True,
                    help=f"Click to use: {suggestion}"
                ):
                    st.session_state.sample_query = suggestion
                    st.rerun()
        
        st.markdown("---")
    
    # Check for sample query from sidebar or suggestions
    if 'sample_query' in st.session_state and st.session_state.sample_query:
        user_query = st.text_area(
            "Enter your business question in natural language",
            value=st.session_state.sample_query,
            height=100,
            help="Examples: 'Show me sales by category', 'Predict customer churn', 'What are the top products?'",
            label_visibility="visible"
        )
        # Don't clear the sample query here - let it be processed first
    else:
        user_query = st.text_area(
            "Enter your business question in natural language",
            height=100,
            help="Examples: 'Show me sales by category', 'Predict customer churn', 'What are the top products?'",
            label_visibility="visible"
        )
    
    # Process query button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Analyze", type="primary", use_container_width=True):
            if user_query.strip():
                process_query(user_query.strip())
            else:
                st.warning("Please enter a question to analyze.")
    
    # Display current results if available
    if st.session_state.current_results is not None:
        results_df = st.session_state.current_results
        
        # Display data summary
        st.subheader("Results Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(results_df):,}")
        with col2:
            st.metric("Total Columns", len(results_df.columns))
        with col3:
            st.metric("Memory Usage", f"{results_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display data table
        st.subheader("Data Table")
        try:
            # Add filters
            filters = StreamlitHelpers.create_sidebar_filters(results_df)
            filtered_results = StreamlitHelpers.apply_filters(results_df, filters)
            
            # Display filtered results
            if not filtered_results.equals(results_df):
                st.info(f"Showing {len(filtered_results)} filtered results out of {len(results_df)} total")
            
            StreamlitHelpers.display_dataframe(filtered_results)
        except Exception as e:
            st.warning(f"Could not apply filters: {str(e)}")
            st.info("Displaying unfiltered results:")
            StreamlitHelpers.display_dataframe(results_df)
        
        # Generate visualizations
        st.subheader("Visualizations")
        try:
            # Auto-generate visualizations
            figures = VisualizationGenerator.auto_visualize(results_df)
            
            if figures:
                for i, fig in enumerate(figures):
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add some spacing between charts
                        if i < len(figures) - 1:
                            st.markdown("---")
                    except Exception as chart_error:
                        st.warning(f"Could not display chart {i+1}: {str(chart_error)}")
            else:
                st.info("No suitable visualizations could be generated for this data.")
        
        except Exception as e:
            st.warning(f"Could not generate visualizations: {str(e)}")
        
        # Generate explanation
        st.subheader("Analysis Explanation")
        try:
            # Get the actual user query from session state or use a default
            user_query = st.session_state.get('last_query', 'data analysis')
            explanation = st.session_state.nlp_processor.explain_results(
                user_query, results_df, "sql"
            )
            st.write(explanation)
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")
            # Provide a basic explanation as fallback
            st.write(generate_basic_explanation(results_df))
        
        # Suggest follow-up questions
        st.subheader("Suggested Follow-up Questions")
        try:
            follow_up_questions = st.session_state.nlp_processor.suggest_follow_up_questions(
                "User query", results_df
            )
            
            cols = st.columns(min(3, len(follow_up_questions)))
            for i, question in enumerate(follow_up_questions):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if st.button(question, key=f"followup_{i}", use_container_width=True):
                        st.session_state.sample_query = question
                        st.rerun()
        except Exception as e:
            st.warning(f"Could not generate follow-up questions: {str(e)}")
        
        # ML Analysis section
        if len(results_df) > 10:  # Only show ML options for larger datasets
            st.markdown("---")
            st.markdown("### Machine Learning Analysis")
            
            # Create a modern container for ML section
            with st.container():
                st.markdown("""
                <div class="ml-section">
                    <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">Advanced Analytics</h4>
                    <p style="color: var(--text-secondary); margin: 0;">Unlock insights with machine learning</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Target column selection with modern styling
                col1, col2 = st.columns([2, 1])
                with col1:
                    target_column = st.selectbox(
                        "Select Target Variable",
                        options=results_df.columns.tolist(),
                        help="Choose the column you want to predict or analyze",
                        key="ml_target_select"
                    )
                
                with col2:
                    st.markdown("#### Data Overview")
                    st.metric("Samples", f"{len(results_df):,}")
                    st.metric("Features", len(results_df.columns) - 1)
                
                if target_column:
                    # Data type and analysis info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        data_type = "Numeric" if pd.api.types.is_numeric_dtype(results_df[target_column]) else "Categorical"
                        st.info(f"**Data Type:** {data_type}")
                    with col2:
                        unique_vals = results_df[target_column].nunique()
                        st.info(f"**Unique Values:** {unique_vals}")
                    with col3:
                        missing_vals = results_df[target_column].isnull().sum()
                        st.info(f"**Missing Values:** {missing_vals}")
                    
                    # ML Analysis button with modern styling
                    if st.button("Run Machine Learning Analysis", type="primary", use_container_width=True):
                        with st.spinner("Analyzing data and training model..."):
                            try:
                                # Analyze data
                                analysis = st.session_state.ml_pipeline.analyze_data(results_df, target_column)
                                
                                # Display analysis in modern cards
                                st.markdown("#### Data Analysis Results")
                                
                                # Analysis metrics in cards
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Problem Type", analysis.get("problem_type", "Unknown").title())
                                with col2:
                                    st.metric("Numeric Columns", len(analysis.get("numeric_columns", [])))
                                with col3:
                                    st.metric("Categorical Columns", len(analysis.get("categorical_columns", [])))
                                with col4:
                                    missing_pct = (results_df.isnull().sum().sum() / (len(results_df) * len(results_df.columns))) * 100
                                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                                
                                # Data quality recommendations
                                if analysis.get("recommendations"):
                                    st.markdown("#### Data Quality Recommendations")
                                    for rec in analysis["recommendations"]:
                                        st.warning(rec)
                                
                                # Prepare data
                                features_df, target_series = st.session_state.ml_pipeline.prepare_data(
                                    results_df, target_column
                                )
                                
                                # Train model
                                problem_type = analysis.get("problem_type", "regression")
                                ml_results = st.session_state.ml_pipeline.train_model(
                                    features_df, target_series, problem_type
                                )
                                
                                if ml_results.get("training_successful", False):
                                    st.success("Model trained successfully")
                                    
                                    # Display results in modern format
                                    st.markdown("#### Model Performance")
                                    
                                    # Metrics in a nice layout
                                    metrics = ml_results.get("metrics", {})
                                    if metrics:
                                        col1, col2, col3, col4 = st.columns(4)
                                        metric_cols = list(metrics.keys())
                                        
                                        for i, (col, metric_key) in enumerate(zip([col1, col2, col3, col4], metric_cols[:4])):
                                            with col:
                                                value = metrics[metric_key]
                                                if isinstance(value, float):
                                                    st.metric(metric_key.replace("_", " ").title(), f"{value:.4f}")
                                                else:
                                                    st.metric(metric_key.replace("_", " ").title(), value)
                                    
                                    # Model summary in simple text format
                                    st.markdown("#### Model Summary")
                                    
                                    summary = st.session_state.ml_pipeline.get_model_summary()
                                    
                                    # Create a modern info box for model summary
                                    with st.container():
                                        st.markdown("##### Model Configuration")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                            st.text(f"Problem Type: {summary.get('problem_type', 'Unknown').title()}")
                                            st.text(f"Model Type: {summary.get('model_type', 'Unknown').replace('_', ' ').title()}")
                                            st.text(f"Target Column: {summary.get('target_column', 'Unknown')}")
                                        
                                    with col2:
                                            st.text(f"Training Samples: {len(features_df):,}")
                                            st.text(f"Features Count: {len(features_df.columns)}")
                                            feature_cols = ', '.join(summary.get('feature_columns', [])[:5])
                                            if len(summary.get('feature_columns', [])) > 5:
                                                feature_cols += "..."
                                            st.text(f"Features: {feature_cols}")
                                    
                                    # Additional model information
                                    if summary.get('metrics'):
                                        st.markdown("##### Model Performance Metrics")
                                        metrics = summary.get('metrics', {})
                                        metric_cols = st.columns(min(len(metrics), 4))
                                        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                                            with metric_cols[idx % len(metric_cols)]:
                                                if isinstance(metric_value, float):
                                                    st.metric(metric_name.replace("_", " ").title(), f"{metric_value:.4f}")
                                                else:
                                                    st.metric(metric_name.replace("_", " ").title(), metric_value)
                                    
                                    # Predictions visualization if available
                                    predictions = ml_results.get("predictions", [])
                                    if len(predictions) > 0:
                                        st.markdown("#### Predictions Visualization")
                                        
                                        # Create a DataFrame for visualization
                                        try:
                                            # Ensure we have the same length for both arrays
                                            min_len = min(len(target_series), len(predictions))
                                            actual_vals = target_series.iloc[:min_len] if hasattr(target_series, 'iloc') else target_series[:min_len]
                                            pred_vals = predictions[:min_len]
                                            
                                            pred_df = pd.DataFrame({
                                                'Actual': actual_vals,
                                                'Predicted': pred_vals
                                            })
                                        except Exception as e:
                                            st.warning(f"Could not create predictions DataFrame: {str(e)}")
                                            pred_df = pd.DataFrame()
                                        
                                        # Scatter plot for regression
                                        if problem_type == "regression" and not pred_df.empty:
                                            try:
                                                import plotly.express as px
                                                import plotly.graph_objects as go
                                                
                                                fig = px.scatter(
                                                    pred_df, 
                                                    x='Actual', 
                                                    y='Predicted',
                                                    title="Actual vs Predicted Values",
                                                    labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'}
                                                )
                                                
                                                # Add perfect prediction line
                                                min_val = float(pred_df['Actual'].min())
                                                max_val = float(pred_df['Actual'].max())
                                                fig.add_trace(go.Scatter(
                                                    x=[min_val, max_val],
                                                    y=[min_val, max_val],
                                                    mode='lines',
                                                    name='Perfect Prediction',
                                                    line=dict(dash='dash', color='red')
                                                ))
                                                
                                                fig.update_layout(showlegend=True)
                                                st.plotly_chart(fig, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not create scatter plot: {str(e)}")
                                        
                                        # Confusion matrix for classification
                                        elif problem_type == "classification":
                                            try:
                                                from sklearn.metrics import confusion_matrix
                                                import plotly.express as px
                                                import plotly.graph_objects as go
                                                
                                                # Ensure we have the same length for both arrays
                                                min_len = min(len(target_series), len(predictions))
                                                y_true = target_series.iloc[:min_len] if hasattr(target_series, 'iloc') else target_series[:min_len]
                                                y_pred = predictions[:min_len]
                                                
                                                cm = confusion_matrix(y_true, y_pred)
                                                fig = px.imshow(
                                                    cm, 
                                                    text_auto=True,
                                                    title="Confusion Matrix",
                                                    labels=dict(x="Predicted", y="Actual")
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not create confusion matrix: {str(e)}")
                                    
                                    # Feature importance if available
                                    if hasattr(st.session_state.ml_pipeline, 'model') and st.session_state.ml_pipeline.model:
                                        model = st.session_state.ml_pipeline.model
                                        if hasattr(model, 'feature_importances_'):
                                            try:
                                                import plotly.express as px
                                                
                                                st.markdown("#### Feature Importance")
                                                importance_df = pd.DataFrame({
                                                    'Feature': features_df.columns,
                                                    'Importance': model.feature_importances_
                                                }).sort_values('Importance', ascending=True)
                                                
                                                fig = px.bar(
                                                    importance_df, 
                                                    x='Importance', 
                                                    y='Feature',
                                                    orientation='h',
                                                    title="Feature Importance",
                                                    color='Importance',
                                                    color_continuous_scale='Viridis'
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not create feature importance chart: {str(e)}")
                                
                                else:
                                    st.error(f"ML Analysis failed: {ml_results.get('error', 'Unknown error')}")
                                    
                            except Exception as e:
                                st.error(f"Error during ML analysis: {str(e)}")
                
                # Tips section with modern styling
                st.markdown("#### Analysis Tips")
                tip_col1, tip_col2, tip_col3 = st.columns(3)
                
                with tip_col1:
                    st.markdown("""
                    **Target Selection**
                    - Choose numeric columns for regression
                    - Choose categorical columns for classification
                    - Ensure sufficient data quality
                    """)
                
                with tip_col2:
                    st.markdown("""
                    **Data Quality**
                    - More data = better models
                    - Clean missing values first
                    - Check for outliers
                    """)
                
                with tip_col3:
                    st.markdown("""
                    **Model Types**
                    - Regression: Predicting numbers
                    - Classification: Predicting categories
                    - Clustering: Finding patterns
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: var(--text-secondary); font-size: 0.875rem; padding: 1rem 0;">'
        'Speak2Data · Powered by Google Gemini Pro · Built with Streamlit'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
