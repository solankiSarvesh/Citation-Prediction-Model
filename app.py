import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from typing import Dict, List, Tuple

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="AI-Enhanced Citation Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS STYLING
# ==============================================================================

st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content container */
    .block-container {
        padding: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1 {
        color: #667eea;
        font-weight: 800;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #764ba2;
        font-weight: 700;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-top: 2rem;
    }
    
    h3 {
        color: #5a67d8;
        font-weight: 600;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        }
        to {
            box-shadow: 0 15px 45px rgba(118, 75, 162, 0.6);
        }
    }
    
    .prediction-number {
        font-size: 5rem;
        font-weight: 900;
        margin: 20px 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 15px 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* CSP/Fuzzy/State Space boxes */
    .csp-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .fuzzy-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .state-box {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 18px;
        border: none;
        padding: 15px 40px;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Warning boxes */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px;
        font-size: 14px;
    }
    
    .badge-q1 { background: #667eea; color: white; }
    .badge-q2 { background: #43e97b; color: white; }
    .badge-q3 { background: #f5af19; color: white; }
    .badge-q4 { background: #fa709a; color: white; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 14px;
        margin-top: 50px;
        border-top: 2px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUZZY LOGIC SYSTEM
# ==============================================================================

class FuzzyLogicSystem:
    """Fuzzy Logic for paper quality assessment"""
    
    @staticmethod
    def triangular_membership(x, a, b, c):
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal_membership(x, a, b, c, d):
        """Trapezoidal membership function"""
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:
            return (d - x) / (d - a)
    
    def fuzzify_sjr(self, sjr):
        """Convert SJR to fuzzy sets"""
        return {
            'Low': self.trapezoidal_membership(sjr, 0, 0, 0.5, 1.0),
            'Medium': self.triangular_membership(sjr, 0.5, 1.5, 2.5),
            'High': self.trapezoidal_membership(sjr, 1.5, 2.5, 10, 10)
        }
    
    def fuzzify_h_index(self, h_index):
        """Convert H-Index to fuzzy sets"""
        return {
            'Low': self.trapezoidal_membership(h_index, 0, 0, 20, 40),
            'Medium': self.triangular_membership(h_index, 20, 50, 80),
            'High': self.trapezoidal_membership(h_index, 60, 100, 500, 500)
        }
    
    def fuzzify_quartile(self, quartile_str):
        """Convert Quartile to fuzzy quality levels"""
        mappings = {
            'Q1': {'Excellent': 1.0, 'Good': 0.3, 'Average': 0.0, 'Poor': 0.0},
            'Q2': {'Excellent': 0.3, 'Good': 1.0, 'Average': 0.3, 'Poor': 0.0},
            'Q3': {'Excellent': 0.0, 'Good': 0.3, 'Average': 1.0, 'Poor': 0.3},
            'Q4': {'Excellent': 0.0, 'Good': 0.0, 'Average': 0.3, 'Poor': 1.0}
        }
        return mappings.get(quartile_str, {'Excellent': 0, 'Good': 0, 'Average': 0.5, 'Poor': 0.5})
    
    def fuzzify_year(self, year, current_year=2025):
        """Convert year to recency fuzzy sets"""
        age = current_year - year
        return {
            'Recent': self.trapezoidal_membership(age, 0, 0, 2, 5),
            'Moderate': self.triangular_membership(age, 3, 8, 15),
            'Old': self.trapezoidal_membership(age, 10, 20, 50, 50)
        }
    
    def apply_fuzzy_rules(self, sjr_fuzzy, h_fuzzy, q_fuzzy, year_fuzzy):
        """Apply fuzzy IF-THEN rules for citation potential"""
        rules = []
        
        # Rule 1: IF SJR is High AND Quartile is Excellent THEN Very_High
        rules.append(min(sjr_fuzzy['High'], q_fuzzy['Excellent']))
        
        # Rule 2: IF H-Index is High AND Year is Recent THEN High
        rules.append(min(h_fuzzy['High'], year_fuzzy['Recent']))
        
        # Rule 3: IF SJR is Medium AND Quartile is Good THEN Medium
        rules.append(min(sjr_fuzzy['Medium'], q_fuzzy['Good']) * 0.7)
        
        # Rule 4: IF Quartile is Poor THEN Low
        rules.append(q_fuzzy['Poor'] * 0.3)
        
        # Rule 5: IF Year is Old THEN reduced
        rules.append((1 - year_fuzzy['Old']) * 0.6)
        
        # Rule 6: IF SJR is Low AND H-Index is Low THEN Very_Low
        rules.append(min(sjr_fuzzy['Low'], h_fuzzy['Low']) * 0.2)
        
        citation_potential = max(rules)
        return citation_potential
    
    def defuzzify_to_multiplier(self, citation_potential):
        """Convert fuzzy output to concrete multiplier"""
        return 0.5 + (citation_potential * 1.5)
    
    def get_fuzzy_features(self, year, quartile, sjr, h_index):
        """Get all fuzzy features as a vector"""
        sjr_f = self.fuzzify_sjr(sjr)
        h_f = self.fuzzify_h_index(h_index)
        q_f = self.fuzzify_quartile(quartile)
        year_f = self.fuzzify_year(year)
        
        # Flatten to feature vector
        features = []
        for fuzzy_set in [sjr_f, h_f, q_f, year_f]:
            features.extend(fuzzy_set.values())
        
        return np.array(features)

# ==============================================================================
# CONSTRAINT SATISFACTION PROBLEM (CSP)
# ==============================================================================

class PaperCSP:
    """CSP for validating paper input"""
    
    def __init__(self):
        self.domains = {
            'Year': (2000, 2025),
            'SJR': (0.0, 10.0),
            'H-Index': (0, 500),
            'Page count': (1, 100),
            'Quartile': ['Q1', 'Q2', 'Q3', 'Q4'],
            'Document Type': ['Article', 'Conference Paper', 'Review', 'Book Chapter', 'Letter'],
        }
        self.constraints = []
        self._define_constraints()
    
    def _define_constraints(self):
        """Define logical constraints"""
        
        def q1_sjr_constraint(variables):
            if variables['Quartile'] == 'Q1':
                return variables['SJR'] >= 0.4
            return True
        
        def h_index_quartile_constraint(variables):
            if variables['H-Index'] > 100:
                return variables['Quartile'] in ['Q1', 'Q2']
            return True
        
        def year_validity_constraint(variables):
            return 2000 <= variables['Year'] <= 2025
        
        def page_count_type_constraint(variables):
            if variables['Document Type'] == 'Letter':
                return variables['Page count'] <= 10
            elif variables['Document Type'] == 'Book Chapter':
                return variables['Page count'] >= 10
            return True
        
        def sjr_quartile_consistency(variables):
            sjr = variables['SJR']
            quartile = variables['Quartile']
            if quartile == 'Q1' and sjr < 0.3:
                return False
            if quartile == 'Q4' and sjr > 1.5:
                return False
            return True
        
        def text_not_empty_constraint(variables):
            return (len(str(variables.get('Title', ''))) > 5 and
                    len(str(variables.get('Abstract', ''))) > 20)
        
        self.constraints = [
            ('Q1_SJR_Consistency', q1_sjr_constraint),
            ('H-Index_Quartile_Consistency', h_index_quartile_constraint),
            ('Year_Validity', year_validity_constraint),
            ('Page_Count_Type_Consistency', page_count_type_constraint),
            ('SJR_Quartile_Consistency', sjr_quartile_consistency),
            ('Text_Not_Empty', text_not_empty_constraint)
        ]
    
    def validate(self, variables: Dict) -> Tuple[bool, List[str]]:
        """Validate constraints"""
        violations = []
        
        # Check domain constraints
        for var, value in variables.items():
            if var in self.domains:
                domain = self.domains[var]
                if isinstance(domain, tuple):
                    if not (domain[0] <= value <= domain[1]):
                        violations.append(f"{var} value {value} outside domain {domain}")
                elif isinstance(domain, list):
                    if value not in domain:
                        violations.append(f"{var} value '{value}' not in allowed values")
        
        # Check logical constraints
        for constraint_name, constraint_func in self.constraints:
            try:
                if not constraint_func(variables):
                    violations.append(f"Constraint violated: {constraint_name}")
            except KeyError:
                pass
        
        return len(violations) == 0, violations
    
    def suggest_fixes(self, variables: Dict, violations: List[str]) -> Dict:
        """Suggest fixes for violations"""
        suggestions = {}
        
        for violation in violations:
            if 'Q1_SJR' in violation and variables['Quartile'] == 'Q1':
                suggestions['SJR'] = max(0.4, variables['SJR'])
            
            if 'Year_Validity' in violation:
                suggestions['Year'] = max(2000, min(2025, variables['Year']))
            
            if 'SJR_Quartile_Consistency' in violation:
                if variables['Quartile'] == 'Q4' and variables['SJR'] > 1.5:
                    suggestions['Quartile'] = 'Q2'
        
        return suggestions

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def clean_text(text):
    """Clean text for TF-IDF processing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_encode(encoder, value):
    """Encode categorical value safely"""
    value = str(value)
    classes = encoder.classes_
    
    if value not in classes:
        if "Unknown" not in classes:
            encoder.classes_ = np.append(classes, "Unknown")
        return encoder.transform(["Unknown"])[0]
    
    return encoder.transform([value])[0]

@st.cache_resource
def load_model_artifacts():
    """Load all AI-enhanced model artifacts"""
    try:
        artifacts = {
            'model': joblib.load('ai_enhanced_model_xgboost.joblib'),
            'vectorizers': joblib.load('ai_enhanced_model_vectorizers.joblib'),
            'encoders': joblib.load('ai_enhanced_model_encoders.joblib'),
            'selected_features': joblib.load('ai_enhanced_model_selected_features.joblib'),
            'feature_names': joblib.load('ai_enhanced_model_feature_names.joblib')
        }
        return artifacts
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure you've run the training script first to generate model files.")
        return None

def predict_citations_with_ai(artifacts, fuzzy_system, csp_system, paper_data):
    """Make prediction with AI enhancements"""
    
    # STEP 1: CSP Validation
    is_valid, violations = csp_system.validate(paper_data)
    suggestions = {}
    if not is_valid:
        suggestions = csp_system.suggest_fixes(paper_data, violations)
    
    # STEP 2: Fuzzy Logic Analysis
    sjr_fuzzy = fuzzy_system.fuzzify_sjr(paper_data['SJR'])
    h_fuzzy = fuzzy_system.fuzzify_h_index(paper_data['H-Index'])
    q_fuzzy = fuzzy_system.fuzzify_quartile(paper_data['Quartile'])
    year_fuzzy = fuzzy_system.fuzzify_year(paper_data['Year'])
    
    citation_potential = fuzzy_system.apply_fuzzy_rules(sjr_fuzzy, h_fuzzy, q_fuzzy, year_fuzzy)
    fuzzy_multiplier = fuzzy_system.defuzzify_to_multiplier(citation_potential)
    
    # STEP 3: ML Prediction
    title_clean = clean_text(paper_data['Title'])
    abstract_clean = clean_text(paper_data['Abstract'])
    author_kw_clean = clean_text(paper_data.get('Author Keywords', ''))
    index_kw_clean = clean_text(paper_data.get('Index Keywords', ''))
    
    # Use vectorizers from loaded artifacts
    vectorizers = artifacts['vectorizers']
    encoders = artifacts['encoders']
    
    t1 = vectorizers['title'].transform([title_clean]).toarray()
    t2 = vectorizers['abstract'].transform([abstract_clean]).toarray()
    t3 = vectorizers['author_kw'].transform([author_kw_clean]).toarray()
    t4 = vectorizers['index_kw'].transform([index_kw_clean]).toarray()
    
    ctry = safe_encode(encoders['Country'], paper_data['Country'])
    pub = safe_encode(encoders['Publisher'], paper_data['Publisher'])
    doc = safe_encode(encoders['Document Type'], paper_data['Document Type'])
    lng = safe_encode(encoders['Language of Original Document'], paper_data['Language of Original Document'])
    
    quartile_map = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}
    quartile_val = quartile_map[paper_data['Quartile']]
    
    # Get fuzzy features
    fuzzy_feat = fuzzy_system.get_fuzzy_features(
        paper_data['Year'],
        paper_data['Quartile'],
        paper_data['SJR'],
        paper_data['H-Index']
    )
    
    # Combine all features
    numeric = [paper_data['Year'], quartile_val, paper_data['SJR'], 
               paper_data['H-Index'], paper_data['Page count'], ctry, pub, doc, lng]
    X_input = np.hstack([numeric, t1[0], t2[0], t3[0], t4[0], fuzzy_feat])
    
    # Apply feature selection if used
    selected_features = artifacts.get('selected_features')
    if selected_features is not None and len(selected_features) > 0:
        X_input = X_input[selected_features]
    
    pred_log = artifacts['model'].predict(np.array([X_input]))[0]
    base_prediction = np.expm1(pred_log)
    
    # STEP 4: Apply fuzzy multiplier
    final_prediction = int(round(base_prediction * fuzzy_multiplier))
    final_prediction = max(0, final_prediction)
    
    # Confidence
    if is_valid and citation_potential > 0.7:
        confidence = "High"
    elif is_valid:
        confidence = "Medium"
    else:
        confidence = "Low (constraint violations)"
    
    return {
        'prediction': final_prediction,
        'base_ml_prediction': int(base_prediction),
        'fuzzy_multiplier': fuzzy_multiplier,
        'citation_potential': citation_potential,
        'csp_valid': is_valid,
        'csp_violations': violations,
        'csp_suggestions': suggestions,
        'confidence': confidence,
        'fuzzy_analysis': {
            'sjr': sjr_fuzzy,
            'h_index': h_fuzzy,
            'quartile': q_fuzzy,
            'year': year_fuzzy
        },
        'features_used': len(selected_features) if selected_features else 'All'
    }

def create_fuzzy_membership_chart(fuzzy_data, title):
    """Create bar chart for fuzzy membership values"""
    categories = list(fuzzy_data.keys())
    values = list(fuzzy_data.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                cmin=0,
                cmax=1
            ),
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Fuzzy Set",
        yaxis_title="Membership Value",
        yaxis=dict(range=[0, 1.1]),
        height=300,
        showlegend=False
    )
    
    return fig

def create_gauge_chart(value, max_value=300):
    """Create a gauge chart for prediction"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Citations", 'font': {'size': 28, 'color': '#667eea'}},
        delta={'reference': max_value/2, 'increasing': {'color': "#38ef7d"}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#667eea",
            'steps': [
                {'range': [0, max_value*0.25], 'color': '#ffeaa7'},
                {'range': [max_value*0.25, max_value*0.5], 'color': '#74b9ff'},
                {'range': [max_value*0.5, max_value*0.75], 'color': '#a29bfe'},
                {'range': [max_value*0.75, max_value], 'color': '#fd79a8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value*0.9
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#667eea", 'family': "Arial"},
        height=400
    )
    
    return fig

def get_citation_category(prediction):
    """Categorize prediction into impact levels"""
    if prediction < 10:
        return {
            'category': 'Low Impact',
            'color': '#e74c3c',
            'emoji': '📉',
            'description': 'Expected to receive minimal citations'
        }
    elif prediction < 50:
        return {
            'category': 'Moderate Impact',
            'color': '#f39c12',
            'emoji': '📊',
            'description': 'Expected to receive moderate attention'
        }
    elif prediction < 100:
        return {
            'category': 'High Impact',
            'color': '#27ae60',
            'emoji': '📈',
            'description': 'Expected to be well-cited in the field'
        }
    else:
        return {
            'category': 'Exceptional Impact',
            'color': '#9b59b6',
            'emoji': '🌟',
            'description': 'Expected to be highly influential'
        }

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    
    # Header
    st.markdown("""
        <h1>🎯 AI-Enhanced Citation Predictor</h1>
        <p style='text-align: center; font-size: 20px; color: #666; margin-bottom: 30px;'>
            XGBoost + Fuzzy Logic + CSP + State Space Search
        </p>
    """, unsafe_allow_html=True)
    
    # Initialize AI systems
    fuzzy_system = FuzzyLogicSystem()
    csp_system = PaperCSP()
    
    # Load model
    artifacts = load_model_artifacts()
    
    if artifacts is None:
        st.error("⚠️ Failed to load model artifacts. Please ensure all files are present.")
        st.stop()
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("## 📝 Research Paper Information")
        
        st.markdown("### 📄 Document Details")
        title = st.text_input(
            "Paper Title",
            value="Deep Learning Approaches for DeepFake Detection",
            help="Enter the full title of your research paper"
        )
        
        abstract = st.text_area(
            "Abstract",
            value="This paper presents novel deep learning methods for detecting deepfake videos using convolutional neural networks and attention mechanisms.",
            height=150,
            help="Enter the paper abstract"
        )
        
        st.markdown("### 🔑 Keywords")
        author_kw = st.text_area(
            "Author Keywords",
            value="deepfake detection; deep learning; CNN; attention mechanism",
            help="Separate keywords with semicolons"
        )
        
        index_kw = st.text_area(
            "Index Keywords",
            value="computer vision; neural networks; video forensics; AI",
            help="Separate keywords with semicolons"
        )
        
        st.markdown("### 📊 Publication Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
            quartile = st.selectbox("Quartile", ["Q1", "Q2", "Q3", "Q4"], index=0)
        
        with col2:
            sjr = st.number_input("SJR Score", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            h_index = st.number_input("H-Index", min_value=0, max_value=500, value=50)
        
        page_count = st.number_input("Page Count", min_value=1, max_value=100, value=12)
        
        st.markdown("### 🌐 Metadata")
        country = st.selectbox(
            "Country",
            ["United States", "China", "United Kingdom", "India", "Germany", 
             "Canada", "Australia", "Japan", "South Korea", "Malaysia", "Unknown"],
            index=0
        )
        
        publisher = st.selectbox(
            "Publisher",
            ["Elsevier", "Springer", "IEEE", "ACM", "Wiley", "MDPI", 
             "Nature", "Science", "Taylor & Francis", "Unknown"],
            index=2
        )
        
        doc_type = st.selectbox(
            "Document Type",
            ["Article", "Conference Paper", "Review", "Book Chapter", "Letter"],
            index=0
        )
        
        language = st.selectbox(
            "Language",
            ["English", "Chinese", "Spanish", "German", "French", "Japanese"],
            index=0
        )
        
        st.markdown("---")
        predict_button = st.button("🚀 PREDICT WITH AI", use_container_width=True)
    
    # Main content area
    if not predict_button:
        # Welcome screen
        st.markdown("""
            <div class='info-card'>
                <h2>👋 Welcome to the AI-Enhanced Citation Predictor!</h2>
                <p style='font-size: 18px; line-height: 1.8;'>
                    This advanced system combines multiple AI techniques:
                </p>
                <ul style='font-size: 16px; line-height: 2;'>
                    <li>🤖 <b>XGBoost Machine Learning</b> - 600 decision trees for accurate predictions</li>
                    <li>🧠 <b>Fuzzy Logic</b> - Reasoning with uncertainty and linguistic variables</li>
                    <li>🔍 <b>Constraint Satisfaction</b> - Validates inputs against domain knowledge</li>
                    <li>🎯 <b>State Space Search</b> - Optimizes feature selection (used during training)</li>
                </ul>
                <p style='font-size: 18px; margin-top: 20px;'>
                    <b>👈 Fill in the details in the sidebar and click "PREDICT WITH AI" to get started!</b>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display AI Techniques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='fuzzy-box'>
                    <h3>🧠 Fuzzy Logic</h3>
                    <p><b>Handles Uncertainty</b></p>
                    <ul>
                        <li>Membership functions</li>
                        <li>IF-THEN rules</li>
                        <li>Citation potential</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='csp-box'>
                    <h3>🔍 CSP Validation</h3>
                    <p><b>Domain Constraints</b></p>
                    <ul>
                        <li>Input validation</li>
                        <li>Logical consistency</li>
                        <li>Auto-suggestions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='state-box'>
                    <h3>🎯 State Space</h3>
                    <p><b>Feature Optimization</b></p>
                    <ul>
                        <li>Hill climbing</li>
                        <li>Best features</li>
                        <li>Performance boost</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        # Prepare paper data
        paper_data = {
            'Title': title,
            'Abstract': abstract,
            'Author Keywords': author_kw,
            'Index Keywords': index_kw,
            'Year': year,
            'Quartile': quartile,
            'SJR': sjr,
            'H-Index': h_index,
            'Page count': page_count,
            'Country': country,
            'Publisher': publisher,
            'Document Type': doc_type,
            'Language of Original Document': language
        }
        
        # Make prediction
        with st.spinner("🔮 Running AI-enhanced prediction..."):
            result = predict_citations_with_ai(artifacts, fuzzy_system, csp_system, paper_data)
            category_info = get_citation_category(result['prediction'])
        
        st.markdown("---")
        
        # ========================================
        # STEP 1: CSP VALIDATION RESULTS
        # ========================================
        st.markdown("## 🔍 Step 1: Constraint Satisfaction Problem (CSP)")
        
        if result['csp_valid']:
            st.markdown("""
                <div class='success-box'>
                    <h3>✅ All Constraints Satisfied!</h3>
                    <p>Your input passes all domain and logical constraints.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='warning-box'>
                    <h3>⚠️ Constraint Violations Detected</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ❌ Violations")
                for violation in result['csp_violations']:
                    st.error(f"• {violation}")
            
            with col2:
                st.markdown("### 💡 Suggested Fixes")
                if result['csp_suggestions']:
                    for key, value in result['csp_suggestions'].items():
                        st.info(f"• {key}: Change to **{value}**")
                else:
                    st.info("No automatic fixes available")
        
        st.markdown("---")
        
        # ========================================
        # STEP 2: FUZZY LOGIC ANALYSIS
        # ========================================
        st.markdown("## 🧠 Step 2: Fuzzy Logic Reasoning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Fuzzy Memberships")
            
            # SJR Fuzzy
            fig_sjr = create_fuzzy_membership_chart(result['fuzzy_analysis']['sjr'], "SJR Fuzzification")
            st.plotly_chart(fig_sjr, use_container_width=True)
            
            # H-Index Fuzzy
            fig_h = create_fuzzy_membership_chart(result['fuzzy_analysis']['h_index'], "H-Index Fuzzification")
            st.plotly_chart(fig_h, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Quality Assessment")
            
            # Quartile Fuzzy
            fig_q = create_fuzzy_membership_chart(result['fuzzy_analysis']['quartile'], "Quartile Quality")
            st.plotly_chart(fig_q, use_container_width=True)
            
            # Year Fuzzy
            fig_y = create_fuzzy_membership_chart(result['fuzzy_analysis']['year'], "Year Recency")
            st.plotly_chart(fig_y, use_container_width=True)
        
        # Fuzzy Rules Output
        st.markdown("""
            <div class='fuzzy-box'>
                <h3>🎲 Fuzzy Inference Results</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Citation Potential", f"{result['citation_potential']:.3f}", help="Fuzzy output (0-1)")
        with col2:
            st.metric("Fuzzy Multiplier", f"{result['fuzzy_multiplier']:.3f}x", help="Applied to base prediction")
        with col3:
            st.metric("Confidence", result['confidence'])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Fuzzy Rules Explanation
        with st.expander("📖 View Fuzzy Rules Applied"):
            st.markdown("""
            **Fuzzy IF-THEN Rules:**
            
            1. **IF** SJR is High **AND** Quartile is Excellent **THEN** Citation_Potential is Very_High
            2. **IF** H-Index is High **AND** Year is Recent **THEN** Citation_Potential is High
            3. **IF** SJR is Medium **AND** Quartile is Good **THEN** Citation_Potential is Medium
            4. **IF** Quartile is Poor **THEN** Citation_Potential is Low
            5. **IF** Year is Old **THEN** Citation_Potential is Reduced
            6. **IF** SJR is Low **AND** H-Index is Low **THEN** Citation_Potential is Very_Low
            
            **Aggregation:** MAX operator (union of all rule activations)
            
            **Defuzzification:** Linear mapping to multiplier range [0.5, 2.0]
            """)
        
        st.markdown("---")
        
        # ========================================
        # STEP 3: MACHINE LEARNING PREDICTION
        # ========================================
        st.markdown("## 🤖 Step 3: XGBoost Machine Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='info-card'>
                    <h3>📊 ML Model Details</h3>
                    <ul>
                        <li><b>Algorithm:</b> XGBoost Gradient Boosting</li>
                        <li><b>Trees:</b> 600 estimators</li>
                        <li><b>Features:</b> 409 (TF-IDF + Numeric)</li>
                        <li><b>R² Score:</b> ~0.94</li>
                        <li><b>Training Data:</b> 876 papers</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='info-card'>
                    <h3>🔢 Prediction Breakdown</h3>
            """, unsafe_allow_html=True)
            
            st.metric("Base ML Prediction", f"{result['base_ml_prediction']} citations", 
                     help="Raw XGBoost output")
            st.metric("Fuzzy Adjustment", f"×{result['fuzzy_multiplier']:.3f}", 
                     help="Multiplier from fuzzy logic")
            st.metric("Final Prediction", f"{result['prediction']} citations", 
                     help="Base × Fuzzy Multiplier")
            
            # Show feature selection info
            if result.get('features_used') != 'All':
                st.info(f"🎯 Using {result['features_used']} optimized features (State Space Search)")
            else:
                st.info(f"📊 Using all {result.get('features_used', 'available')} features")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========================================
        # FINAL PREDICTION DISPLAY
        # ========================================
        st.markdown("## 🎯 Final Prediction Results")
        
        # Main prediction box
        st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='color: white; margin-bottom: 10px;'>🎯 AI-ENHANCED PREDICTION</h2>
                <div class='prediction-number'>{result['prediction']}</div>
                <h3 style='color: white; margin-top: 10px;'>Expected Citations</h3>
                <p style='font-size: 18px; opacity: 0.9;'>
                    Base ML: {result['base_ml_prediction']} × Fuzzy: {result['fuzzy_multiplier']:.2f} = {result['prediction']}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            gauge_fig = create_gauge_chart(result['prediction'], max_value=max(300, result['prediction'] * 1.5))
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.markdown(f"""
                <div class='info-card' style='border-left: 5px solid {category_info["color"]};'>
                    <h3>{category_info["emoji"]} {category_info["category"]}</h3>
                    <p style='font-size: 16px; color: #666;'>{category_info["description"]}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Citation trajectory
            years = [1, 2, 3, 4, 5]
            citations = [
                int(result['prediction'] * 0.3),
                int(result['prediction'] * 0.5),
                int(result['prediction'] * 0.75),
                int(result['prediction'] * 0.9),
                result['prediction']
            ]
            
            fig_trajectory = go.Figure()
            fig_trajectory.add_trace(go.Scatter(
                x=years,
                y=citations,
                mode='lines+markers',
                line=dict(color='#667eea', width=4),
                marker=dict(size=12, color='#764ba2'),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig_trajectory.update_layout(
                title="Predicted Citation Trajectory",
                xaxis_title="Years After Publication",
                yaxis_title="Cumulative Citations",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trajectory, use_container_width=True)
        
        st.markdown("---")
        
        # ========================================
        # STATE SPACE SEARCH INFO
        # ========================================
        st.markdown("## 🎯 Step 4: State Space Feature Selection")
        
        st.markdown("""
            <div class='state-box'>
                <h3>🔍 Hill Climbing Search (Training Phase)</h3>
                <p>State space search was used during model training to optimize feature selection:</p>
                <ul>
                    <li><b>Initial State:</b> All 409+ features</li>
                    <li><b>Search Algorithm:</b> Hill Climbing (local search)</li>
                    <li><b>Evaluation:</b> R² score on validation set</li>
                    <li><b>Optimization:</b> Found optimal feature subset</li>
                    <li><b>Result:</b> Improved accuracy & reduced overfitting</li>
                </ul>
                <p><i>Note: This optimization runs during training, not prediction.</i></p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🧮 How State Space Search Works"):
            st.markdown("""
            ### State Space Search for Feature Selection
            
            **Problem Formulation:**
            - **State:** A subset of features (represented as a set of indices)
            - **Initial State:** All available features or a random subset
            - **Goal:** Find feature subset that maximizes model performance
            - **Actions:** Add or remove one feature
            - **Search Strategy:** Hill Climbing
            
            **Hill Climbing Algorithm:**
            ```
            1. Start with initial feature set
            2. Evaluate model performance (R² score)
            3. Generate neighbors (add/remove 1 feature)
            4. Evaluate all neighbors
            5. Move to best neighbor if better than current
            6. Repeat until no improvement found
            ```
            
            **Benefits:**
            - Reduces dimensionality
            - Prevents overfitting
            - Improves interpretability
            - Faster predictions
            - Better generalization
            
            **Search Space Size:** 2^409 possible feature combinations!
            """)
        
        st.markdown("---")
        
        # ========================================
        # DETAILED ANALYSIS TABS
        # ========================================
        st.markdown("## 📊 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Citation Breakdown", 
            "💡 Recommendations", 
            "📋 Paper Details",
            "🤖 AI Techniques Summary"
        ])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class='info-card' style='text-align: center;'>
                        <h3>First Year</h3>
                        <p style='font-size: 32px; font-weight: bold; color: #667eea;'>
                            ~{int(result['prediction'] * 0.3)}
                        </p>
                        <p style='color: #666;'>Expected citations</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='info-card' style='text-align: center;'>
                        <h3>Years 2-3</h3>
                        <p style='font-size: 32px; font-weight: bold; color: #764ba2;'>
                            ~{int(result['prediction'] * 0.5)}
                        </p>
                        <p style='color: #666;'>Peak citations</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class='info-card' style='text-align: center;'>
                        <h3>After Year 5</h3>
                        <p style='font-size: 32px; font-weight: bold; color: #38ef7d;'>
                            ~{int(result['prediction'] * 0.2)}
                        </p>
                        <p style='color: #666;'>Steady citations</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            recommendations = []
            
            if quartile in ['Q3', 'Q4']:
                recommendations.append("🎯 Target higher-ranked journals (Q1/Q2) for greater visibility")
            
            if sjr < 1.0:
                recommendations.append("📊 Consider journals with higher SJR scores")
            
            if h_index < 30:
                recommendations.append("🤝 Collaborate with established researchers")
            
            if page_count < 8:
                recommendations.append("📝 Consider expanding your analysis for more comprehensive coverage")
            
            if result['citation_potential'] < 0.5:
                recommendations.append("🧠 Fuzzy logic suggests improving paper quality metrics")
            
            if not result['csp_valid']:
                recommendations.append("🔍 Address CSP constraint violations for better validation")
            
            if not recommendations:
                st.markdown("""
                    <div class='success-box'>
                        <h3>✅ Excellent Paper Profile!</h3>
                        <p>Your paper has strong indicators for high citation counts. Keep up the great work!</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                st.markdown("### 💡 AI-Powered Suggestions:")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("### 📄 Submitted Paper Information")
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Abstract:** {abstract[:300]}...")
            st.markdown(f"**Author Keywords:** {author_kw}")
            st.markdown(f"**Index Keywords:** {index_kw}")
            st.markdown(f"**Publisher:** {publisher}")
            st.markdown(f"**Country:** {country}")
            st.markdown(f"**Document Type:** {doc_type}")
            st.markdown(f"**Language:** {language}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 🤖 AI Techniques Applied")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class='info-card'>
                        <h4>1️⃣ Constraint Satisfaction Problem (CSP)</h4>
                        <ul>
                            <li><b>Purpose:</b> Validate inputs</li>
                            <li><b>Variables:</b> Year, SJR, H-Index, etc.</li>
                            <li><b>Constraints:</b> 6 logical rules</li>
                            <li><b>Result:</b> """ + ("✅ Valid" if result['csp_valid'] else "⚠️ Violations") + """</li>
                        </ul>
                    </div>
                    
                    <div class='info-card'>
                        <h4>2️⃣ Fuzzy Logic System</h4>
                        <ul>
                            <li><b>Membership Functions:</b> Triangular & Trapezoidal</li>
                            <li><b>Fuzzy Sets:</b> Low/Medium/High</li>
                            <li><b>Rules:</b> 6 IF-THEN statements</li>
                            <li><b>Output:</b> Citation potential: """ + f"{result['citation_potential']:.3f}" + """</li>
                            <li><b>Multiplier:</b> """ + f"{result['fuzzy_multiplier']:.3f}x applied" + """</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='info-card'>
                        <h4>3️⃣ XGBoost Machine Learning</h4>
                        <ul>
                            <li><b>Algorithm:</b> Gradient Boosting</li>
                            <li><b>Estimators:</b> 600 trees</li>
                            <li><b>Features:</b> """ + str(result.get('features_used', 'All')) + """ selected</li>
                            <li><b>Accuracy:</b> R² ~0.94</li>
                            <li><b>Training:</b> 876 papers</li>
                        </ul>
                    </div>
                    
                    <div class='info-card'>
                        <h4>4️⃣ State Space Search</h4>
                        <ul>
                            <li><b>Algorithm:</b> Hill Climbing</li>
                            <li><b>Purpose:</b> Feature selection</li>
                            <li><b>Search Space:</b> 2^420+ combinations</li>
                            <li><b>Phase:</b> Training optimization</li>
                            <li><b>Result:</b> Optimal feature subset found</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='success-box'>
                    <h3>🎓 Integration Strategy</h3>
                    <p><b>Step 1:</b> CSP validates inputs and ensures consistency</p>
                    <p><b>Step 2:</b> Fuzzy Logic reasons about quality with uncertainty</p>
                    <p><b>Step 3:</b> XGBoost makes data-driven prediction</p>
                    <p><b>Step 4:</b> Fuzzy multiplier adjusts ML output</p>
                    <p><b>Result:</b> Hybrid AI system combining symbolic + statistical reasoning!</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class='footer'>
            <p><b>AI-Enhanced Citation Predictor v2.0</b></p>
            <p>🤖 XGBoost ML | 🧠 Fuzzy Logic | 🔍 CSP | 🎯 State Space Search</p>
            <p>Model Accuracy: R² ~0.94 | Trained on 876 DeepFake Research Papers</p>
            <p style='margin-top: 10px;'>
                Made with Streamlit | © 2025
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
