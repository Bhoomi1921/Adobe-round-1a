"""
Enhanced PDF Structure Analyzer - Interactive Streamlit Application
"""
import numpy as np
import json


import streamlit as st
import asyncio
import tempfile
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import logging

# Import custom modules
from main import PDFDocumentAnalyzer
from document_parser import DocumentElement
# json_data = json.dumps(result, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, (np.generic,)) else str(o))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF Structure Analyzer Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .hierarchy-item {
        margin-left: 1rem;
        padding: 0.5rem;
        border-left: 2px solid #ddd;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitPDFAnalyzer:
    """Streamlit interface for PDF structure analysis"""
    
    def __init__(self):
        self.analyzer = PDFDocumentAnalyzer()
        self.session_key = "pdf_analysis_results"
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {}
        if "processing_complete" not in st.session_state:
            st.session_state.processing_complete = False
        if "selected_files" not in st.session_state:
            st.session_state.selected_files = []
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">📚 PDF Structure Analyzer Pro</h1>
            <p style="color: #f0f0f0; margin: 0.5rem 0 0 0;">
                Advanced document structure extraction with AI-powered analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with settings and information"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Analysis settings
            st.subheader("Analysis Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.0, 1.0, 0.7, 0.1,
                help="Minimum confidence for heading classification"
            )
            
            max_hierarchy_levels = st.slider(
                "Max Hierarchy Levels", 
                2, 8, 6, 1,
                help="Maximum number of heading levels to detect"
            )
            
            enable_semantic_analysis = st.checkbox(
                "Enable Semantic Analysis", 
                value=True,
                help="Use AI embeddings for better structure detection"
            )
            
            # Processing options
            st.subheader("Processing Options")
            batch_size = st.number_input(
                "Batch Size", 
                min_value=1, max_value=20, value=5,
                help="Number of files to process simultaneously"
            )
            
            save_individual_files = st.checkbox(
                "Save Individual Results", 
                value=True,
                help="Save separate JSON file for each processed PDF"
            )
            
            # Information section
            st.header("ℹ️ Features")
            st.markdown("""
            **Advanced Capabilities:**
            - 🎯 Smart heading detection
            - 🧠 AI-powered semantic analysis
            - 📊 Hierarchical structure building
            - 📈 Confidence scoring
            - 🔍 Multi-modal classification
            - 📋 Comprehensive metadata
            
            **Supported Formats:**
            - PDF documents
            - Multi-page documents
            - Various font styles
            """)
            
            return {
                'confidence_threshold': confidence_threshold,
                'max_hierarchy_levels': max_hierarchy_levels,
                'enable_semantic_analysis': enable_semantic_analysis,
                'batch_size': batch_size,
                'save_individual_files': save_individual_files
            }
    
    def render_file_upload(self, config):
        """Render file upload interface"""
        st.header("📤 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Select PDF files for analysis",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to analyze their structure"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            # Display file information
            with st.expander("📋 File Details", expanded=True):
                file_data = []
                for file in uploaded_files:
                    file_size_mb = len(file.getvalue()) / (1024 * 1024)
                    file_data.append({
                        'Filename': file.name,
                        'Size (MB)': f"{file_size_mb:.2f}",
                        'Type': 'PDF Document'
                    })
                
                df = pd.DataFrame(file_data)
                st.dataframe(df, use_container_width=True)
            
            # Processing controls
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    self.process_files(uploaded_files, config)
            
            with col2:
                if st.session_state.get(self.session_key):
                    if st.button("🔄 Clear Results", use_container_width=True):
                        st.session_state[self.session_key] = {}
                        st.session_state.processing_complete = False
                        st.rerun()
            
            with col3:
                st.metric("Queue", len(uploaded_files))
    
    def process_files(self, uploaded_files, config):
        """Process uploaded files with progress tracking"""
        total_files = len(uploaded_files)
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        results = {}
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing: {uploaded_file.name} ({i+1}/{total_files})")
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = Path(tmp_file.name)
                
                # Process the file
                result = asyncio.run(self.analyzer.analyze_document(tmp_file_path))
                results[uploaded_file.name] = result
                
                # Display immediate feedback
                with results_container:
                    if "error" in result:
                        st.error(f"❌ {uploaded_file.name}: {result['error']}")
                    else:
                        st.success(f"✅ {uploaded_file.name}: Found {result['processing_info']['total_sections']} sections")
                
                # Clean up
                tmp_file_path.unlink()
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                results[uploaded_file.name] = {"error": str(e)}
                st.error(f"❌ {uploaded_file.name}: Processing failed")
        
        # Store results and mark as complete
        st.session_state[self.session_key] = results
        st.session_state.processing_complete = True
        
        # Final status update
        successful = sum(1 for r in results.values() if "error" not in r)
        failed = len(results) - successful
        
        status_text.text(f"✅ Processing complete: {successful} successful, {failed} failed")
        progress_bar.progress(1.0)
        
        # Trigger rerun to show results
        st.rerun()
    
    def render_results(self):
        """Render analysis results"""
        if not st.session_state.get(self.session_key):
            st.info("👆 Upload PDF files and start analysis to see results here")
            return
        
        results = st.session_state[self.session_key]
        
        st.header("📊 Analysis Results")
        
        # Summary statistics
        self.render_summary_statistics(results)
        
        # Individual file results
        self.render_individual_results(results)
        
        # Batch operations
        self.render_batch_operations(results)
    
    def render_summary_statistics(self, results):
        """Render summary statistics"""
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        failed_results = {k: v for k, v in results.items() if "error" in v}
        
        st.subheader("📈 Processing Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", len(results))
        
        with col2:
            st.metric("Successful", len(successful_results))
        
        with col3:
            st.metric("Failed", len(failed_results))
        
        with col4:
            if successful_results:
                total_sections = sum(r['processing_info']['total_sections'] 
                                   for r in successful_results.values())
                st.metric("Total Sections", total_sections)
        
        # Success rate chart
        if len(results) > 1:
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Successful', 'Failed'],
                    values=[len(successful_results), len(failed_results)],
                    hole=0.3,
                    marker_colors=['#28a745', '#dc3545']
                )
            ])
            fig.update_layout(
                title="Processing Success Rate",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_individual_results(self, results):
        """Render individual file results"""
        st.subheader("📄 Individual Results")
        
        # File selector
        successful_files = [k for k, v in results.items() if "error" not in v]
        
        if not successful_files:
            st.warning("No successful analyses to display")
            return
        
        selected_file = st.selectbox(
            "Select file to view detailed results:",
            successful_files,
            key="file_selector"
        )
        
        if selected_file:
            result = results[selected_file]
            self.render_file_analysis(selected_file, result)
    
    def render_file_analysis(self, filename, result):
        """Render detailed analysis for a single file"""
        st.markdown(f"### 📖 {filename}")
        
        # Title and metadata
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Document Title:** {result['document_title']}")
            st.markdown(f"**Analysis Date:** {result['analysis_timestamp'][:19]}")
        
        with col2:
            analytics = result['analytics']
            st.metric("Pages", analytics.get('pages_analyzed', 0))
            st.metric("Sections", analytics.get('total_sections_found', 0))
        
        # Detailed analytics
        with st.expander("📊 Detailed Analytics", expanded=False):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**Document Statistics:**")
                st.write(f"- Average confidence: {analytics.get('average_confidence', 0):.3f}")
                st.write(f"- Total word count: {analytics.get('total_word_count', 0):,}")
                st.write(f"- File size: {analytics.get('file_size_mb', 0)} MB")
            
            with col_b:
                st.write("**Section Types:**")
                section_types = analytics.get('section_types', [])
                for stype in section_types:
                    st.write(f"- {stype.title()}")
        
        # Document outline
        st.subheader("🗂️ Document Structure")
        
        outline = result['document_outline']
        
        if outline:
            # Interactive outline view
            self.render_interactive_outline(outline)
            
            # Tabular view
            with st.expander("📋 Tabular View"):
                outline_df = pd.DataFrame(outline)
                st.dataframe(outline_df, use_container_width=True)
            
            # Confidence distribution
            self.render_confidence_distribution(outline)
        
        # Download options
        st.subheader("💾 Download Options")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            # json_data = json.dumps(result, indent=2, ensure_ascii=False)
            json_data = json.dumps(result, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, (np.generic,)) else str(o))
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"{Path(filename).stem}_analysis.json",
                mime="application/json"
            )
        
        with col_d2:
            # Create structured text outline
            text_outline = self.create_text_outline(result)
            st.download_button(
                label="📝 Download Outline",
                data=text_outline,
                file_name=f"{Path(filename).stem}_outline.txt",
                mime="text/plain"
            )
    
    def render_interactive_outline(self, outline):
        """Render interactive hierarchical outline"""
        for item in outline:
            level = item['hierarchy_level']
            text = item['content']
            page = item['page']
            confidence = item['confidence_score']
            
            # Calculate indentation
            level_num = int(level[1:]) if level.startswith('H') else 1
            indent = "  " * (level_num - 1)
            
            # Choose emoji based on level
            emoji = "📘" if level_num == 1 else "📄" if level_num == 2 else "📝"
            
            # Color code by confidence
            if confidence >= 0.8:
                confidence_color = "🟢"
            elif confidence >= 0.6:
                confidence_color = "🟡"
            else:
                confidence_color = "🔴"
            
            st.markdown(
                f"{indent}{emoji} **{text}** "
                f"*({level}, Page {page}) {confidence_color}*"
            )
    
    def render_confidence_distribution(self, outline):
        """Render confidence score distribution chart"""
        if not outline:
            return
        
        confidences = [item['confidence_score'] for item in outline]
        
        fig = px.histogram(
            x=confidences,
            nbins=10,
            title="Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Count'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_text_outline(self, result):
        """Create text-based outline for download"""
        lines = []
        lines.append(f"Document: {result['document_title']}")
        lines.append(f"Analysis Date: {result['analysis_timestamp']}")
        lines.append("=" * 50)
        lines.append("")
        
        for item in result['document_outline']:
            level = item['hierarchy_level']
            text = item['content']
            page = item['page']
            
            level_num = int(level[1:]) if level.startswith('H') else 1
            indent = "  " * (level_num - 1)
            
            lines.append(f"{indent}{level}. {text} (Page {page})")
        
        return "\n".join(lines)
    
    def render_batch_operations(self, results):
        """Render batch operation options"""
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            return
        
        st.subheader("📦 Batch Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download all results as ZIP
            if st.button("📦 Download All Results (ZIP)", use_container_width=True):
                zip_data = self.create_results_zip(successful_results)
                st.download_button(
                    label="💾 Download ZIP Archive",
                    data=zip_data,
                    file_name=f"pdf_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        with col2:
            # Download batch summary
            if st.button("📊 Download Summary Report", use_container_width=True):
                summary_report = self.create_summary_report(results)
                st.download_button(
                    label="📄 Download Report",
                    data=summary_report,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    def create_results_zip(self, results):
        """Create ZIP archive of all results"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, result in results.items():
                json_data = json.dumps(result, indent=2, ensure_ascii=False)
                json_filename = f"{Path(filename).stem}_analysis.json"
                zip_file.writestr(json_filename, json_data)
                
                # Add text outline
                text_outline = self.create_text_outline(result)
                text_filename = f"{Path(filename).stem}_outline.txt"
                zip_file.writestr(text_filename, text_outline)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def create_summary_report(self, results):
        """Create comprehensive summary report"""
        successful = {k: v for k, v in results.items() if "error" not in v}
        failed = {k: v for k, v in results.items() if "error" in v}
        
        summary = {
            "report_generated": datetime.now().isoformat(),
            "processing_summary": {
                "total_files": len(results),
                "successful_analyses": len(successful),
                "failed_analyses": len(failed),
                "success_rate": len(successful) / len(results) if results else 0
            },
            "aggregate_statistics": {},
            "individual_results": results
        }
        
        if successful:
            total_sections = sum(r['processing_info']['total_sections'] for r in successful.values())
            total_pages = sum(r['analytics']['pages_analyzed'] for r in successful.values())
            avg_confidence = sum(r['analytics']['average_confidence'] for r in successful.values()) / len(successful)
            
            summary["aggregate_statistics"] = {
                "total_sections_found": total_sections,
                "total_pages_analyzed": total_pages,
                "average_confidence": avg_confidence,
                "files_processed": list(successful.keys())
            }
        
        return json.dumps(summary, indent=2, ensure_ascii=False)

def main():
    """Main Streamlit application"""
    app = StreamlitPDFAnalyzer()
    app.initialize_session_state()
    
    # Render UI components
    app.render_header()
    config = app.render_sidebar()
    
    # Main content area
    tab1, tab2 = st.tabs(["🔄 Process Documents", "📊 View Results"])
    
    with tab1:
        app.render_file_upload(config)
    
    with tab2:
        app.render_results()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**PDF Structure Analyzer Pro** | "
        "Built with Streamlit, PyMuPDF, and Advanced ML Models"
    )

if __name__ == "__main__":
    main()