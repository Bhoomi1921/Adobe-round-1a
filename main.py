"""
Enhanced PDF Document Structure Analyzer
A comprehensive tool for extracting hierarchical document structure from PDF files
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Custom imports
from document_parser import DocumentParser
from semantic_analyzer import SemanticAnalyzer
from hierarchy_builder import HierarchyBuilder
from content_classifier import ContentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Data class for document sections with enhanced metadata"""
    text: str
    level: str
    confidence: float
    page_number: int
    position: Dict[str, float]
    font_properties: Dict[str, any]
    semantic_score: float
    section_type: str
    word_count: int
    
class PDFDocumentAnalyzer:
    """
    Advanced PDF Document Structure Analyzer with enhanced features
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the analyzer with configuration"""
        self.config = self._load_configuration(config_path)
        self.input_directory = Path(self.config.get('input_dir', 'documents'))
        self.output_directory = Path(self.config.get('output_dir', 'results'))
        
        # Initialize components
        self.document_parser = DocumentParser()
        self.semantic_analyzer = SemanticAnalyzer()
        self.hierarchy_builder = HierarchyBuilder()
        self.content_classifier = ContentClassifier()
        
        # Create directories
        self._setup_directories()
        
    def _load_configuration(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'input_dir': 'documents',
            'output_dir': 'results',
            'max_heading_words': 20,
            'min_text_length': 4,
            'confidence_threshold': 0.7,
            'semantic_similarity_threshold': 0.8,
            'enable_advanced_features': True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                
        return default_config
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.input_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
    async def analyze_document(self, pdf_path: Path) -> Dict:
        """
        Analyze a single PDF document and extract its structure
        """
        logger.info(f"Starting analysis of: {pdf_path.name}")
        
        try:
            # Step 1: Parse document content
            document_elements = await self.document_parser.extract_elements(pdf_path)
            if not document_elements:
                return {"error": "No extractable content found", "file": pdf_path.name}
            
            # Debug: Check page numbers after parsing
            page_nums_after_parsing = [elem.page_num for elem in document_elements]
            logger.info(f"After parsing - Page range: {min(page_nums_after_parsing)} to {max(page_nums_after_parsing)}")
            
            # Step 2: Perform semantic analysis
            semantic_vectors = await self.semantic_analyzer.generate_embeddings(document_elements)
            
            # Step 3: Classify content types
            classified_elements = await self.content_classifier.classify_elements(
                document_elements, semantic_vectors
            )
            
            # Debug: Check page numbers after classification
            page_nums_after_classification = [elem.page_num for elem in classified_elements]
            logger.info(f"After classification - Page range: {min(page_nums_after_classification)} to {max(page_nums_after_classification)}")
            
            # Step 4: Build document hierarchy
            document_structure = await self.hierarchy_builder.build_hierarchy(
                classified_elements, semantic_vectors
            )

            # Convert hierarchy builder output to DocumentSection objects
            document_sections = []
            for item in document_structure:
                if isinstance(item, dict):
                    # FIXED: Use correct page number field mapping
                    page_number = item.get('page', 1)  # Get 'page' field, default to 1 if missing
                    
                    section = DocumentSection(
                        text=item.get('text', ''),
                        level=item.get('level', ''),
                        confidence=item.get('confidence', 0.0),
                        page_number=page_number,  # CRITICAL: Ensure correct page number assignment
                        position=item.get('position', {}),
                        font_properties=item.get('font_properties', {}),
                        semantic_score=item.get('semantic_score', 0.0),
                        section_type=item.get('section_type', ''),
                        word_count=item.get('word_count', 0)
                    )
                    document_sections.append(section)
                elif isinstance(item, DocumentSection):
                    document_sections.append(item)
            
            # Debug: Check page numbers in final sections
            if document_sections:
                section_page_nums = [section.page_number for section in document_sections]
                logger.info(f"Final sections - Page range: {min(section_page_nums)} to {max(section_page_nums)}")
            
            # Step 5: Extract title and create final structure
            document_title = self._extract_document_title(document_sections)
            
            # Step 6: Generate comprehensive results
            analysis_result = self._compile_results(
                document_title, document_sections, pdf_path
            )
            
            logger.info(f"Successfully analyzed: {pdf_path.name}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed for {pdf_path.name}: {str(e)}")
            return {"error": str(e), "file": pdf_path.name}
    
    def _extract_document_title(self, structure: List[DocumentSection]) -> str:
        """
        Advanced title extraction using multiple heuristics
        """
        title_candidates = []
        
        for section in structure:
            # Title criteria: high confidence, early page, appropriate length
            if (section.page_number <= 2 and 
                section.confidence > 0.8 and
                5 <= len(section.text) <= 100 and
                section.semantic_score > 0.7):
                
                title_score = self._calculate_title_score(section)
                title_candidates.append((section.text, title_score))
        
        if title_candidates:
            # Sort by score and return best candidate
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            return title_candidates[0][0]
        
        return "Document Analysis Result"
    
    def _calculate_title_score(self, section: DocumentSection) -> float:
        """Calculate title likelihood score"""
        score = 0.0
        
        # Page position bonus (earlier = better)
        if section.page_number == 1:
            score += 3.0
        elif section.page_number == 2:
            score += 1.5
            
        # Font size bonus
        font_size_ratio = section.font_properties.get('size_ratio', 1.0)
        score += min(font_size_ratio * 2, 4.0)
        
        # Position bonus (centered or left-aligned)
        x_position = section.position.get('x', 0)
        if 100 <= x_position <= 400:
            score += 2.0
        elif x_position < 100:
            score += 1.0
            
        # Text characteristics
        text_length = len(section.text)
        if 15 <= text_length <= 60:
            score += 2.0
        elif 8 <= text_length <= 80:
            score += 1.0
            
        # Semantic relevance
        score += section.semantic_score * 2
        
        return score
    
    def _compile_results(self, title: str, structure: List[DocumentSection], 
                        pdf_path: Path) -> Dict:
        """Compile comprehensive analysis results"""
        
        # Convert sections to serializable format
        sections_data = []
        for section in structure:
            sections_data.append({
                'hierarchy_level': section.level,
                'content': section.text,
                'page': section.page_number,  # CRITICAL: Use the correct page number
                'confidence_score': round(section.confidence, 3),
                'coordinates': section.position,
                'typography': section.font_properties,
                'semantic_relevance': round(section.semantic_score, 3),
                'classification': section.section_type,
                'word_count': section.word_count
            })
        
        # Generate metadata
        metadata = self._generate_metadata(structure, pdf_path)
        
        # Create structured output
        result = {
            'document_title': title,
            'analysis_timestamp': datetime.now().isoformat(),
            'source_file': pdf_path.name,
            'document_outline': sections_data,
            'analytics': metadata,
            'processing_info': {
                'total_sections': len(structure),
                'confidence_distribution': self._get_confidence_distribution(structure),
                'hierarchy_depth': self._calculate_hierarchy_depth(structure)
            }
        }
        
        return result
    
    def _generate_metadata(self, structure: List[DocumentSection], 
                          pdf_path: Path) -> Dict:
        """Generate comprehensive document metadata"""
        if not structure:
            return {}
            
        pages_analyzed = max(s.page_number for s in structure) if structure else 1
        font_sizes = [s.font_properties.get('size', 0) for s in structure]
        
        return {
            'total_sections_found': len(structure),
            'pages_analyzed': pages_analyzed,  # This should now show correct page count
            'font_sizes_detected': sorted(list(set(font_sizes)), reverse=True),
            'average_confidence': sum(s.confidence for s in structure) / len(structure),
            'section_types': list(set(s.section_type for s in structure)),
            'total_word_count': sum(s.word_count for s in structure),
            'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 2)
        }
    
    def _get_confidence_distribution(self, structure: List[DocumentSection]) -> Dict:
        """Calculate confidence score distribution"""
        if not structure:
            return {}
            
        confidences = [s.confidence for s in structure]
        return {
            'high_confidence': len([c for c in confidences if c >= 0.8]),
            'medium_confidence': len([c for c in confidences if 0.5 <= c < 0.8]),
            'low_confidence': len([c for c in confidences if c < 0.5]),
            'average': sum(confidences) / len(confidences)
        }
    
    def _calculate_hierarchy_depth(self, structure: List[DocumentSection]) -> int:
        """Calculate the depth of the document hierarchy"""
        levels = set()
        for section in structure:
            if section.level.startswith('H'):
                try:
                    level_num = int(section.level[1:])
                    levels.add(level_num)
                except ValueError:
                    continue
        return max(levels) if levels else 0
    
    async def process_documents_batch(self) -> Dict:
        """Process all PDF files in the input directory"""
        pdf_files = list(self.input_directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_directory}")
            return {"message": "No PDF files to process"}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {}
        successful_count = 0
        
        for pdf_file in pdf_files:
            try:
                analysis_result = await self.analyze_document(pdf_file)
                results[pdf_file.name] = analysis_result
                
                if "error" not in analysis_result:
                    successful_count += 1
                    # Save individual result
                    output_file = self.output_directory / f"{pdf_file.stem}_analysis.json"
                    await self._save_result(analysis_result, output_file)
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                results[pdf_file.name] = {"error": str(e)}
        
        # Save batch summary
        batch_summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_files": len(pdf_files),
            "successful_analyses": successful_count,
            "failed_analyses": len(pdf_files) - successful_count,
            "results": results
        }
        
        summary_file = self.output_directory / "batch_analysis_summary.json"
        await self._save_result(batch_summary, summary_file)
        
        logger.info(f"Batch processing complete: {successful_count}/{len(pdf_files)} successful")
        return batch_summary
    
    async def _save_result(self, result: Dict, output_path: Path):
        """Save analysis result to JSON file with NumPy type handling"""
        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()  # Convert numpy types to native Python types
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        try:
            # Convert numpy types first
            result = convert_numpy_types(result)
        
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save result to {output_path}: {e}")

async def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = PDFDocumentAnalyzer()
        
        # Process all documents
        results = await analyzer.process_documents_batch()
        
        # Print summary
        print(f"\n📊 Processing Summary:")
        print(f"Total files: {results.get('total_files', 0)}")
        print(f"Successful: {results.get('successful_analyses', 0)}")
        print(f"Failed: {results.get('failed_analyses', 0)}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Main execution failed")

if __name__ == "__main__":
    asyncio.run(main())

# """
# Enhanced PDF Document Structure Analyzer
# A comprehensive tool for extracting hierarchical document structure from PDF files
# """

# import os
# import json
# import asyncio
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# from dataclasses import dataclass, asdict
# from datetime import datetime
# import logging

# # Custom imports
# from document_parser import DocumentParser
# from semantic_analyzer import SemanticAnalyzer
# from hierarchy_builder import HierarchyBuilder
# from content_classifier import ContentClassifier

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# @dataclass
# class DocumentSection:
#     """Data class for document sections with enhanced metadata"""
#     text: str
#     level: str
#     confidence: float
#     page_number: int
#     position: Dict[str, float]
#     font_properties: Dict[str, any]
#     semantic_score: float
#     section_type: str
#     word_count: int
    
# class PDFDocumentAnalyzer:
#     """
#     Advanced PDF Document Structure Analyzer with enhanced features
#     """
    
#     def __init__(self, config_path: Optional[str] = None):
#         """Initialize the analyzer with configuration"""
#         self.config = self._load_configuration(config_path)
#         self.input_directory = Path(self.config.get('input_dir', 'documents'))
#         self.output_directory = Path(self.config.get('output_dir', 'results'))
        
#         # Initialize components
#         self.document_parser = DocumentParser()
#         self.semantic_analyzer = SemanticAnalyzer()
#         self.hierarchy_builder = HierarchyBuilder()
#         self.content_classifier = ContentClassifier()
        
#         # Create directories
#         self._setup_directories()
        
#     def _load_configuration(self, config_path: Optional[str]) -> Dict:
#         """Load configuration from file or use defaults"""
#         default_config = {
#             'input_dir': 'documents',
#             'output_dir': 'results',
#             'max_heading_words': 20,
#             'min_text_length': 4,
#             'confidence_threshold': 0.7,
#             'semantic_similarity_threshold': 0.8,
#             'enable_advanced_features': True
#         }
        
#         if config_path and Path(config_path).exists():
#             try:
#                 with open(config_path, 'r', encoding='utf-8') as f:
#                     user_config = json.load(f)
#                 default_config.update(user_config)
#             except Exception as e:
#                 logger.warning(f"Failed to load config: {e}. Using defaults.")
                
#         return default_config
    
#     def _setup_directories(self):
#         """Create necessary directories"""
#         self.input_directory.mkdir(parents=True, exist_ok=True)
#         self.output_directory.mkdir(parents=True, exist_ok=True)
        
#     async def analyze_document(self, pdf_path: Path) -> Dict:
#         """
#         Analyze a single PDF document and extract its structure
#         """
#         logger.info(f"Starting analysis of: {pdf_path.name}")
        
#         try:
#             # Step 1: Parse document content
#             document_elements = await self.document_parser.extract_elements(pdf_path)
#             if not document_elements:
#                 return {"error": "No extractable content found", "file": pdf_path.name}
            
#             # Step 2: Perform semantic analysis
#             semantic_vectors = await self.semantic_analyzer.generate_embeddings(document_elements)
            
#             # Step 3: Classify content types
#             classified_elements = await self.content_classifier.classify_elements(
#                 document_elements, semantic_vectors
#             )
            
#             # Step 4: Build document hierarchy
#             document_structure = await self.hierarchy_builder.build_hierarchy(
#                 classified_elements, semantic_vectors
#             )

#             # Convert hierarchy builder output to DocumentSection objects
#             document_sections = []
#             for item in document_structure:
#                 if isinstance(item, dict):
#                     section = DocumentSection(
#                         text=item.get('text', ''),
#                         level=item.get('level', ''),
#                         confidence=item.get('confidence', 0.0),
#                         page_number=item.get('page_number', 0),
#                         position=item.get('position', {}),
#                         font_properties=item.get('font_properties', {}),
#                         semantic_score=item.get('semantic_score', 0.0),
#                         section_type=item.get('section_type', ''),
#                         word_count=item.get('word_count', 0)
#                     )
#                     document_sections.append(section)
#                 elif isinstance(item, DocumentSection):
#                     document_sections.append(item)
            
#             # Step 5: Extract title and create final structure
#             document_title = self._extract_document_title(document_sections)
            
#             # Step 6: Generate comprehensive results
#             analysis_result = self._compile_results(
#                 document_title, document_sections, pdf_path
#             )
            
#             logger.info(f"Successfully analyzed: {pdf_path.name}")
#             return analysis_result
            
#         except Exception as e:
#             logger.error(f"Analysis failed for {pdf_path.name}: {str(e)}")
#             return {"error": str(e), "file": pdf_path.name}
    
#     def _extract_document_title(self, structure: List[DocumentSection]) -> str:
#         """
#         Advanced title extraction using multiple heuristics
#         """
#         title_candidates = []
        
#         for section in structure:
#             # Title criteria: high confidence, early page, appropriate length
#             if (section.page_number <= 2 and 
#                 section.confidence > 0.8 and
#                 5 <= len(section.text) <= 100 and
#                 section.semantic_score > 0.7):
                
#                 title_score = self._calculate_title_score(section)
#                 title_candidates.append((section.text, title_score))
        
#         if title_candidates:
#             # Sort by score and return best candidate
#             title_candidates.sort(key=lambda x: x[1], reverse=True)
#             return title_candidates[0][0]
        
#         return "Document Analysis Result"
    
#     def _calculate_title_score(self, section: DocumentSection) -> float:
#         """Calculate title likelihood score"""
#         score = 0.0
        
#         # Page position bonus (earlier = better)
#         if section.page_number == 1:
#             score += 3.0
#         elif section.page_number == 2:
#             score += 1.5
            
#         # Font size bonus
#         font_size_ratio = section.font_properties.get('size_ratio', 1.0)
#         score += min(font_size_ratio * 2, 4.0)
        
#         # Position bonus (centered or left-aligned)
#         x_position = section.position.get('x', 0)
#         if 100 <= x_position <= 400:
#             score += 2.0
#         elif x_position < 100:
#             score += 1.0
            
#         # Text characteristics
#         text_length = len(section.text)
#         if 15 <= text_length <= 60:
#             score += 2.0
#         elif 8 <= text_length <= 80:
#             score += 1.0
            
#         # Semantic relevance
#         score += section.semantic_score * 2
        
#         return score
    
#     def _compile_results(self, title: str, structure: List[DocumentSection], 
#                         pdf_path: Path) -> Dict:
#         """Compile comprehensive analysis results"""
        
#         # Convert sections to serializable format
#         sections_data = []
#         for section in structure:
#             sections_data.append({
#                 'hierarchy_level': section.level,
#                 'content': section.text,
#                 'page': section.page_number,
#                 'confidence_score': round(section.confidence, 3),
#                 'coordinates': section.position,
#                 'typography': section.font_properties,
#                 'semantic_relevance': round(section.semantic_score, 3),
#                 'classification': section.section_type,
#                 'word_count': section.word_count
#             })
        
#         # Generate metadata
#         metadata = self._generate_metadata(structure, pdf_path)
        
#         # Create structured output
#         result = {
#             'document_title': title,
#             'analysis_timestamp': datetime.now().isoformat(),
#             'source_file': pdf_path.name,
#             'document_outline': sections_data,
#             'analytics': metadata,
#             'processing_info': {
#                 'total_sections': len(structure),
#                 'confidence_distribution': self._get_confidence_distribution(structure),
#                 'hierarchy_depth': self._calculate_hierarchy_depth(structure)
#             }
#         }
        
#         return result
    
#     def _generate_metadata(self, structure: List[DocumentSection], 
#                           pdf_path: Path) -> Dict:
#         """Generate comprehensive document metadata"""
#         if not structure:
#             return {}
            
#         pages_analyzed = max(s.page_number for s in structure)
#         font_sizes = [s.font_properties.get('size', 0) for s in structure]
        
#         return {
#             'total_sections_found': len(structure),
#             'pages_analyzed': pages_analyzed,
#             'font_sizes_detected': sorted(list(set(font_sizes)), reverse=True),
#             'average_confidence': sum(s.confidence for s in structure) / len(structure),
#             'section_types': list(set(s.section_type for s in structure)),
#             'total_word_count': sum(s.word_count for s in structure),
#             'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 2)
#         }
    
#     def _get_confidence_distribution(self, structure: List[DocumentSection]) -> Dict:
#         """Calculate confidence score distribution"""
#         if not structure:
#             return {}
            
#         confidences = [s.confidence for s in structure]
#         return {
#             'high_confidence': len([c for c in confidences if c >= 0.8]),
#             'medium_confidence': len([c for c in confidences if 0.5 <= c < 0.8]),
#             'low_confidence': len([c for c in confidences if c < 0.5]),
#             'average': sum(confidences) / len(confidences)
#         }
    
#     def _calculate_hierarchy_depth(self, structure: List[DocumentSection]) -> int:
#         """Calculate the depth of the document hierarchy"""
#         levels = set()
#         for section in structure:
#             if section.level.startswith('H'):
#                 try:
#                     level_num = int(section.level[1:])
#                     levels.add(level_num)
#                 except ValueError:
#                     continue
#         return max(levels) if levels else 0
    
#     async def process_documents_batch(self) -> Dict:
#         """Process all PDF files in the input directory"""
#         pdf_files = list(self.input_directory.glob("*.pdf"))
        
#         if not pdf_files:
#             logger.warning(f"No PDF files found in {self.input_directory}")
#             return {"message": "No PDF files to process"}
        
#         logger.info(f"Found {len(pdf_files)} PDF files to process")
        
#         results = {}
#         successful_count = 0
        
#         for pdf_file in pdf_files:
#             try:
#                 analysis_result = await self.analyze_document(pdf_file)
#                 results[pdf_file.name] = analysis_result
                
#                 if "error" not in analysis_result:
#                     successful_count += 1
#                     # Save individual result
#                     output_file = self.output_directory / f"{pdf_file.stem}_analysis.json"
#                     await self._save_result(analysis_result, output_file)
                    
#             except Exception as e:
#                 logger.error(f"Failed to process {pdf_file.name}: {e}")
#                 results[pdf_file.name] = {"error": str(e)}
        
#         # Save batch summary
#         batch_summary = {
#             "processing_timestamp": datetime.now().isoformat(),
#             "total_files": len(pdf_files),
#             "successful_analyses": successful_count,
#             "failed_analyses": len(pdf_files) - successful_count,
#             "results": results
#         }
        
#         summary_file = self.output_directory / "batch_analysis_summary.json"
#         await self._save_result(batch_summary, summary_file)
        
#         logger.info(f"Batch processing complete: {successful_count}/{len(pdf_files)} successful")
#         return batch_summary
    
#     async def _save_result(self, result: Dict, output_path: Path):
#         """Save analysis result to JSON file with NumPy type handling"""
#         def convert_numpy_types(obj):
#             import numpy as np
#             if isinstance(obj, np.generic):
#                 return obj.item()  # Convert numpy types to native Python types
#             elif isinstance(obj, dict):
#                 return {k: convert_numpy_types(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [convert_numpy_types(v) for v in obj]
#             return obj

#         try:
#         # Convert numpy types first
#             result = convert_numpy_types(result)
        
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 json.dump(result, f, indent=2, ensure_ascii=False)
#             logger.info(f"Result saved to: {output_path}")
#         except Exception as e:
#             logger.error(f"Failed to save result to {output_path}: {e}")

# async def main():
#     """Main execution function"""
#     try:
#         # Initialize analyzer
#         analyzer = PDFDocumentAnalyzer()
        
#         # Process all documents
#         results = await analyzer.process_documents_batch()
        
#         # Print summary
#         print(f"\n📊 Processing Summary:")
#         print(f"Total files: {results.get('total_files', 0)}")
#         print(f"Successful: {results.get('successful_analyses', 0)}")
#         print(f"Failed: {results.get('failed_analyses', 0)}")
        
#     except KeyboardInterrupt:
#         print("\n⚠️ Process interrupted by user")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#         logger.exception("Main execution failed")

# if __name__ == "__main__":
#     asyncio.run(main())
