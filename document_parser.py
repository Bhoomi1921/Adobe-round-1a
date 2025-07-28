"""
Document Parser Module - Extract and preprocess content from PDF documents
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentElement:
    """Represents a single text element from the document"""
    content: str
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_name: str
    font_size: float
    font_flags: int
    color: int
    is_bold: bool
    is_italic: bool
    line_height: float
    char_count: int

class DocumentParser:
    """Advanced PDF document parser with enhanced text extraction"""
    
    def __init__(self):
        self.content_filters = {
            'min_text_length': 3,
            'max_text_length': 200,
            'exclude_patterns': [
                r'^\d+$',  # Pure numbers
                r'^[^\w\s]+$',  # Pure symbols
                r'^\s*$',  # Whitespace only
                r'^(page|p\.)\s*\d+$',  # Page numbers
                r'^www\.',  # URLs
                r'@.*\.com$',  # Emails
            ]
        }
    
    async def extract_elements(self, pdf_path: Path) -> List[DocumentElement]:
        """
        Extract text elements from PDF with enhanced preprocessing
        """
        try:
            document = fitz.open(str(pdf_path))
            all_elements = []
            
            # FIXED: Ensure correct page numbering
            for page_index in range(len(document)):
                page = document[page_index]
                # Pass page_index + 1 to make it 1-based (human-readable page numbers)
                page_elements = await self._process_page(page, page_index + 1)
                all_elements.extend(page_elements)
                
                # Debug logging to verify page numbers
                if page_elements:
                    logger.debug(f"Processed page {page_index + 1} with {len(page_elements)} elements")
            
            document.close()
            
            # Post-process elements
            filtered_elements = self._apply_content_filters(all_elements)
            enhanced_elements = self._enhance_elements(filtered_elements)
            
            # Final verification of page numbers
            page_numbers = set(elem.page_num for elem in enhanced_elements)
            logger.info(f"Extracted {len(enhanced_elements)} valid elements from {pdf_path.name}")
            logger.info(f"Page numbers found: {sorted(page_numbers)}")
            
            return enhanced_elements
            
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {e}")
            return []
    
    async def _process_page(self, page, page_num: int) -> List[DocumentElement]:
        """Process a single page and extract text elements"""
        elements = []
        
        try:
            # Get text dictionary with detailed information
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        element = self._create_element_from_span(span, page_num)
                        if element and self._is_valid_element(element):
                            elements.append(element)
                            
        except Exception as e:
            logger.warning(f"Error processing page {page_num}: {e}")
        
        # Debug: Log the page number for each element created
        if elements:
            logger.debug(f"Created {len(elements)} elements for page {page_num}")
            # Verify all elements have correct page number
            for elem in elements:
                if elem.page_num != page_num:
                    logger.error(f"Page number mismatch! Expected {page_num}, got {elem.page_num}")
        
        return elements
    
    def _create_element_from_span(self, span: Dict, page_num: int) -> Optional[DocumentElement]:
        """Create DocumentElement from a text span"""
        try:
            text_content = span["text"].strip()
            if not text_content:
                return None
            
            # Extract font properties
            font_flags = span.get("flags", 0)
            is_bold = bool(font_flags & 2**4) or "bold" in span["font"].lower()
            is_italic = bool(font_flags & 2**1) or "italic" in span["font"].lower()
            
            element = DocumentElement(
                content=text_content,
                page_num=page_num,  # CRITICAL: Ensure page_num is correctly assigned
                bbox=tuple(span["bbox"]),
                font_name=span["font"],
                font_size=round(span["size"], 2),
                font_flags=font_flags,
                color=span.get("color", 0),
                is_bold=is_bold,
                is_italic=is_italic,
                line_height=self._calculate_line_height(span),
                char_count=len(text_content)
            )
            
            # Debug: Verify page number assignment
            logger.debug(f"Created element on page {page_num}: '{text_content[:30]}...'")
            
            return element
            
        except (KeyError, TypeError) as e:
            logger.debug(f"Error creating element from span: {e}")
            return None
    
    def _calculate_line_height(self, span: Dict) -> float:
        """Calculate line height from bounding box"""
        bbox = span["bbox"]
        return round(bbox[3] - bbox[1], 2)
    
    def _is_valid_element(self, element: DocumentElement) -> bool:
        """Check if element meets basic validity criteria"""
        # Length checks
        if (len(element.content) < self.content_filters['min_text_length'] or
            len(element.content) > self.content_filters['max_text_length']):
            return False
        
        # Pattern exclusions
        for pattern in self.content_filters['exclude_patterns']:
            if re.match(pattern, element.content.lower().strip()):
                return False
        
        # Font size check (avoid very small text like footnotes)
        if element.font_size < 6:
            return False
            
        return True
    
    def _apply_content_filters(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Apply advanced content filtering"""
        if not elements:
            return []
        
        # Remove duplicates based on content and page
        seen_content = set()
        unique_elements = []
        
        for element in elements:
            content_key = (element.content.lower().strip(), element.page_num)
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_elements.append(element)
        
        # Sort by page and position
        unique_elements.sort(key=lambda e: (e.page_num, e.bbox[1], e.bbox[0]))
        
        # Debug: Verify page numbers after filtering
        page_distribution = {}
        for elem in unique_elements:
            page_distribution[elem.page_num] = page_distribution.get(elem.page_num, 0) + 1
        
        logger.info(f"Filtered elements: {len(elements)} -> {len(unique_elements)}")
        logger.info(f"Page distribution: {dict(sorted(page_distribution.items()))}")
        
        return unique_elements
    
    def _enhance_elements(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Enhance elements with additional computed properties"""
        if not elements:
            return []
        
        # Calculate font size statistics for relative sizing
        font_sizes = [e.font_size for e in elements]
        avg_font_size = sum(font_sizes) / len(font_sizes)
        max_font_size = max(font_sizes)
        
        # Enhance each element
        for element in elements:
            # Add relative font size ratio
            element.font_size_ratio = element.font_size / avg_font_size
            element.max_font_ratio = element.font_size / max_font_size
            
            # Calculate position metrics
            element.x_position = element.bbox[0]
            element.y_position = element.bbox[1]
            element.width = element.bbox[2] - element.bbox[0]
            element.height = element.bbox[3] - element.bbox[1]
            
            # Text characteristics
            element.word_count = len(element.content.split())
            element.has_numbers = bool(re.search(r'\d', element.content))
            element.capitalization_ratio = sum(1 for c in element.content if c.isupper()) / len(element.content)
        
        # Final verification
        page_nums_final = [e.page_num for e in elements]
        logger.info(f"Enhanced elements page range: {min(page_nums_final)} to {max(page_nums_final)}")
        
        return elements
    
    def get_document_statistics(self, elements: List[DocumentElement]) -> Dict:
        """Generate statistical information about the document"""
        if not elements:
            return {}
        
        return {
            'total_elements': len(elements),
            'pages_covered': len(set(e.page_num for e in elements)),
            'font_families': list(set(e.font_name for e in elements)),
            'font_size_range': (min(e.font_size for e in elements), 
                              max(e.font_size for e in elements)),
            'average_font_size': sum(e.font_size for e in elements) / len(elements),
            'bold_elements': len([e for e in elements if e.is_bold]),
            'italic_elements': len([e for e in elements if e.is_italic]),
            'total_characters': sum(e.char_count for e in elements),
            'average_line_height': sum(e.line_height for e in elements) / len(elements)
        }

# """
# Document Parser Module - Extract and preprocess content from PDF documents
# """

# import fitz  # PyMuPDF
# import re
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# from dataclasses import dataclass
# import logging

# logger = logging.getLogger(__name__)

# @dataclass
# class DocumentElement:
#     """Represents a single text element from the document"""
#     content: str
#     page_num: int
#     bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
#     font_name: str
#     font_size: float
#     font_flags: int
#     color: int
#     is_bold: bool
#     is_italic: bool
#     line_height: float
#     char_count: int

# class DocumentParser:
#     """Advanced PDF document parser with enhanced text extraction"""
    
#     def __init__(self):
#         self.content_filters = {
#             'min_text_length': 3,
#             'max_text_length': 200,
#             'exclude_patterns': [
#                 r'^\d+$',  # Pure numbers
#                 r'^[^\w\s]+$',  # Pure symbols
#                 r'^\s*$',  # Whitespace only
#                 r'^(page|p\.)\s*\d+$',  # Page numbers
#                 r'^www\.',  # URLs
#                 r'@.*\.com$',  # Emails
#             ]
#         }
    
#     async def extract_elements(self, pdf_path: Path) -> List[DocumentElement]:
#         """
#         Extract text elements from PDF with enhanced preprocessing
#         """
#         try:
#             document = fitz.open(str(pdf_path))
#             all_elements = []
            
#             for page_index in range(len(document)):
#                 page = document[page_index]
#                 page_elements = await self._process_page(page, page_index + 1)
#                 all_elements.extend(page_elements)
            
#             document.close()
            
#             # Post-process elements
#             filtered_elements = self._apply_content_filters(all_elements)
#             enhanced_elements = self._enhance_elements(filtered_elements)
            
#             logger.info(f"Extracted {len(enhanced_elements)} valid elements from {pdf_path.name}")
#             return enhanced_elements
            
#         except Exception as e:
#             logger.error(f"Error extracting from {pdf_path}: {e}")
#             return []
    
#     async def _process_page(self, page, page_num: int) -> List[DocumentElement]:
#         """Process a single page and extract text elements"""
#         elements = []
        
#         try:
#             # Get text dictionary with detailed information
#             text_dict = page.get_text("dict")
            
#             for block in text_dict.get("blocks", []):
#                 if "lines" not in block:
#                     continue
                
#                 for line in block["lines"]:
#                     for span in line["spans"]:
#                         element = self._create_element_from_span(span, page_num)
#                         if element and self._is_valid_element(element):
#                             elements.append(element)
                            
#         except Exception as e:
#             logger.warning(f"Error processing page {page_num}: {e}")
        
#         return elements
    
#     def _create_element_from_span(self, span: Dict, page_num: int) -> Optional[DocumentElement]:
#         """Create DocumentElement from a text span"""
#         try:
#             text_content = span["text"].strip()
#             if not text_content:
#                 return None
            
#             # Extract font properties
#             font_flags = span.get("flags", 0)
#             is_bold = bool(font_flags & 2**4) or "bold" in span["font"].lower()
#             is_italic = bool(font_flags & 2**1) or "italic" in span["font"].lower()
            
#             return DocumentElement(
#                 content=text_content,
#                 page_num=page_num,
#                 bbox=tuple(span["bbox"]),
#                 font_name=span["font"],
#                 font_size=round(span["size"], 2),
#                 font_flags=font_flags,
#                 color=span.get("color", 0),
#                 is_bold=is_bold,
#                 is_italic=is_italic,
#                 line_height=self._calculate_line_height(span),
#                 char_count=len(text_content)
#             )
            
#         except (KeyError, TypeError) as e:
#             logger.debug(f"Error creating element from span: {e}")
#             return None
    
#     def _calculate_line_height(self, span: Dict) -> float:
#         """Calculate line height from bounding box"""
#         bbox = span["bbox"]
#         return round(bbox[3] - bbox[1], 2)
    
#     def _is_valid_element(self, element: DocumentElement) -> bool:
#         """Check if element meets basic validity criteria"""
#         # Length checks
#         if (len(element.content) < self.content_filters['min_text_length'] or
#             len(element.content) > self.content_filters['max_text_length']):
#             return False
        
#         # Pattern exclusions
#         for pattern in self.content_filters['exclude_patterns']:
#             if re.match(pattern, element.content.lower().strip()):
#                 return False
        
#         # Font size check (avoid very small text like footnotes)
#         if element.font_size < 6:
#             return False
            
#         return True
    
#     def _apply_content_filters(self, elements: List[DocumentElement]) -> List[DocumentElement]:
#         """Apply advanced content filtering"""
#         if not elements:
#             return []
        
#         # Remove duplicates based on content and page
#         seen_content = set()
#         unique_elements = []
        
#         for element in elements:
#             content_key = (element.content.lower().strip(), element.page_num)
#             if content_key not in seen_content:
#                 seen_content.add(content_key)
#                 unique_elements.append(element)
        
#         # Sort by page and position
#         unique_elements.sort(key=lambda e: (e.page_num, e.bbox[1], e.bbox[0]))
        
#         logger.info(f"Filtered elements: {len(elements)} -> {len(unique_elements)}")
#         return unique_elements
    
#     def _enhance_elements(self, elements: List[DocumentElement]) -> List[DocumentElement]:
#         """Enhance elements with additional computed properties"""
#         if not elements:
#             return []
        
#         # Calculate font size statistics for relative sizing
#         font_sizes = [e.font_size for e in elements]
#         avg_font_size = sum(font_sizes) / len(font_sizes)
#         max_font_size = max(font_sizes)
        
#         # Enhance each element
#         for element in elements:
#             # Add relative font size ratio
#             element.font_size_ratio = element.font_size / avg_font_size
#             element.max_font_ratio = element.font_size / max_font_size
            
#             # Calculate position metrics
#             element.x_position = element.bbox[0]
#             element.y_position = element.bbox[1]
#             element.width = element.bbox[2] - element.bbox[0]
#             element.height = element.bbox[3] - element.bbox[1]
            
#             # Text characteristics
#             element.word_count = len(element.content.split())
#             element.has_numbers = bool(re.search(r'\d', element.content))
#             element.capitalization_ratio = sum(1 for c in element.content if c.isupper()) / len(element.content)
        
#         return elements
    
#     def get_document_statistics(self, elements: List[DocumentElement]) -> Dict:
#         """Generate statistical information about the document"""
#         if not elements:
#             return {}
        
#         return {
#             'total_elements': len(elements),
#             'pages_covered': len(set(e.page_num for e in elements)),
#             'font_families': list(set(e.font_name for e in elements)),
#             'font_size_range': (min(e.font_size for e in elements), 
#                               max(e.font_size for e in elements)),
#             'average_font_size': sum(e.font_size for e in elements) / len(elements),
#             'bold_elements': len([e for e in elements if e.is_bold]),
#             'italic_elements': len([e for e in elements if e.is_italic]),
#             'total_characters': sum(e.char_count for e in elements),
#             'average_line_height': sum(e.line_height for e in elements) / len(elements)
#         }