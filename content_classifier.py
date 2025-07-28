"""
Content Classifier Module - Classify and enhance document elements
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

from document_parser import DocumentElement

logger = logging.getLogger(__name__)

class ContentClassifier:
    """Advanced content classifier for document elements"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize content classifier with configuration"""
        self.config = config or {
            'heading_patterns': [
                r'^\d+\.?\s+[A-Z]',  # Numbered headings
                r'^[A-Z][A-Za-z\s]{2,50}$',  # Title case
                r'^[A-Z\s]{3,50}$',  # All caps (short)
                r'^\d+\.\d+\.?\s+',  # Subsection numbering
                r'^(Chapter|Section|Part)\s+\d+',  # Explicit chapter/section
            ],
            'content_indicators': [
                r'\.\s+[A-Z]',  # Sentence structure
                r'[a-z]\s+[a-z]',  # Lowercase words
                r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',  # Common words
            ],
            'title_indicators': [
                r'^\s*[A-Z][^.!?]*$',  # Starts with capital, no sentence ending
                r'^[^a-z]*[A-Z][^a-z]*$',  # No lowercase letters
            ]
        }
        
        # Precompile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.heading_regex = [re.compile(pattern, re.IGNORECASE) 
                             for pattern in self.config['heading_patterns']]
        self.content_regex = [re.compile(pattern, re.IGNORECASE) 
                             for pattern in self.config['content_indicators']]
        self.title_regex = [re.compile(pattern) 
                           for pattern in self.config['title_indicators']]
    
    async def classify_elements(self, elements: List[DocumentElement], 
                              embeddings: List[np.ndarray]) -> List[DocumentElement]:
        """
        Classify and enhance document elements with additional metadata
        """
        if not elements:
            return []
        
        classified_elements = []
        
        # Calculate document-wide statistics for context
        doc_stats = self._calculate_document_statistics(elements)
        
        for i, element in enumerate(elements):
            # Create enhanced element copy
            enhanced_element = self._create_enhanced_element(
                element, embeddings[i] if i < len(embeddings) else None, doc_stats
            )
            
            # Classify element type
            element_type = self._classify_element_type(enhanced_element, doc_stats)
            enhanced_element.classification = element_type
            
            # Calculate heading probability
            heading_probability = self._calculate_heading_probability(
                enhanced_element, doc_stats
            )
            enhanced_element.heading_probability = heading_probability
            
            # Calculate title probability
            title_probability = self._calculate_title_probability(
                enhanced_element, doc_stats
            )
            enhanced_element.title_probability = title_probability
            
            # Add contextual information
            enhanced_element.context = self._analyze_element_context(
                enhanced_element, elements, i
            )
            
            classified_elements.append(enhanced_element)
        
        # Post-process classifications with context
        refined_elements = self._refine_classifications(classified_elements)
        
        logger.info(f"Classified {len(refined_elements)} elements")
        return refined_elements
    
    def _calculate_document_statistics(self, elements: List[DocumentElement]) -> Dict:
        """Calculate document-wide statistics for classification context"""
        if not elements:
            return {}
        
        font_sizes = [e.font_size for e in elements if e.font_size is not None]
        word_counts = [len(e.content.split()) for e in elements if e.content]
        
        # Handle edge cases for empty lists
        if not font_sizes:
            font_sizes = [12.0]  # Default font size
        if not word_counts:
            word_counts = [0]
        
        page_nums = [e.page_num for e in elements if e.page_num is not None]
        max_page = max(page_nums) if page_nums else 1
        
        return {
            'avg_font_size': np.mean(font_sizes),
            'max_font_size': max(font_sizes),
            'min_font_size': min(font_sizes),
            'font_size_std': np.std(font_sizes),
            'avg_word_count': np.mean(word_counts),
            'total_elements': len(elements),
            'bold_elements': sum(1 for e in elements if e.is_bold),
            'pages_total': max_page,
            'unique_fonts': len(set(e.font_name for e in elements if e.font_name))
        }
    
    def _create_enhanced_element(self, element: DocumentElement, 
                               embedding: Optional[np.ndarray],
                               doc_stats: Dict) -> DocumentElement:
        """Create enhanced element with additional computed properties"""
        # Create a copy of the element
        enhanced = DocumentElement(
            content=element.content,
            page_num=element.page_num,
            bbox=element.bbox,
            font_name=element.font_name,
            font_size=element.font_size,
            font_flags=element.font_flags,
            color=element.color,
            is_bold=element.is_bold,
            is_italic=element.is_italic,
            line_height=element.line_height,
            char_count=element.char_count
        )
        
        # Add computed properties from document_parser if they exist
        if hasattr(element, 'font_size_ratio'):
            enhanced.font_size_ratio = element.font_size_ratio
            enhanced.max_font_ratio = element.max_font_ratio
            enhanced.x_position = element.x_position
            enhanced.y_position = element.y_position
            enhanced.width = element.width
            enhanced.height = element.height
            enhanced.word_count = element.word_count
            enhanced.has_numbers = element.has_numbers
            enhanced.capitalization_ratio = element.capitalization_ratio
        else:
            # Calculate them if not present
            avg_font_size = doc_stats.get('avg_font_size', 12.0)
            max_font_size = doc_stats.get('max_font_size', 12.0)
            
            enhanced.font_size_ratio = element.font_size / avg_font_size if avg_font_size > 0 else 1.0
            enhanced.max_font_ratio = element.font_size / max_font_size if max_font_size > 0 else 1.0
            enhanced.x_position = element.bbox[0] if element.bbox else 0
            enhanced.y_position = element.bbox[1] if element.bbox else 0
            enhanced.width = (element.bbox[2] - element.bbox[0]) if element.bbox else 0
            enhanced.height = (element.bbox[3] - element.bbox[1]) if element.bbox else 0
            enhanced.word_count = len(element.content.split()) if element.content else 0
            enhanced.has_numbers = bool(re.search(r'\d', element.content)) if element.content else False
            enhanced.capitalization_ratio = (sum(1 for c in element.content if c.isupper()) / 
                                           len(element.content) if element.content else 0)
        
        # Add new enhanced properties
        enhanced.embedding = embedding
        enhanced.text_density = len(element.content) / enhanced.width if enhanced.width > 0 else 0
        enhanced.relative_position = enhanced.y_position / 800  # Normalize to typical page height
        enhanced.is_first_page = element.page_num == 1
        enhanced.is_early_page = element.page_num <= 2
        enhanced.length_category = self._categorize_text_length(len(element.content) if element.content else 0)
        enhanced.formatting_score = self._calculate_formatting_score(enhanced)
        
        # Initialize additional attributes
        enhanced.classification = 'text'
        enhanced.heading_probability = 0.0
        enhanced.title_probability = 0.0
        enhanced.context = {}
        enhanced.is_document_title = False
        enhanced.hierarchy_level = None
        enhanced.heading_rank = None
        enhanced.is_toc_entry = False
        
        return enhanced
    
    def _categorize_text_length(self, length: int) -> str:
        """Categorize text length for classification"""
        if length <= 10:
            return 'very_short'
        elif length <= 30:
            return 'short'
        elif length <= 80:
            return 'medium'
        elif length <= 200:
            return 'long'
        else:
            return 'very_long'
    
    def _calculate_formatting_score(self, element: DocumentElement) -> float:
        """Calculate a formatting score indicating heading likelihood"""
        score = 0.0
        
        # Font size contribution
        if hasattr(element, 'font_size_ratio'):
            if element.font_size_ratio > 1.5:
                score += 0.3
            elif element.font_size_ratio > 1.2:
                score += 0.2
            elif element.font_size_ratio > 1.0:
                score += 0.1
        
        # Bold formatting
        if element.is_bold:
            score += 0.2
        
        # Position (left-aligned or centered)
        if element.x_position < 100:  # Left-aligned
            score += 0.1
        elif 100 <= element.x_position <= 400:  # Centered
            score += 0.15
        
        # Page position (earlier is better)
        if element.page_num == 1:
            score += 0.1
        elif element.page_num <= 3:
            score += 0.05
        
        # Length appropriateness
        text_length = len(element.content) if element.content else 0
        if 10 <= text_length <= 60:
            score += 0.15
        elif 5 <= text_length <= 100:
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_element_context(self, element: DocumentElement, 
                               all_elements: List[DocumentElement], 
                               current_index: int) -> Dict:
        """Analyze contextual information around the element"""
        context = {
            'preceding_elements': [],
            'following_elements': [],
            'same_page_elements': 0,
            'similar_font_elements': 0,
            'position_in_document': current_index / len(all_elements) if all_elements else 0
        }
        
        # Analyze preceding elements (up to 3)
        start_idx = max(0, current_index - 3)
        context['preceding_elements'] = [
            {
                'text': elem.content[:50] if elem.content else '',
                'font_size': elem.font_size,
                'is_bold': elem.is_bold,
                'page': elem.page_num
            }
            for elem in all_elements[start_idx:current_index]
        ]
        
        # Analyze following elements (up to 3)
        end_idx = min(len(all_elements), current_index + 4)
        context['following_elements'] = [
            {
                'text': elem.content[:50] if elem.content else '',
                'font_size': elem.font_size,
                'is_bold': elem.is_bold,
                'page': elem.page_num
            }
            for elem in all_elements[current_index + 1:end_idx]
        ]
        
        # Count elements on same page
        context['same_page_elements'] = sum(
            1 for elem in all_elements 
            if elem.page_num == element.page_num
        )
        
        # Count elements with similar font size
        similar_font_threshold = 2.0  # Points
        context['similar_font_elements'] = sum(
            1 for elem in all_elements 
            if abs(elem.font_size - element.font_size) <= similar_font_threshold
        )
        
        return context
    
    def _classify_element_type(self, element: DocumentElement, 
                             doc_stats: Dict) -> str:
        """Classify element type based on multiple signals"""
        content = element.content.strip() if element.content else ''
        
        if not content:
            return 'empty'
        
        # Check for explicit patterns
        for pattern in self.heading_regex:
            if pattern.match(content):
                return 'heading'
        
        # Check for title patterns
        for pattern in self.title_regex:
            if pattern.match(content):
                if element.page_num <= 2 and getattr(element, 'font_size_ratio', 1.0) > 1.3:
                    return 'title'
        
        # Check for content patterns
        content_matches = sum(1 for pattern in self.content_regex 
                            if pattern.search(content))
        
        if content_matches >= 2 and len(content) > 50:
            return 'content'
        
        # Heuristic classification
        if (element.is_bold and 
            getattr(element, 'font_size_ratio', 1.0) > 1.1 and 
            len(content.split()) <= 15):
            return 'heading'
        
        if (element.page_num == 1 and 
            getattr(element, 'y_position', 0) < 200 and 
            getattr(element, 'font_size_ratio', 1.0) > 1.2):
            return 'title'
        
        if len(content) > 100 or content.count('.') > 2:
            return 'content'
        
        return 'text'
    
    def _calculate_heading_probability(self, element: DocumentElement, 
                                     doc_stats: Dict) -> float:
        """Calculate probability that element is a heading"""
        probability = 0.0
        content = element.content if element.content else ''
        
        # Font size factor
        if hasattr(element, 'font_size_ratio'):
            font_factor = min(element.font_size_ratio - 1.0, 1.0)
            probability += max(0, font_factor) * 0.3
        
        # Bold formatting
        if element.is_bold:
            probability += 0.25
        
        # Length appropriateness
        word_count = len(content.split())
        if 2 <= word_count <= 12:
            probability += 0.2
        elif 1 <= word_count <= 20:
            probability += 0.1
        
        # Position factors
        x_pos = getattr(element, 'x_position', 0)
        y_pos = getattr(element, 'y_position', 0)
        
        if x_pos < 150:  # Left-aligned
            probability += 0.1
        
        if y_pos < 200:  # Top of page
            probability += 0.1
        
        # Pattern matching
        content_stripped = content.strip()
        pattern_matches = sum(1 for pattern in self.heading_regex 
                            if pattern.match(content_stripped))
        probability += min(pattern_matches * 0.15, 0.3)
        
        # Capitalization
        cap_ratio = getattr(element, 'capitalization_ratio', 0)
        if cap_ratio > 0.5:
            probability += 0.1
        
        return min(probability, 1.0)
    
    def _calculate_title_probability(self, element: DocumentElement, 
                                   doc_stats: Dict) -> float:
        """Calculate probability that element is the document title"""
        if element.page_num > 2:
            return 0.0
        
        probability = 0.0
        
        # Page position (first page strongly preferred)
        if element.page_num == 1:
            probability += 0.4
            y_pos = getattr(element, 'y_position', 0)
            if y_pos < 300:  # Top half of first page
                probability += 0.2
        
        # Font size contribution
        if hasattr(element, 'max_font_ratio'):
            if element.max_font_ratio > 0.9:  # Close to maximum font size
                probability += 0.3
            elif element.max_font_ratio > 0.7:
                probability += 0.2
        
        # Bold formatting
        if element.is_bold:
            probability += 0.15
        
        # Length appropriateness for titles
        text_length = len(element.content) if element.content else 0
        if 10 <= text_length <= 80:
            probability += 0.2
        elif 5 <= text_length <= 120:
            probability += 0.1
        
        # Center alignment (common for titles)
        x_pos = getattr(element, 'x_position', 0)
        if 100 <= x_pos <= 400:
            probability += 0.15
        
        # Title-like patterns
        content = element.content.strip() if element.content else ''
        for pattern in self.title_regex:
            if pattern.match(content):
                probability += 0.1
        
        # Avoid content-like characteristics
        if content.count('.') > 1:  # Multiple sentences
            probability *= 0.5
        
        if any(pattern.search(content) for pattern in self.content_regex):
            probability *= 0.7
        
        return min(probability, 1.0)
    
    def _refine_classifications(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Refine classifications using contextual information"""
        refined_elements = elements.copy()
        
        # Find potential titles (highest probability on first pages)
        title_candidates = [
            (i, elem) for i, elem in enumerate(elements)
            if getattr(elem, 'title_probability', 0) > 0.3 and elem.page_num <= 2
        ]
        
        if title_candidates:
            # Select best title candidate
            best_title_idx, best_title = max(
                title_candidates, 
                key=lambda x: getattr(x[1], 'title_probability', 0)
            )
            refined_elements[best_title_idx].classification = 'title'
            refined_elements[best_title_idx].is_document_title = True
        
        # Refine heading classifications based on hierarchy
        self._refine_heading_hierarchy(refined_elements)
        
        # Identify table of contents
        self._identify_table_of_contents(refined_elements)
        
        return refined_elements
    
    def _refine_heading_hierarchy(self, elements: List[DocumentElement]):
        """Refine heading classifications to ensure proper hierarchy"""
        heading_elements = [
            (i, elem) for i, elem in enumerate(elements)
            if getattr(elem, 'classification', '') in ['heading', 'title'] and 
            getattr(elem, 'heading_probability', 0) > 0.4
        ]
        
        if not heading_elements:
            return
        
        # Sort by font size (descending) and then by position
        heading_elements.sort(
            key=lambda x: (-x[1].font_size, x[1].page_num, getattr(x[1], 'y_position', 0))
        )
        
        # Assign hierarchy levels based on font size groupings
        font_sizes = [elem.font_size for _, elem in heading_elements]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        for idx, (elem_idx, element) in enumerate(heading_elements):
            size_rank = unique_sizes.index(element.font_size)
            element.hierarchy_level = f"H{size_rank + 1}"
            element.heading_rank = size_rank
    
    def _identify_table_of_contents(self, elements: List[DocumentElement]):
        """Identify table of contents sections"""
        for i, element in enumerate(elements):
            content_lower = element.content.lower() if element.content else ''
            
            # Look for TOC indicators
            if any(indicator in content_lower for indicator in 
                   ['table of contents', 'contents', 'index']):
                element.classification = 'table_of_contents'
                
                # Mark following elements as potential TOC entries
                for j in range(i + 1, min(i + 20, len(elements))):
                    if (elements[j].page_num <= element.page_num + 2 and
                        getattr(elements[j], 'has_numbers', False)):
                        elements[j].is_toc_entry = True
    
    def get_classification_statistics(self, elements: List[DocumentElement]) -> Dict:
        """Generate statistics about the classification results"""
        if not elements:
            return {}
        
        classifications = Counter(getattr(elem, 'classification', 'unknown') for elem in elements)
        
        heading_probs = [
            getattr(elem, 'heading_probability', 0) for elem in elements 
            if hasattr(elem, 'heading_probability')
        ]
        
        title_probs = [
            getattr(elem, 'title_probability', 0) for elem in elements 
            if hasattr(elem, 'title_probability')
        ]
        
        avg_heading_prob = np.mean(heading_probs) if heading_probs else 0.0
        avg_title_prob = np.mean(title_probs) if title_probs else 0.0
        
        return {
            'total_elements': len(elements),
            'classification_counts': dict(classifications),
            'average_heading_probability': float(avg_heading_prob),
            'average_title_probability': float(avg_title_prob),
            'elements_with_high_heading_prob': sum(
                1 for elem in elements 
                if getattr(elem, 'heading_probability', 0) > 0.7
            ),
            'potential_titles': sum(
                1 for elem in elements 
                if getattr(elem, 'title_probability', 0) > 0.5
            )
        }