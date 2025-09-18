from typing import Dict, List, Optional, Any, Pattern
import re
import logging

class RegexExtractor:
    """
    Handles regex-based extraction of business information from text
    """
    
    def __init__(self):
        # Indian mobile number patterns
        self.mobile_patterns = [
            r'(?:\+91[\s-]?)?[6-9]\d{9}',  # Standard Indian mobile
            r'[6-9]\d{4}[\s-]?\d{5}',     # 5+5 digit format
            r'\(0\d{2,4}\)[\s-]?\d{6,8}' # Landline with STD code
        ]
        
        # GST number patterns
        self.gst_patterns = [
            r'GST[\s:]*([0-9A-Z]{15})',  # GST followed by 15 alphanumeric
            r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b',  # Standard GST format
            r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z]\d\b'  # GST with number as last character
        ]
        
        # CIN number patterns
        self.cin_patterns = [
            r'CIN[\s:]*([A-Z][A-Z0-9]{20})',  # CIN followed by 21 alphanumeric
            r'\b[UL][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b'  # Standard CIN format
        ]
        
        # Pincode pattern (6 digits, not part of other numbers)
        self.pincode_pattern = r'\b[1-9]\d{5}\b'
        
        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def extract(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract business information using regex patterns
        
        Args:
            text: Input text to extract information from
            
        Returns:
            Dictionary containing extracted fields
        """
        data = {
            'mobile': None,
            'gst': None,
            'cin': None,
            'pincode': None,
            'email': None
        }
        
        if not text:
            return data
            
        # Extract mobile
        data['mobile'] = self._extract_mobile(text)
        
        # Extract GST
        data['gst'] = self._extract_gst(text)
        
        # Extract CIN
        data['cin'] = self._extract_cin(text)
        
        # Extract pincode
        pincode_matches = re.findall(self.pincode_pattern, text)
        # Filter pincode to avoid GST numbers
        valid_pincodes = [
            pc for pc in pincode_matches 
            if not any(pc in (data['gst'] or '', data['cin'] or '') or len(pc) != 6)
        ]
        data['pincode'] = valid_pincodes[0] if valid_pincodes else None
        
        # Extract email
        email_match = re.search(self.email_pattern, text)
        data['email'] = email_match.group() if email_match else None
        
        return data
    
    def _extract_mobile(self, text: str) -> Optional[str]:
        """Extract mobile number from text"""
        for pattern in self.mobile_patterns:
            match = re.search(pattern, text)
            if match:
                mobile = re.sub(r'[^\d+]', '', match.group())
                if len(mobile.replace('+91', '')) == 10:
                    return mobile
        return None
    
    def _extract_gst(self, text: str) -> Optional[str]:
        """Extract GST number from text"""
        for pattern in self.gst_patterns:
            match = re.search(pattern, text.upper())
            if match:
                gst = match.group(1) if match.groups() else match.group()
                gst = re.sub(r'[^\dA-Z]', '', gst)
                if len(gst) == 15:
                    return gst
        return None
    
    def _extract_cin(self, text: str) -> Optional[str]:
        """Extract CIN number from text"""
        for pattern in self.cin_patterns:
            match = re.search(pattern, text.upper())
            if match:
                cin = match.group(1) if match.groups() else match.group()
                cin = re.sub(r'[^\dA-Z]', '', cin)
                if len(cin) == 21:
                    return cin
        return None


class ExtractionUtils:
    """
    Utility class containing extraction methods for business information
    """
    
    @staticmethod
    def smart_name_extraction(text_blocks: List[Dict], all_text: str) -> Optional[str]:
        """
        Smart business name extraction based on position and size
        """
        if not text_blocks:
            return None
        
        # Score each text block for likelihood of being business name
        name_candidates = []
        
        for block in text_blocks:
            text = block['text'].strip()
            
            # Skip if too short or contains only numbers
            if len(text) < 3 or text.isdigit():
                continue
            
            # Skip if contains phone number, pincode, or GST patterns
            if re.search(r'\d{6,}', text) or 'gstin' in text.lower() or 'cin' in text.lower():
                continue
            
            # Skip common business document terms
            skip_terms = ['registered', 'officer', 'address', 'private', 'limited', 'technologies']
            if any(term in text.lower() for term in skip_terms):
                continue
            
            # Score based on position (higher score for top)
            position_score = 0
            if block.get('position') and block['position'].get('top', float('inf')) < 350:  # Near top
                position_score += 3
            if block.get('position') and block['position'].get('center_x', 0) > 100:  # Not at very left
                position_score += 2
            
            # Score based on size (larger text more likely to be name)
            size_score = 0
            if block.get('position'):
                size_score = min(block['position'].get('area', 0) / 1000, 5)
            
            # Score based on text characteristics
            text_score = 0
            if any(c.isupper() for c in text):  # Contains uppercase
                text_score += 1
            if len(text.split()) > 1:  # Multiple words
                text_score += 1
            if not any(c.isdigit() for c in text):  # No digits
                text_score += 2
            
            # Bonus for being the first text (likely title)
            if text_blocks.index(block) == 0:
                text_score += 3
            
            total_score = position_score + size_score + text_score
            
            name_candidates.append({
                'text': text,
                'score': total_score,
                'confidence': block.get('confidence', 0)
            })
        
        # Sort by score and return best candidate
        if name_candidates:
            name_candidates.sort(key=lambda x: (x['score'], x['confidence']), reverse=True)
            return name_candidates[0]['text']
        
        return None

    @staticmethod
    def extract_category(text: str, text_blocks: List[Dict] = None) -> Optional[str]:
        """
        Extract business category using predefined patterns
        """
        categories = {
            'technology': ['technology', 'technologies', 'software', 'it', 'tech', 'digital'],
            'restaurant': ['restaurant', 'cafe', 'hotel', 'food', 'kitchen', 'dining', 'biryani', 'pizza'],
            'medical': ['hospital', 'clinic', 'medical', 'doctor', 'pharmacy', 'dental'],
            'retail': ['store', 'shop', 'mart', 'bazaar', 'emporium', 'showroom'],
            'electronics': ['electronics', 'mobile', 'computer', 'laptop', 'gadget'],
            'automotive': ['garage', 'service', 'auto', 'car', 'bike', 'vehicle'],
            'beauty': ['salon', 'beauty', 'parlour', 'spa', 'cosmetic'],
            'education': ['school', 'college', 'institute', 'academy', 'training'],
            'finance': ['bank', 'atm', 'finance', 'loan', 'insurance']
        }
        
        text_lower = text.lower()
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category.title()
        
        return None

    @staticmethod
    def extract_address(text_blocks: List[Dict], full_text: str) -> Optional[str]:
        """
        Extract address from text blocks
        """
        # Look for address indicators
        address_indicators = ['address', 'apartments', 'layout', 'road', 'street', 'colony', 'nagar']
        
        address_parts = []
        for block in text_blocks:
            text = block.get('text', '').strip()
            
            # Skip if it's likely name, mobile, GST, etc.
            if any(indicator in text.lower() for indicator in ['gstin', 'cin', 'registered']):
                continue
            
            # Check if contains address indicators or seems like address
            if (any(indicator in text.lower() for indicator in address_indicators) or
                re.search(r'\d+-\d+', text) or  # Contains building numbers like B-305
                'bangalore' in text.lower() or
                'karnataka' in text.lower()):
                address_parts.append(text)
        
        return ', '.join(address_parts) if address_parts else None
