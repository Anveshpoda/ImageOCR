import cv2
import numpy as np
import os
import re
from paddleocr import PaddleOCR
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TokenClassificationPipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignboardExtractor:
    """
    High-accuracy open-source signboard data extraction system
    """
    
    def __init__(self):
        # Initialize PaddleOCR with best settings

        # self.ocr = PaddleOCR(
        #     use_angle_cls=True,
        #     lang='en',
        #     # use_gpu=torch.cuda.is_available(),
        #     show_log=False,
        #     det_model_dir=None,  # Use default detection model
        #     rec_model_dir=None,  # Use default recognition model
        #     cls_model_dir=None   # Use default classification model
        # )

        self.ocr = PaddleOCR(
            use_textline_orientation=True,  # Updated parameter name
            lang='en',
            text_detection_model_dir=None,  # Updated parameter name
            text_recognition_model_dir=None,  # Updated parameter name
            textline_orientation_model_dir=None   # Updated parameter name
        )
        
        # NER labels for BIO tagging
        self.labels = [
            'O',           # Outside
            'B-NAME',      # Beginning of business name
            'I-NAME',      # Inside business name
            'B-MOBILE',    # Beginning of mobile number
            'I-MOBILE',    # Inside mobile number
            'B-CATEGORY',  # Beginning of category
            'I-CATEGORY',  # Inside category
            'B-GST',       # Beginning of GST
            'I-GST',       # Inside GST
            'B-ADDRESS',   # Beginning of address
            'I-ADDRESS',   # Inside address
            'B-PINCODE',   # Beginning of pincode
            'I-PINCODE'    # Inside pincode
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        # Initialize tokenizer and model (will be loaded/trained)
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        """
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1], 
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    def extract_text_with_positions(self, image_path: str) -> Tuple[str, List[Dict], List[str]]:
        """
        Extract text using PaddleOCR with position information
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_path)

        # Run OCR
        result = self.ocr.predict(image_path)

        if not result:
            return "", [], []

        # Handle new PaddleOCR format
        text_blocks = []
        full_text = ""
        all_detected_text = []

        print(f"DEBUG: OCR result type: {type(result)}")
        print(f"DEBUG: OCR result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

        # Extract data from the new format - result is directly a dictionary
        if isinstance(result, dict):
            # New format returns a dictionary directly
            rec_texts = result.get('rec_texts', [])
            rec_scores = result.get('rec_scores', [])
            rec_polys = result.get('rec_polys', [])
            
            print("DEBUG: All detected text by OCR:")
            for i, text in enumerate(rec_texts):
                confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                print(f"  {i+1}. '{text}' (confidence: {confidence:.3f})")
                all_detected_text.append(text)
                
                if confidence > 0.5 and text.strip():
                    bbox = rec_polys[i] if i < len(rec_polys) else None
                    
                    text_blocks.append({
                        "text": text,
                        "bbox": bbox.tolist() if bbox is not None else None,
                        "confidence": confidence,
                        "position": self._get_position_info(bbox) if bbox is not None else None
                    })
                    full_text += text + " "
        elif isinstance(result, list):
            # Handle if result is a list containing the dictionary
            for item in result:
                if isinstance(item, dict) and 'rec_texts' in item:
                    rec_texts = item.get('rec_texts', [])
                    rec_scores = item.get('rec_scores', [])
                    rec_polys = item.get('rec_polys', [])
                    
                    print("DEBUG: All detected text by OCR:")
                    for i, text in enumerate(rec_texts):
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        print(f"  {i+1}. '{text}' (confidence: {confidence:.3f})")
                        all_detected_text.append(text)
                        
                        if confidence > 0.5 and text.strip():
                            bbox = rec_polys[i] if i < len(rec_polys) else None
                            
                            text_blocks.append({
                                "text": text,
                                "bbox": bbox.tolist() if bbox is not None else None,
                                "confidence": confidence,
                                "position": self._get_position_info(bbox) if bbox is not None else None
                            })
                            full_text += text + " "
                    break
                else:
                    # Fallback for older format
                    if hasattr(item, "txt"):
                        text = item.txt
                        confidence = getattr(item, "confidence", 1.0)
                        bbox = getattr(item, "poly", None)
                        all_detected_text.append(text)

                        if confidence > 0.5 and text.strip():
                            text_blocks.append({
                                "text": text,
                                "bbox": bbox,
                                "confidence": confidence,
                                "position": self._get_position_info(bbox) if bbox else None
                            })
                            full_text += text + " "
                    else:
                        logger.warning(f"Unexpected OCR result format: {item}")
        else:
            logger.warning(f"Unexpected OCR result format: {result}")

        print(f"\nDEBUG: Combined full text: '{full_text.strip()}'")
        return full_text.strip(), text_blocks, all_detected_text

    def _get_position_info(self, bbox) -> Dict:
        """
        Extract position information from bounding box
        """
        try:
            # Handle numpy array
            if hasattr(bbox, 'tolist'):
                bbox = bbox.tolist()
            
            # Calculate center, area, and position
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            area = width * height
            
            return {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'area': area,
                'top': min(y_coords),
                'left': min(x_coords)
            }
        except Exception as e:
            logger.warning(f"Error processing bbox: {e}")
            return {
                'center_x': 0, 'center_y': 0, 'width': 0, 
                'height': 0, 'area': 0, 'top': 0, 'left': 0
            }
    
    def extract_with_regex(self, text: str) -> Dict[str, Optional[str]]:
        """
        Rule-based extraction using regex patterns
        """
        # Indian mobile number patterns
        mobile_patterns = [
            r'(?:\+91[-.\s]?)?[6-9]\d{9}',
            r'(?:\+91[-.\s]?)?[6-9]\d{2}[-.\s]?\d{3}[-.\s]?\d{4}',
            r'(?:0\d{2,4}[-.\s]?)?[6-9]\d{2}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        # GST pattern (15 characters) - Updated to handle the format in your data
        gst_patterns = [
            r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b',
            r'GSTIN\s+N[O0][-\s]*([A-Z0-9]{15})',
            r'GST\s+N[O0][-\s]*([A-Z0-9]{15})',
            r'\b[0-9]{2}[A-Z]{2}[0-9A-Z]{11}\b'
        ]
        
        # Pincode pattern (6 digits)
        pincode_pattern = r'\b[1-9]\d{5}\b'
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # CIN pattern - updated to handle the format in your data
        cin_patterns = [
            r'CIN\s+No\s*[-\s]*([A-Z0-9]{21})',
            r'CIN\s*[-:\s]*([A-Z0-9]{21})',
            r'\b([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b'
        ]
        
        data = {}
        
        # Extract mobile
        mobile = None
        for pattern in mobile_patterns:
            match = re.search(pattern, text)
            if match:
                mobile = re.sub(r'[^\d+]', '', match.group())
                if len(mobile.replace('+91', '')) == 10:
                    break
        data['mobile'] = mobile
        
        # Extract GST
        gst = None
        for pattern in gst_patterns:
            match = re.search(pattern, text.upper())
            if match:
                gst = match.group(1) if match.groups() else match.group()
                # Clean GST number
                gst = re.sub(r'[^\dA-Z]', '', gst)
                if len(gst) == 15:
                    break
        data['gst'] = gst
        
        # Extract CIN
        cin = None
        for pattern in cin_patterns:
            match = re.search(pattern, text.upper())
            if match:
                cin = match.group(1) if match.groups() else match.group()
                # Clean CIN number
                cin = re.sub(r'[^\dA-Z]', '', cin)
                if len(cin) == 21:
                    break
        data['cin'] = cin
        
        # Extract pincode
        pincode_matches = re.findall(pincode_pattern, text)
        # Filter pincode to avoid GST numbers
        valid_pincodes = [pc for pc in pincode_matches if not any(pc in gst_text for gst_text in [data['gst'] or '', data['cin'] or ''])]
        data['pincode'] = valid_pincodes[0] if valid_pincodes else None
        
        # Extract email
        email_match = re.search(email_pattern, text)
        data['email'] = email_match.group() if email_match else None
        
        return data
    
    def load_or_create_model(self, model_path: str = None):
        """
        Load pre-trained model or create new one
        """
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            logger.info("Creating new model based on BERT")
            model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id
            )
        
        # Create NER pipeline
        self.ner_pipeline = TokenClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
    
    def extract_with_ner(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract information using trained NER model
        """
        if not self.ner_pipeline:
            logger.warning("NER model not loaded. Using regex fallback.")
            return self.extract_with_regex(text)
        
        # Run NER
        entities = self.ner_pipeline(text)
        
        # Group entities by type
        extracted = {}
        for entity in entities:
            entity_type = entity['entity_group'].replace('B-', '').replace('I-', '').lower()
            if entity_type not in extracted:
                extracted[entity_type] = entity['word']
            else:
                # Concatenate if multiple parts
                extracted[entity_type] += " " + entity['word']
        
        # Clean up extracted data
        cleaned_data = {}
        for key, value in extracted.items():
            cleaned_value = value.replace('##', '').strip()
            cleaned_data[key] = cleaned_value if cleaned_value else None
        
        return cleaned_data
    
    def smart_name_extraction(self, text_blocks: List[Dict], all_text: str) -> Optional[str]:
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
            if block['position'] and block['position']['top'] < 350:  # Near top
                position_score += 3
            if block['position'] and block['position']['center_x'] > 100:  # Not at very left
                position_score += 2
            
            # Score based on size (larger text more likely to be name)
            size_score = 0
            if block['position']:
                size_score = min(block['position']['area'] / 1000, 5)
            
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
                'confidence': block['confidence']
            })
        
        # Sort by score and return best candidate
        if name_candidates:
            name_candidates.sort(key=lambda x: (x['score'], x['confidence']), reverse=True)
            return name_candidates[0]['text']
        
        return None
    
    def extract_category(self, text: str, text_blocks: List[Dict] = None) -> Optional[str]:
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
    
    def extract_address(self, text_blocks: List[Dict], full_text: str) -> Optional[str]:
        """
        Extract address from text blocks
        """
        # Look for address indicators
        address_indicators = ['address', 'apartments', 'layout', 'road', 'street', 'colony', 'nagar']
        
        address_parts = []
        for block in text_blocks:
            text = block['text'].strip()
            
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
    
    def extract_data(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Main extraction function combining all methods
        """
        try:
            # Extract text and positions
            full_text, text_blocks, all_detected_text = self.extract_text_with_positions(image_path)
            
            if not full_text:
                return self._empty_result(), []
            
            logger.info(f"Extracted text: {full_text}")
            
            # Extract using different methods
            regex_data = self.extract_with_regex(full_text)
            ner_data = self.extract_with_ner(full_text) if self.ner_pipeline else {}
            
            # Smart business name extraction
            business_name = self.smart_name_extraction(text_blocks, full_text)
            
            # Extract category
            category = self.extract_category(full_text, text_blocks)
            
            # Extract address
            address = self.extract_address(text_blocks, full_text)
            
            # Combine results with priority (NER > Regex > Smart extraction)
            result = {
                'name': ner_data.get('name') or business_name,
                'mobile': ner_data.get('mobile') or regex_data.get('mobile'),
                'category': ner_data.get('category') or category,
                'gst': ner_data.get('gst') or regex_data.get('gst'),
                'cin': regex_data.get('cin'),
                'address': ner_data.get('address') or address,
                'pincode': ner_data.get('pincode') or regex_data.get('pincode'),
                'email': regex_data.get('email')  # Email from regex is usually reliable
            }
            
            # Clean address (remove mobile, gst, pincode from address)
            if result['address']:
                clean_address = result['address']
                if result['mobile']:
                    clean_address = clean_address.replace(result['mobile'], '').strip()
                if result['gst']:
                    clean_address = clean_address.replace(result['gst'], '').strip()
                if result['pincode']:
                    clean_address = clean_address.replace(result['pincode'], '').strip()
                # Clean up extra commas and spaces
                clean_address = re.sub(r'\s*,\s*,\s*', ', ', clean_address)
                clean_address = re.sub(r'^\s*,\s*|\s*,\s*$', '', clean_address)
                result['address'] = clean_address if clean_address else None
            
            return result, all_detected_text
            
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return self._empty_result(), []
    
    def _empty_result(self) -> Dict[str, None]:
        return {
            'name': None,
            'mobile': None,
            'category': None,
            'gst': None,
            'cin': None,
            'address': None,
            'pincode': None,
            'email': None
        }

# Dataset class for training NER model
class SignboardDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokens
        aligned_labels = self.align_labels_with_tokens(labels, encoding)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def align_labels_with_tokens(self, labels, encoding):
        # This is a simplified version - you'll need to implement proper alignment
        # based on your specific labeling format
        aligned = [-100] * self.max_length  # -100 is ignored in loss calculation
        # Implementation depends on your exact data format
        return aligned

# Training function
def train_ner_model(train_data, val_data, output_dir="./signboard_ner_model"):
    """
    Train NER model for signboard data extraction
    """
    extractor = SignboardExtractor()
    extractor.load_or_create_model()
    
    # Create datasets
    train_dataset = SignboardDataset(
        train_data['texts'], 
        train_data['labels'], 
        extractor.tokenizer
    )
    val_dataset = SignboardDataset(
        val_data['texts'], 
        val_data['labels'], 
        extractor.tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(extractor.tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=extractor.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    extractor.tokenizer.save_pretrained(output_dir)
    
    return extractor

# Usage example
# if __name__ == "__main__":
#     # Initialize extractor
#     extractor = SignboardExtractor()
    
#     # Load model (if available) or use regex-based extraction
#     try:
#         extractor.load_or_create_model("./signboard_ner_model")
#     except:
#         logger.info("No trained model found. Using regex-based extraction.")
    
#     # Extract data from image
#     image_path = "signboard_image.jpg"
#     result = extractor.extract_data(image_path)
    
#     print("Extracted Data:")
#     for key, value in result.items():
#         print(f"{key.title()}: {value}")

if __name__ == "__main__":
    extractor = SignboardExtractor()

    # Just use regex + OCR (no model)
    image_path = "../test_img.jpg"
    result, all_text = extractor.extract_data(image_path)

    print("\n" + "="*50)
    print("EXTRACTED DATA:")
    print("="*50)
    for key, value in result.items():
        print(f"{key.title()}: {value}")
    
    print("\n" + "="*50)
    print("ALL DETECTED TEXT:")
    print("="*50)
    for i, text in enumerate(all_text, 1):
        print(f"{i:2d}. {text}")