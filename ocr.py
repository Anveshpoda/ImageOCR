import cv2
import numpy as np
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
        # Initialize PaddleOCR with more detailed configuration
        print("Initializing PaddleOCR...")
        try:
            # Initialize with absolute minimum parameters
            self.ocr = PaddleOCR(lang='en')
            print("PaddleOCR initialized successfully")
        except Exception as e:
            print(f"Error initializing PaddleOCR: {str(e)}")
            raise
        
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
    
    def extract_text_with_positions(self, image_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text using PaddleOCR with position information
        """
        try:
            # Read image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return "", []
                
            # Convert to RGB (PaddleOCR expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run OCR on the image
            result = self.ocr.ocr(img_rgb)
            
            if not result or not result[0]:
                return "", []
            
            # Extract text and positions
            text_blocks = []
            full_text = ""
            
            for line in result[0]:
                if not line or len(line) < 2:
                    continue
                    
                bbox = line[0]  # Bounding box coordinates
                if len(line) > 1 and isinstance(line[1], (list, tuple)) and len(line[1]) > 1:
                    text = str(line[1][0])  # Ensure text is a string
                    confidence = float(line[1][1])  # Ensure confidence is a float
                    
                    # Only include high-confidence text
                    if confidence > 0.5:
                        position_info = self._get_position_info(bbox)
                        if position_info:  # Only add if we got valid position info
                            text_blocks.append({
                                'text': text,
                                'bbox': bbox,
                                'confidence': confidence,
                                'position': position_info
                            })
                            full_text += text + " "
            
            return full_text.strip(), text_blocks
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return "", []
    
    def _get_position_info(self, bbox: List) -> Optional[Dict]:
        """
        Extract position information from bounding box
        Returns None if the bounding box is invalid
        """
        try:
            if not bbox or len(bbox) < 4:
                return None
                
            # Ensure all points have at least 2 coordinates
            valid_points = [point for point in bbox if len(point) >= 2]
            if len(valid_points) < 4:
                return None
                
            # Extract x and y coordinates, ensuring they are numbers
            x_coords = []
            y_coords = []
            
            for point in valid_points:
                try:
                    x = float(point[0])
                    y = float(point[1])
                    x_coords.append(x)
                    y_coords.append(y)
                except (TypeError, ValueError, IndexError):
                    continue
                    
            if not x_coords or not y_coords:
                return None
                
            # Calculate position information
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Skip if the bounding box is too small or invalid
            if width <= 0 or height <= 0:
                return None
                
            return {
                'center_x': (min_x + max_x) / 2,
                'center_y': (min_y + max_y) / 2,
                'width': width,
                'height': height,
                'area': width * height,
                'top': min_y,
                'left': min_x,
                'right': max_x,
                'bottom': max_y
            }
            
        except Exception as e:
            logger.warning(f"Error processing bounding box: {str(e)}")
            return None
    
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
        
        # GST pattern (15 characters)
        gst_pattern = r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b'
        
        # Pincode pattern (6 digits)
        pincode_pattern = r'\b[1-9]\d{5}\b'
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
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
        gst_match = re.search(gst_pattern, text.upper())
        data['gst'] = gst_match.group() if gst_match else None
        
        # Extract pincode
        pincode_matches = re.findall(pincode_pattern, text)
        data['pincode'] = pincode_matches[0] if pincode_matches else None
        
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
    
    def smart_name_extraction(self, text_blocks: List[Dict]) -> Optional[str]:
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
            
            # Skip if contains phone number or pincode patterns
            if re.search(r'\d{6,}', text):
                continue
            
            # Score based on position (higher score for top and center)
            position_score = 0
            if block['position']['top'] < 100:  # Near top
                position_score += 3
            if block['position']['center_x'] > 100:  # Not at very left
                position_score += 2
            
            # Score based on size (larger text more likely to be name)
            size_score = min(block['position']['area'] / 1000, 5)
            
            # Score based on text characteristics
            text_score = 0
            if any(c.isupper() for c in text):  # Contains uppercase
                text_score += 1
            if len(text.split()) > 1:  # Multiple words
                text_score += 1
            if not any(c.isdigit() for c in text):  # No digits
                text_score += 2
            
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
    
    def extract_data(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Main method to extract data from a signboard image
        """
        try:
            # Extract text with positions
            full_text, text_blocks = self.extract_text_with_positions(image_path)
            
            # Extract using regex patterns
            regex_data = self.extract_with_regex(full_text)
            
            # Initialize result with regex data
            result = {
                'name': None,
                'mobile': regex_data.get('mobile'),
                'gst': regex_data.get('gst'),
                'pincode': regex_data.get('pincode'),
                'email': regex_data.get('email'),
                'address': None,
                'category': self.extract_category(full_text, text_blocks)
            }
            
            # Try to extract name from the first line if not found by regex
            if not result['name'] and text_blocks and len(text_blocks) > 0:
                result['name'] = text_blocks[0]['text'] if text_blocks[0]['text'] else None
            
            # Clean address (remove mobile, gst, pincode from address)
            if result['address']:
                clean_address = result['address']
                if result['mobile']:
                    clean_address = clean_address.replace(result['mobile'], '').strip()
                if result['gst']:
                    clean_address = clean_address.replace(result['gst'], '').strip()
                if result['pincode']:
                    clean_address = clean_address.replace(result['pincode'], '').strip()
                result['address'] = clean_address if clean_address else None
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, None]:
        return {
            'name': None,
            'mobile': None,
            'category': None,
            'gst': None,
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
if __name__ == "__main__":
    # Initialize extractor
    extractor = SignboardExtractor()
    
    # Load model (if available) or use regex-based extraction
    try:
        extractor.load_or_create_model("./signboard_ner_model")
    except:
        logger.info("No trained model found. Using regex-based extraction.")
    
    # Extract data from image
    image_path = "signboard_image.jpg"
    result = extractor.extract_data(image_path)
    
    print("Extracted Data:")
    for key, value in result.items():
        print(f"{key.title()}: {value}")