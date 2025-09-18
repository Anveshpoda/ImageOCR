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
            use_angle_cls=True,
            lang='en',
            det_model_dir=None,  # Use default detection model
            rec_model_dir=None,  # Use default recognition model
            cls_model_dir=None   # Use default classification model
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
    
    def extract_text_with_positions(self, image_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text using PaddleOCR with position information
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Run OCR
        # result = self.ocr.ocr(processed_image, cls=True)
        result = self.ocr.ocr(image_path)
        
        if not result or not result[0]:
            return "", []
        
        # Extract text and positions
        text_blocks = []
        full_text = ""
        
        for line in result[0]:
            bbox = line[0]  # Bounding box coordinates
            text = line[1][0]  # Recognized text
            confidence = line[1][1]  # Confidence score
            
            # Only include high-confidence text
            if confidence > 0.5:
                text_blocks.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': confidence,
                    'position': self._get_position_info(bbox)
                })
                full_text += text + " "
        
        return full_text.strip(), text_blocks
    
    def _get_position_info(self, bbox: List) -> Dict:
        """
        Extract position information from bounding box
        """
        # Calculate center, area, and position
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
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
        Main extraction function combining all methods
        """
        try:
            # Extract text and positions
            full_text, text_blocks = self.extract_text_with_positions(image_path)
            
            if not full_text:
                return self._empty_result()
            
            logger.info(f"Extracted text: {full_text}")
            
            # Extract using different methods
            regex_data = self.extract_with_regex(full_text)
            ner_data = self.extract_with_ner(full_text) if self.ner_pipeline else {}
            
            # Smart business name extraction
            business_name = self.smart_name_extraction(text_blocks)
            
            # Extract category
            category = self.extract_category(full_text, text_blocks)
            
            # Combine results with priority (NER > Regex > Smart extraction)
            result = {
                'name': ner_data.get('name') or business_name,
                'mobile': ner_data.get('mobile') or regex_data.get('mobile'),
                'category': ner_data.get('category') or category,
                'gst': ner_data.get('gst') or regex_data.get('gst'),
                'address': ner_data.get('address'),
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
    image_path = "signboard_image.jpg"
    result = extractor.extract_data(image_path)

    print("Extracted Data:")
    for key, value in result.items():
        print(f"{key.title()}: {value}")
