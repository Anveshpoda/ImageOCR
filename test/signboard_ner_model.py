import json
import os
from typing import List, Dict, Tuple
import pandas as pd

class SignboardTrainingDataCreator:
    """
    Create training data for NER model from signboard text annotations
    """
    
    def __init__(self):
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
    
    def create_bio_tags(self, text: str, annotations: Dict) -> List[Tuple[str, str]]:
        """
        Create BIO tags for given text and annotations
        """
        words = text.split()
        tags = ['O'] * len(words)
        
        # Tag each entity type
        for field, value in annotations.items():
            if value and field.upper() in ['NAME', 'MOBILE', 'CATEGORY', 'GST', 'ADDRESS', 'PINCODE']:
                self._tag_entity(words, tags, value, field.upper())
        
        return list(zip(words, tags))
    
    def _tag_entity(self, words: List[str], tags: List[str], entity_text: str, entity_type: str):
        """
        Tag entity in BIO format
        """
        entity_words = entity_text.split()
        
        # Find entity in text
        for i in range(len(words) - len(entity_words) + 1):
            if self._words_match(words[i:i+len(entity_words)], entity_words):
                # Tag first word as B-
                tags[i] = f'B-{entity_type}'
                # Tag remaining words as I-
                for j in range(1, len(entity_words)):
                    if i + j < len(tags):
                        tags[i + j] = f'I-{entity_type}'
                break
    
    def _words_match(self, words1: List[str], words2: List[str]) -> bool:
        """
        Check if two word sequences match (case-insensitive, punctuation-tolerant)
        """
        if len(words1) != len(words2):
            return False
        
        for w1, w2 in zip(words1, words2):
            # Remove punctuation and convert to lowercase
            clean_w1 = ''.join(c.lower() for c in w1 if c.isalnum())
            clean_w2 = ''.join(c.lower() for c in w2 if c.isalnum())
            if clean_w1 != clean_w2:
                return False
        
        return True
    
    def process_training_data(self, data_file: str) -> Dict:
        """
        Process training data from JSON file
        
        Expected format:
        [
            {
                "text": "Sharma Electronics Mobile Shop 9876543210 MG Road Bangalore 560001",
                "annotations": {
                    "name": "Sharma Electronics",
                    "mobile": "9876543210",
                    "category": "Mobile Shop", 
                    "address": "MG Road Bangalore",
                    "pincode": "560001"
                }
            }
        ]
        """
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        
        texts = []
        labels = []
        
        for item in raw_data:
            text = item['text']
            annotations = item['annotations']
            
            # Create BIO tags
            word_tag_pairs = self.create_bio_tags(text, annotations)
            
            words = [pair[0] for pair in word_tag_pairs]
            tags = [pair[1] for pair in word_tag_pairs]
            
            texts.append(words)
            labels.append(tags)
        
        return {
            'texts': texts,
            'labels': labels
        }
    
    def create_sample_data(self, output_file: str = 'sample_signboard_data.json'):
        """
        Create sample training data
        """
        sample_data = [
            {
                "text": "Sharma Electronics Mobile Repair Shop 9876543210 29ABCDE1234F1Z5 MG Road Bangalore 560001",
                "annotations": {
                    "name": "Sharma Electronics",
                    "category": "Mobile Repair Shop",
                    "mobile": "9876543210",
                    "gst": "29ABCDE1234F1Z5",
                    "address": "MG Road Bangalore",
                    "pincode": "560001"
                }
            },
            {
                "text": "Royal Restaurant Pure Veg Hotel +91-9988776655 Food Court Building Church Street 560025",
                "annotations": {
                    "name": "Royal Restaurant",
                    "category": "Pure Veg Hotel", 
                    "mobile": "+91-9988776655",
                    "address": "Food Court Building Church Street",
                    "pincode": "560025"
                }
            },
            {
                "text": "Dr. Kumar Dental Clinic Dentist 080-25551234 Dental Care Centre Indiranagar 560038",
                "annotations": {
                    "name": "Dr. Kumar Dental Clinic",
                    "category": "Dentist",
                    "mobile": "080-25551234", 
                    "address": "Dental Care Centre Indiranagar",
                    "pincode": "560038"
                }
            },
            {
                "text": "Anand Bakery Fresh Cakes Sweets 9123456789 27XYZTE5678G2H4 Commercial Street Market 560001",
                "annotations": {
                    "name": "Anand Bakery",
                    "category": "Fresh Cakes Sweets",
                    "mobile": "9123456789",
                    "gst": "27XYZTE5678G2H4",
                    "address": "Commercial Street Market", 
                    "pincode": "560001"
                }
            },
            {
                "text": "City Fashion Store Ladies Gents Garments 9876543210 Fashion Plaza Mall Brigade Road 560025",
                "annotations": {
                    "name": "City Fashion Store",
                    "category": "Ladies Gents Garments",
                    "mobile": "9876543210",
                    "address": "Fashion Plaza Mall Brigade Road",
                    "pincode": "560025"
                }
            }
        ]
        
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"Sample data created: {output_file}")
        return sample_data
    
    def export_for_training(self, processed_data: Dict, train_split: float = 0.8) -> Tuple[Dict, Dict]:
        """
        Split data into train and validation sets
        """
        total_samples = len(processed_data['texts'])
        train_size = int(total_samples * train_split)
        
        train_data = {
            'texts': processed_data['texts'][:train_size],
            'labels': processed_data['labels'][:train_size]
        }
        
        val_data = {
            'texts': processed_data['texts'][train_size:],
            'labels': processed_data['labels'][train_size:]
        }
        
        return train_data, val_data
    
    def validate_annotations(self, data_file: str) -> List[str]:
        """
        Validate annotation quality
        """
        issues = []
        
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        
        for i, item in enumerate(raw_data):
            text = item['text'].lower()
            annotations = item.get('annotations', {})
            
            # Check if annotated entities exist in text
            for field, value in annotations.items():
                if value and value.lower() not in text:
                    issues.append(f"Sample {i}: '{value}' not found in text")
            
            # Check for missing important fields
            if not annotations.get('name'):
                issues.append(f"Sample {i}: Missing business name")
            
            if not annotations.get('mobile'):
                issues.append(f"Sample {i}: Missing mobile number")
        
        return issues

# Data augmentation functions
class SignboardDataAugmenter:
    """
    Augment training data for better model performance
    """
    
    def __init__(self):
        self.variations = {
            'phone_formats': [
                lambda x: x,  # Original
                lambda x: f"+91-{x}",
                lambda x: f"0{x[:2]}-{x[2:]}",
                lambda x: f"{x[:3]} {x[3:6]} {x[6:]}"
            ],
            'address_suffixes': [
                '', 'Road', 'Street', 'Avenue', 'Lane', 'Cross', 'Layout', 'Block'
            ],
            'business_suffixes': [
                '', 'Store', 'Shop', 'Centre', 'Center', 'Emporium', 'Mart'
            ]
        }
    
    def augment_sample(self, sample: Dict) -> List[Dict]:
        """
        Create augmented versions of a training sample
        """
        augmented = [sample]  # Include original
        
        # Phone number format variations
        if 'mobile' in sample['annotations']:
            mobile = sample['annotations']['mobile']
            clean_mobile = ''.join(filter(str.isdigit, mobile))
            
            for format_func in self.variations['phone_formats'][1:]:
                new_sample = sample.copy()
                new_mobile = format_func(clean_mobile[-10:])  # Last 10 digits
                new_sample['text'] = sample['text'].replace(mobile, new_mobile)
                new_sample['annotations'] = sample['annotations'].copy()
                new_sample['annotations']['mobile'] = new_mobile
                augmented.append(new_sample)
        
        return augmented
    
    def augment_dataset(self, data_file: str, output_file: str, multiplier: int = 3):
        """
        Augment entire dataset
        """
        with open(data_file, 'r') as f:
            original_data = json.load(f)
        
        augmented_data = []
        
        for sample in original_data:
            variants = self.augment_sample(sample)
            augmented_data.extend(variants[:multiplier])  # Limit variants
        
        with open(output_file, 'w') as f:
            json.dump(augmented_data, f, indent=2)
        
        print(f"Augmented dataset created: {output_file}")
        print(f"Original samples: {len(original_data)}, Augmented: {len(augmented_data)}")

# Complete training pipeline
class SignboardModelTrainer:
    """
    Complete pipeline for training signboard extraction model
    """
    
    def __init__(self):
        self.data_creator = SignboardTrainingDataCreator()
        self.augmenter = SignboardDataAugmenter()
    
    def prepare_training_data(self, raw_data_file: str) -> Tuple[Dict, Dict]:
        """
        Complete data preparation pipeline
        """
        print("Step 1: Validating annotations...")
        issues = self.data_creator.validate_annotations(raw_data_file)
        if issues:
            print("Annotation issues found:")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            print("Please fix these issues before training.")
            return None, None
        
        print("Step 2: Creating augmented dataset...")
        augmented_file = raw_data_file.replace('.json', '_augmented.json')
        self.augmenter.augment_dataset(raw_data_file, augmented_file)
        
        print("Step 3: Processing training data...")
        processed_data = self.data_creator.process_training_data(augmented_file)
        
        print("Step 4: Splitting into train/validation...")
        train_data, val_data = self.data_creator.export_for_training(processed_data)
        
        print(f"Training samples: {len(train_data['texts'])}")
        print(f"Validation samples: {len(val_data['texts'])}")
        
        return train_data, val_data
    
    def train_model(self, raw_data_file: str, output_model_dir: str = "./signboard_ner_model"):
        """
        Complete training pipeline
        """
        # Prepare data
        train_data, val_data = self.prepare_training_data(raw_data_file)
        
        if train_data is None:
            return None
        
        # Train model
        print("Step 5: Training NER model...")
        extractor = train_ner_model(train_data, val_data, output_model_dir)
        
        print(f"Model training completed! Saved to: {output_model_dir}")
        return extractor

# Usage examples and utilities
def create_annotation_template(image_folder: str, output_file: str):
    """
    Create annotation template for images in a folder
    """
    import os
    from pathlib import Path
    
    template_data = []
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in image_files:
        template_data.append({
            "image_path": os.path.join(image_folder, image_file),
            "text": "",  # To be filled manually after OCR
            "annotations": {
                "name": "",
                "mobile": "",
                "category": "",
                "gst": "",
                "address": "",
                "pincode": ""
            }
        })
    
    with open(output_file, 'w') as f:
        json.dump(template_data, f, indent=2)
    
    print(f"Annotation template created for {len(image_files)} images: {output_file}")

def batch_extract_text_for_annotation(image_folder: str, output_file: str):
    """
    Extract text from all images to help with annotation
    """
    import os
    from signboard_extractor import SignboardExtractor
    
    extractor = SignboardExtractor()
    annotation_data = []
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        print(f"Processing {image_file}...")
        
        # Extract text only (no parsing)
        full_text, _ = extractor.extract_text_with_positions(image_path)
        
        annotation_data.append({
            "image_path": image_path,
            "text": full_text,
            "annotations": {
                "name": "",
                "mobile": "",
                "category": "", 
                "gst": "",
                "address": "",
                "pincode": ""
            }
        })
    
    with open(output_file, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"Text extraction completed for {len(image_files)} images")
    print(f"Please manually annotate the data in: {output_file}")

def evaluate_model_performance(extractor, test_images_folder: str, ground_truth_file: str):
    """
    Evaluate model performance on test data
    """
    import os
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    results = {
        'name': {'correct': 0, 'total': 0},
        'mobile': {'correct': 0, 'total': 0},
        'category': {'correct': 0, 'total': 0},
        'gst': {'correct': 0, 'total': 0},
        'address': {'correct': 0, 'total': 0},
        'pincode': {'correct': 0, 'total': 0}
    }
    
    for item in ground_truth:
        image_path = item['image_path']
        true_annotations = item['annotations']
        
        if not os.path.exists(image_path):
            continue
        
        # Extract data
        predicted = extractor.extract_data(image_path)
        
        # Compare each field
        for field in results.keys():
            results[field]['total'] += 1
            
            true_value = true_annotations.get(field, '').strip().lower()
            pred_value = (predicted.get(field) or '').strip().lower()
            
            if true_value == pred_value:
                results[field]['correct'] += 1
    
    # Calculate accuracies
    print("Model Performance:")
    print("-" * 40)
    
    overall_correct = 0
    overall_total = 0
    
    for field, stats in results.items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"{field.title():10}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
            overall_correct += stats['correct']
            overall_total += stats['total']
    
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print("-" * 40)
    print(f"{'Overall':10}: {overall_accuracy:.1f}% ({overall_correct}/{overall_total})")

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    print("Creating sample training data...")
    creator = SignboardTrainingDataCreator()
    creator.create_sample_data('sample_signboard_data.json')
    
    # Initialize trainer
    trainer = SignboardModelTrainer()
    
    # For actual training (uncomment when you have real data):
    # trainer.train_model('your_annotated_data.json')
    
    # For creating annotation templates:
    # create_annotation_template('./signboard_images', 'annotation_template.json')
    # batch_extract_text_for_annotation('./signboard_images', 'extracted_text.json')
    
    print("\nTraining pipeline ready!")
    print("\nNext steps:")
    print("1. Collect signboard images")
    print("2. Use batch_extract_text_for_annotation() to extract text")
    print("3. Manually annotate the extracted text")
    print("4. Run trainer.train_model() with your annotated data")
    print("5. Evaluate model performance with evaluate_model_performance()")