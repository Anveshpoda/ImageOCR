from signboard_extractor import SignboardExtractor

# Initialize
extractor = SignboardExtractor()

# Extract from image
result = extractor.extract_data("../test_img.jpg")
print(result)