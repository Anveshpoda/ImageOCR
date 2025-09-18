from signboard_extractor import SignboardExtractor

# Initialize
extractor = SignboardExtractor()

# Extract from image
result, all_text = extractor.extract_data("../img/test_img.jpg")

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