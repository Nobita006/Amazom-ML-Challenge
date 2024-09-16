from tensorflow.keras.models import load_model

# Load the model
model = load_model('ocr_model50k-20.h5')

# Save the model in the SavedModel format
model.save('ocr_model', save_format='tf')
