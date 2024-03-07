import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your pre-trained model
model = tf.keras.models.load_model('path/to/your/model.h5')

# Set the maximum sequence length used during training
max_length = 100

# Function to preprocess and predict denial code
def predict_denial_code(reason):
  tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>")  # Replace with your tokenizer settings
  tokenizer.fit_on_texts([reason])  # Fit tokenizer only on the current reason, not new data

  reason_sequence = tokenizer.texts_to_sequences([reason])
  reason_padded = pad_sequences(reason_sequence, maxlen=max_length, padding='post')

  prediction = model.predict(reason_padded)[0]
  predicted_class_index = np.argmax(prediction)

  predicted_code = list(denial_codes.keys())[predicted_class_index]

  return predicted_code

# Example usage
new_reason = "Provider not elig. to perform service billed"
predicted_code = predict_denial_code(new_reason)

print("Predicted Denial Code:", predicted_code)
