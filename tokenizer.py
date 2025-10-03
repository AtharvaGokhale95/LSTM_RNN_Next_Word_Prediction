from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from data_loader import load_data
import numpy as np
import tensorflow as tf

# # load the text from the file
# file_path = load_data()
# with open(file_path, 'r') as file:
#     text = file.read().lower()
    
def load_and_prepare_text():
    """
    Load raw text using load_data() and return lowercase text.
    """
    file_path = load_data()
    with open(file_path, 'r') as file:
        text = file.read().lower()
    return text

# # Tokenization - Creating numerical indexes for each word
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts([text])
# total_words = len(tokenizer.word_index) + 1
# print(f"Total no of distinct words in the corpus are {total_words} ")

def build_tokenizer(text):
    """
    Create and fit tokenizer on the given text.
    Returns tokenizer object and total number of words.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    print(f"Total no. of distinct words in the corpus: {total_words}")
    return tokenizer, total_words

# Create input training sequences:
'''
E.g.: "I love cat, and her colour is white" -- This is one sentence from the corpus 
Following are the n-gram sequences created for the sentence to create the training sequence. We need to start at least with a pair of words.
input sequence 1: "I love"
input sequence 2: "I love cat"
input sequence 3: "I love cat and"
input sequence 4: "I love cat and her"
input sequence 5: "I love cat and her color"
input sequence 6: "I love cat and her color is"
input sequence 7: "I love cat and her color is white"

For loop logic:
1. Loop over each sentence separated by a new line 
2. Convert that sentence to a list of numerical tokens - Only for the words that are available in the vocabulary (text) using the same tokenizer object used above
3. texts_to_sequences expects an iterable of texts so we need to pass a list and not a string
4. Then we choose the 0th element to only store the tokens
5. We have a list of tokens representing a complete sentence stores in token_list
6. Iterate over this token_list to form the input sequences. Start from 1st index as the 0th index is required to form a pair of at least 2 tokens every time
8. Range: 1 - length of the sentence
9. When i = 1:
    n_gram_sequences = token_list[0:2] -> 
10. So now, for a sentence with 5 words, we have 4 vectors with (2, 3, 4, 5) words
'''
# input_sequences = []
# for line in text.split('\n'):
#     token_list = tokenizer.texts_to_sequences([line])[0]
#     for i in range(1, len(token_list)):
#         n_gram_sequence = token_list[0 : i + 1]
#         input_sequences.append(n_gram_sequence)
        
def create_input_sequences(text, tokenizer):
    """
    Create n-gram sequences for each sentence in the corpus.
    """
    input_sequences = []

    for line in text.split('\n'):
        # Convert line into a list of tokens
        token_list = tokenizer.texts_to_sequences([line])[0]

        # Create n-gram sequences from tokens
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[0 : i + 1]
            input_sequences.append(n_gram_sequence)

    return input_sequences
        
# # Padding the sequences to have a uniform length:
# max_sequence_len = max([len(x) for x in input_sequences]) # Determines the max length of a sentence
# print(f"The maximum length of a sentence in the corpus is {max_sequence_len}")

# # Making all the sentences in the corpus of the same length as the max length:

# # Currently input_sequences is a array of arrays - Now we make it a numpy array
# input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_len, padding = 'pre')) # Now every input sequence will be of the same length,
# # padded with zeros at the start

def pad_sequences_and_split(input_sequences):
    """
    Pad all sequences to the same length.
    Returns padded numpy array and max sequence length.
    """
    max_sequence_len = max([len(x) for x in input_sequences])
    print(f"🧩 The maximum length of a sentence in the corpus is {max_sequence_len}")

    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    )
    return input_sequences, max_sequence_len


# # Creating Predictors/ Label split:
# # Training Data/ Predictors (X): Dependent features, input sequence you feed into the model (All the words in the input_sequence, but the last word)
# # Label (y): Target word that the model tries to predict (Last word in the input sequence)
# # Each training example is a sequence of tokens (words), and the label is the next word that should follow that sequence

# X, y = input_sequences[ : , : -1], input_sequences[ : , -1] # First : for rows, second : for columns

# # Convert Labels to Categorical data: One Hot encoding
# # The vector length will be equal to the count of total # of unique words in the corpus (4818)
# # Only the index of the label will be 1 and the rest will be zero
# y = tf.keras.utils.to_categorical(y, num_classes = total_words)
# # Why we have to convert the numerical value for y to categorical value:
# # 1. Neural networks usually cannot predict an integer directly for a classification problem like this as the model outputs a probability distribution over all words using softmax
# # 2. To train the model with categorical_crossentropy loss, the labels must be one-hot encoded
# # 3. If you use sparse_categorical_crossentropy, we can use numerical values for y - Many people do this to save memory, especially with large vocabularies


def split_predictors_labels(input_sequences, total_words, one_hot=True):
    """
    Split into predictors (X) and labels (y).
    Optionally convert labels to one-hot encoded vectors.
    """
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    if one_hot:
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        print("✅ Labels converted to one-hot encoding")

    return X, y


# # Train_Test split:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

def split_train_test(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test



# Driver function:

def main():
    # 1️⃣ Load and clean text
    text = load_and_prepare_text()

    # 2️⃣ Build tokenizer
    tokenizer, total_words = build_tokenizer(text)

    # 3️⃣ Create input sequences
    input_sequences = create_input_sequences(text, tokenizer)

    # 4️⃣ Pad sequences
    input_sequences, max_sequence_len = pad_sequences_and_split(input_sequences)

    # 5️⃣ Split into predictors & labels
    X, y = split_predictors_labels(input_sequences, total_words, one_hot=True)

    # 6️⃣ Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("✅ Preprocessing complete!")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Max sequence length: {max_sequence_len}")
    print(f"Total words in corpus: {total_words}")


if __name__ == "__main__":
    main() 
