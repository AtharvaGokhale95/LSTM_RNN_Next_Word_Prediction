import nltk
import os
from nltk.corpus import gutenberg

   
def load_data(file_id = 'shakespeare-hamlet.txt', output_folder = 'data', output_file = 'hamlet.txt'):
    """
    Load text from NLTK Gutenberg corpus and save it to a local file.
    
    Parameters:
        file_id (str): Name of the file in the Gutenberg corpus.
        output_file (str): Name of the local file to save the text.
        
    Returns:
        str: The file name in which the text is stored
    """
    # Load raw text from Gutenberg
    data = gutenberg.raw(file_id)
    
    # Ensure folder exists to store the file
    os.makedirs(output_folder, exist_ok = True)
    
    # Full file path
    file_path = os.path.join(output_folder, output_file)
    
    # Write the text into a local file
    with open(file_path, 'w') as file:
        file.write(data)

    return file_path

# Driver code
if __name__ == "__main__":
    file_path = load_data()
    print(f"Data saved to: {file_path}")

