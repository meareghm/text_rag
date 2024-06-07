import json
import fitz
import openai

""" Used the openai version==0.28. You may have to make some changes in the script if you are using a different version.
    To install 'pip install openai==0.28'
"""

""" Replace with a valid openai apikey. Make sure to replace the term 'api_key' with the 'name' the apikey.
    You can find your details here: https://platform.openai.com/api-keys
"""
# Replace with a valid OpenAI's API 'key name' and corresponding 'API key'.
# openai.api_key_name = '**************'


def classify_text(text):
    """
    Given a string of text, this function identifies whether the text is a review or not.
    It returns a JSON string indicating the classification.
    """
    prompt = f"Classify whether the text is a machine learning review or non-review machine learning related paper. Respond in JSON format as follows: {{'isReview': 1 if review else 0}}. Text:\n\n{text}\n\nisReview"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies whether the text is a machine learning review or a non-review machine learning-related paper. Provide the result in JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0,
    )

    return response['choices'][0]['message']['content'].strip()

def identify_ml_methods(text_chunk):
    """
    Extracts machine learning methods used in the given text along with a confidence score.
    Returns the result in JSON format.
    """

    prompt = f"Extract the machine learning methods used in the following text along with a confidence score for their usage. Provide the result in the following JSON format: {{'ml_methods': [{{'method': 'method_name', 'confidence_score': confidence_value}}]}}.\n\nText:\n\n{text_chunk}\n\ReviewClassification:"
    
    response = openai.ChatCompletion.create(
        # gpt-3.5-turbo has a maximum limit 16385 tokens
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts the machine learning methods used in the paper in JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0,
    )
    return response['choices'][0]['message']['content'].strip()


def summarize_text(text_chunk):
    """
    Summarizes the given text chunk using the GPT-3.5-turbo model.
    """

    prompt = f"Summarize the following text:\n\n{text_chunk}\n\nSummary:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes the machine learning methods approach in the paper in JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0,
    )
    
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def read_pdf(pdf_path):
    """Read the text content of a PDF file."""
    with fitz.open(pdf_path) as pdf:
        text = ''.join(page.get_text() for page in pdf)
    return text

def read_pdf(pdf_file):

    """
    Read the text content of a PDF file.
    """

    with fitz.open(pdf_file) as pdf_text:
        text = ''.join(page.get_text() for page in pdf_text)
    #len(text))
    return text

def split_text_into_chunks(text, chunk_size=10000):

    """
    Split text into chunks of maximum length.
    """

    chunks = []
    while text:
        chunk = text[:chunk_size].rsplit(' ', 1)
        chunks.append(chunk)
        text = text[chunk_size:]
    return chunks

def main(pdf_file):

    """
    Read the PDF text, split it into chunks, and process each chunk.
    """

    # Read the PDF text
    text_content = read_pdf(pdf_file)
    
    # Break the text into chunks
    text_chunks = split_text_into_chunks(text_content)
    
    # Initialize lists to store the results
    classifications, ml_lists, summaries = [], [], []

    # Process each chunk
    for chunk in text_chunks:
        # Identify where the chunk text is review type or not 
        classification = classify_text(chunk)
        
        # Identify the machine learning methods in the chunk
        ml_list = identify_ml_methods(chunk)
        
        # Uncomment the following line to generate a summary for the chunk
        # summary = summarize_text(chunk)
        
        # Append the results to the lists
        classifications.append(classification)
        ml_lists.append(ml_list)
        # summaries.append(summary)
    
    return classifications, ml_lists #, summaries

if __name__ == '__main__':
    # Example pdf file 
    pdf_file = 'Machine learning in agriculture domain- A state-of-art survey.pdf'
    classifications, ml_lists = main(pdf_file)

    # To generate summaries Also make sure to uncomment relevant lines in the main function. 
    # topics, ml_lists, summaries = main(pdf_file)

    
    # Concatenate the lists and print them. 
    print("Paper Type:")
    print("\n".join(classifications))
    print("\n Machine Learning Methods:")
    print("\n".join(ml_lists))
    #print("\nSummaries:")
    #print("\n".join(summaries))