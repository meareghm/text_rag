import json
import fitz
import openai

""" I'm using the openai version==0.28. You may have to make some changes in the script if you are using a different version.
    To install 'pip install openai==0.28'
"""

""" Replace with a valid openai apikey. Make sure to replace the term 'api_key' with the 'name' the apikey.
    You can find your details here: https://platform.openai.com/api-keys
"""
# openai.api_key = '**************'


def identify_topic(text_chunk):
    # prompt = f"Identify the topic of the following text:\n\n{text}\n\nTopic:"
    prompt = f"Classify whether the text is a machine learning review or non-review machine learning related paper. Respond in JSON format as follows: {{'classification': 1 if review else 0}}. Text:\n\n{text_chunk}\n\nTopic:"


    # response = client.chat.completions.create(
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies whether the text is a machine learning review or a non-review machine learning-related paper. Provide the result in JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7,
    )

    # topic = response.choices[0].message.content    
    topic = response['choices'][0]['message']['content'].strip()
    return topic

def identify_ml_methods(text_chunk):
    prompt = f"Extract the machine learning methods used in the following text along with a confidence score for their usage. Provide the result in the following JSON format: {{'ml_methods': [{{'method': 'method_name', 'confidence_score': confidence_value}}]}}.\n\nText:\n\n{text_chunk}"
    
    response = openai.ChatCompletion.create(
        # gpt-3.5-turbo has a maximum limit 16385 tokens
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts the machine learning methods used in the paper in JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()


def summarize_text(text_chunk):
    prompt = f"Summarize the following text:\n\n{text_chunk}\n\nSummary:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes the machine learning methods approach in the paper in JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def read_pdf(pdf_text):
    doc = fitz.open(pdf_text)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text(text, max_length=10000):
    """Split text into chunks"""
    chunks = []
    print('input text length:', len(text))
    while len(text) > max_length:
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    print('input text length:', len(text))
    return chunks

def process_pdf(pdf_path):
    """
        - read the pdf text
        - break the text into chunks texts and 
        - execute the prompt functions to process the chunk texts
    """
    text_content = read_pdf(pdf_path)
    text_chunks = split_text(text_content)
    topics = []
    ml_lists = []
    summaries = []
    
    for chunk in text_chunks:
        topic = identify_topic(chunk)
        ml_list = identify_ml_methods(chunk)
        # summary = summarize_text(chunk)
        topics.append(topic)
        ml_lists.append(ml_list)
        # summaries.append(summary)
    
    return topics, ml_lists, summaries

# Example PDF file path
pdf_path = "Machine learning in agriculture domain- A state-of-art survey.pdf"

# Process the PDF
topics, ml_lists, summaries = process_pdf(pdf_path)

# Combine and print the results. 
print("Identified Topics:")
print("\n".join(topics))
print("\n Machine Learning Methods:")
print("\n".join(ml_lists))
# print("\nSummaries:")
# print("\n".join(summaries))

