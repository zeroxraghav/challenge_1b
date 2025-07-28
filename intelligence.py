import os
import json
import time
import sys
import fitz  # PyMuPDF
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string
import numpy as np

"""
Round 1B Solution: Persona-Driven Document Intelligence

V4 Changes:
- Selective Ranking: Instead of returning all extracted sections, the script
  now filters the results to only include the Top 20 most relevant sections,
  providing a much more focused and useful output.
- Robust Section Extraction: Uses advanced, style-based heading identification
  to correctly identify true section titles.
"""

# --- Reusable PDF Parsing Logic (Upgraded) ---

def clean_text(text):
    """Removes common artifacts and extra whitespace from extracted text."""
    text = text.strip().replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    return re.sub(r'\s+', ' ', text)

def extract_sections_from_pdf(pdf_path):
    """
    Extracts a list of structured sections from a PDF using robust style analysis.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}", file=sys.stderr)
        return []

    # 1. Style Analysis to find body text and potential heading styles
    style_counts = defaultdict(int)
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        style_key = (round(span["size"], 1), "bold" in span["font"].lower())
                        style_counts[style_key] += len(span["text"])
    
    if not style_counts:
        return []
    body_style_key = max(style_counts, key=style_counts.get)
    body_size, _ = body_style_key

    # Identify a set of true heading styles
    heading_styles = set()
    SIZE_THRESHOLD = 1.1 # Must be at least 10% larger
    for style, count in style_counts.items():
        size, is_bold = style
        if style != body_style_key:
            if size > (body_size * SIZE_THRESHOLD) or (is_bold and not body_style_key[1]):
                heading_styles.add(style)

    # 2. Extract all text blocks with style info
    all_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks", sort=True)
        for b in blocks:
            text = clean_text(b[4])
            if not text: continue
            
            block_dict = page.get_text("dict", clip=b[:4])
            if not block_dict.get('blocks') or not block_dict['blocks'][0].get('lines') or not block_dict['blocks'][0]['lines'][0].get('spans'):
                continue
            span = block_dict['blocks'][0]['lines'][0]['spans'][0]
            style_key = (round(span["size"], 1), "bold" in span["font"].lower())
            all_blocks.append({"text": text, "style": style_key, "page": page_num})

    # 3. Group text into sections based on identified headings
    sections = []
    current_heading_block = None
    current_text = ""
    
    for block in all_blocks:
        is_heading = block["style"] in heading_styles

        if is_heading:
            if current_heading_block:
                sections.append({
                    "title": current_heading_block["text"],
                    "content": current_text.strip(),
                    "page": current_heading_block["page"]
                })
            current_heading_block = block
            current_text = ""
        elif current_heading_block:
            current_text += " " + block["text"]
            
    if current_heading_block:
        sections.append({
            "title": current_heading_block["text"],
            "content": current_text.strip(),
            "page": current_heading_block["page"]
        })
        
    return sections

# --- Text Preprocessing & Summarization Logic ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK resources ('stopwords', 'punkt')...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Lowercases, removes punctuation and stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def create_smart_summary(text, vectorizer, doc_vector):
    """Creates a summary by scoring and selecting the most important sentences."""
    if not text:
        return ""
    
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        return " ".join(text.split()[:50])

    if not sentences:
        return ""

    sentence_scores = []
    feature_names = vectorizer.get_feature_names_out()
    
    for sentence in sentences:
        score = 0
        word_count = 0
        processed_sentence = preprocess_text(sentence)
        for word in processed_sentence.split():
            if word in feature_names:
                word_index = vectorizer.vocabulary_.get(word)
                if word_index is not None:
                    score += doc_vector[0, word_index]
                    word_count += 1
        sentence_scores.append(score / (word_count + 1e-5))

    top_sentence_indices = np.argsort(sentence_scores)[-2:]
    top_sentence_indices = sorted(top_sentence_indices)
    
    summary = " ".join([sentences[i] for i in top_sentence_indices])
    return summary


# --- Main Processing Function for a Single Collection ---

def process_collection(collection_path):
    """
    Processes a single collection directory (e.g., "Collection 1/").
    """
    print(f"--- Processing Collection: {os.path.basename(collection_path)} ---")
    input_json_path = os.path.join(collection_path, 'challenge1b_input.json')
    pdfs_dir = os.path.join(collection_path, 'PDFs')

    if not os.path.exists(input_json_path):
        print(f"Warning: Skipping collection. No 'challenge1b_input.json' found in {collection_path}")
        return

    # 1. Load Input Config
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona = input_data.get("persona", {}).get("role", "")
    job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
    query = f"{persona} {job_to_be_done}"
    
    doc_filenames = [doc['filename'] for doc in input_data.get("documents", [])]

    # 2. Extract Sections from all PDFs in the collection
    all_sections = []
    for filename in doc_filenames:
        pdf_path = os.path.join(pdfs_dir, filename)
        if os.path.exists(pdf_path):
            doc_sections = extract_sections_from_pdf(pdf_path)
            for section in doc_sections:
                section['document'] = filename
            all_sections.extend(doc_sections)
        else:
            print(f"Warning: PDF file {filename} not found in {pdfs_dir}")

    if not all_sections:
        print("Warning: Could not extract any sections from the PDFs in this collection.")
        return

    # 3. Calculate Relevance using TF-IDF
    section_contents = [s['content'] for s in all_sections]
    processed_contents = [preprocess_text(text) for text in section_contents]
    processed_query = preprocess_text(query)

    vectorizer = TfidfVectorizer()
    corpus_vectors = vectorizer.fit_transform(processed_contents)
    query_vector = vectorizer.transform([processed_query])

    similarity_scores = cosine_similarity(query_vector, corpus_vectors)[0]

    for i, section in enumerate(all_sections):
        section['relevance_score'] = similarity_scores[i]
        section['summary'] = create_smart_summary(section['content'], vectorizer, corpus_vectors[i])

    # --- NEW: Filter for only the most relevant sections ---
    ranked_sections = sorted(all_sections, key=lambda x: x['relevance_score'], reverse=True)
    top_sections = ranked_sections[:20] # Keep only the top 20 most relevant sections

    # 4. Format Output JSON
    output_data = {
        "metadata": {
            "input_documents": doc_filenames,
            "persona": persona,
            "job_to_be_done": job_to_be_done
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for i, section in enumerate(top_sections): # Use the filtered list
        output_data["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["title"],
            "importance_rank": i + 1,
            "page_number": section["page"]
        })
        output_data["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": section["summary"],
            "page_number": section["page"]
        })

    # 5. Write Output
    output_path = os.path.join(collection_path, 'challenge1b_output.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully generated output for {os.path.basename(collection_path)} at {output_path}\n")


# --- Main Execution ---
if __name__ == '__main__':
    root_dir = "." 
    collection_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("Collection")]
    
    if not collection_dirs:
        print("No 'Collection X' directories found. Please check your project structure.")
        sys.exit(1)

    for collection in sorted(collection_dirs):
        process_collection(os.path.join(root_dir, collection))
