# NCKU Project 2 — Zipf Analysis & Query Expansion Models

Biomedical Information Retrieval Project 2 using 1,000 English GLP-1 scientific abstracts from PubMed / PMC JATS.

## Main Functions

- Text preprocessing
- Collection Frequency (CF) and Document Frequency (DF)
- IDF / TF-IDF
- Zipf distribution and log-log regression
- A/B/C/D preprocessing comparison
- Porter stemming
- Word2Vec Skip-gram
- Spelling correction using Edit Distance
- Semantic query expansion
- Keyword retrieval
- Significant Words visualization
- Streamlit interactive interface

## Corpus

- 1,000 scientific abstracts
- 1,000 unique PMIDs
- Domain: GLP-1
- Language: English
- Main abstract body only

Prepared corpus:

prepared_data/glp1_1000_abstracts.jsonl.gz

Raw XML files are not included in this repository.

## Preprocessing Conditions

A Basic: whitespace tokenization + case folding

B Remove punctuation: punctuation-aware tokenization + case folding

C Remove stopwords: B + English stopword removal

D Porter stemming: C + Porter stemming

Pipeline:

A -> B -> C -> D

## Run

Install dependencies:

pip install -r requirements.txt

Run the final application:

streamlit run src/m09_app_v3.py

## Prepared Artifacts

- artifacts/project2_runtime.pkl.gz
- artifacts/project2_analysis.json.gz
- artifacts/glp1_word2vec.model
- artifacts/project2_manifest.json

Prepared artifacts are included so the final system can run without rebuilding the full corpus.

## Word2Vec

- Architecture: Skip-gram
- Vector size: 100
- Window: 5
- Minimum count: 2
- Epochs: 20
- Seed: 42

## Validation

Run:

python src/project2_core_selftest.py

Final verified result:

Project 2 V3 core self-test: PASS

## Course

National Cheng Kung University  
Artificial Intelligence Information Retrieval  
Project 2 — Zipf Analysis & Query Expansion Models
