# PDF Document Structure and Hierarchy Extraction

**A solution for the Adobe India Hackathon 2025: "Connecting the Dots" Challenge (Round 1A)**

This project presents a robust, multi-stage pipeline designed to parse PDF documents, understand their structure, and extract a hierarchical outline (Title, H1, H2, H3) in a structured JSON format. The solution is built to be fast, accurate, and resilient to the complex and inconsistent layouts often found in real-world PDFs.

## 1. Problem Statement

The core challenge of Round 1A is to build an automated system that can analyze a given PDF file and produce a structured, hierarchical outline.

**Mission:**
> You're handed a PDF, but instead of simply reading it, you're tasked with making sense of it like a machine would. Your job is to extract a structured outline of the document—essentially the Title, and headings like H1, H2, and H3—in a clean, hierarchical format.

The solution must adhere to strict performance and resource constraints, including a 10-second execution time for a 50-page document, a model size limit of under 200MB, and fully offline execution within a Docker container.

## 2. Our Approach: A Hybrid AI & Algorithmic Pipeline

To tackle the complexities of PDF structure analysis, we developed a sophisticated hybrid pipeline that combines the speed of low-level parsing, the intelligence of machine learning, and the logic of deterministic algorithms. This approach ensures both high accuracy and compliance with the hackathon's performance constraints.

Our pipeline is divided into four main stages:

### Stage 1: High-Fidelity Text & Layout Extraction
The foundation of our pipeline is accurate data extraction. We employ a multi-faceted approach to capture both the content and the layout with high precision.

-   **Advanced Markdown Conversion with `pymupdf4llm`**: We use the state-of-the-art `pymupdf4llm` library to convert the PDF into a structured Markdown format. This tool is specifically designed for Large Language Model consumption, providing a clean, semantic representation of the document's content. For robustness, if `pymupdf4llm` encounters issues, the system automatically falls back to a standard `PyMuPDF` text extraction method.
-   **Parallel Span & Column Extraction**: Simultaneously, we use `PyMuPDF` to extract detailed span-level information, including **precise bounding box coordinates**, **font details** (name, size, weight), and **page numbers**. Our custom logic also detects multi-column layouts and table structures, ensuring text reading order is preserved.
-   **Intelligent Header/Footer Removal**: Before further processing, we run a custom algorithm that analyzes repeating text patterns across the top and bottom margins of pages. This effectively identifies and removes recurring headers and footers, preventing them from being mistaken for document content.
-   **Multilingual Document Support**: The pipeline features comprehensive multilingual support, automatically detecting document languages and applying appropriate Unicode normalization and text processing techniques. This enables accurate hierarchy extraction from PDFs in any language, including complex scripts like Arabic, Chinese, and Devanagari, making the solution globally applicable across diverse document types and languages.

### Stage 2: Textline Merging (ML Model 1)

**Custom Dataset Creation & Training**:
To ensure robust performance across diverse document types, we created our own comprehensive training dataset from scratch. We manually annotated 60+ diverse PDF documents spanning academic papers, technical reports, business documents, and research publications. This meticulous process generated approximately 18,000 labeled textline pairs, creating one of the most comprehensive datasets for PDF textline merging tasks.

Text lines are merged into coherent blocks (like paragraphs or multi-line headings) using a binary classification model. For any two consecutive lines, a **Random Forest Classifier** predicts whether they should be merged (`block_in`) or kept separate (`block_start`). The model is trained on a rich set of features(custom datasets made on our own), including:
-   **Layout Features**: `normalized_vertical_gap`, `indentation_change`, `same_alignment`, `is_centered_A`, `is_centered_B`.
-   **Font Features**: `font_size_a`, `font_size_b`, `font_size_diff`, `same_font`, `is_bold_A`, `is_bold_B`, `is_italic_A`, `is_italic_B`, `is_monospace_A`, `is_monospace_B`, `same_bold`, `same_italic`, `same_monospace`.
-   **Textual Features**: `line_a_ends_punctuation`, `line_b_starts_lowercase`, `line_length_ratio`.
-   **Structural Features**: `is_linea_in_rectangle`, `is_lineb_in_rectangle`, `both_in_table`, `neither_in_table`, `is_linea_hashed`, `is_lineb_hashed`, `both_hashed`, `neither_hashed`.

**Model Performance**: Our trained Random Forest Classifier achieved >96% accuracy with exceptional recall performance, ensuring minimal loss of important textual content during the merging process.

### Stage 3: Title vs. Text Classification (ML Model 2)

**Advanced Dataset Engineering & Model Performance**:
Building on our comprehensive dataset creation approach, we extended our manual annotation to cover textblock classification. Using the same 60+ documents, we generated additional training samples for title/text classification, resulting in a robust training corpus that captures the nuanced differences between headings and body text across various document styles and domains.

Once textblocks are formed, a second machine learning stage classifies each block as either a **`title`** or **`text`**. This stage is designed to be exceptionally robust.

-   **Competitive Model Evaluation**: Instead of relying on a single model, we train and evaluate three different classifiers: a **Decision Tree**, a **Gradient Boosting Classifier**, and an **SGD Classifier**. We select the model that achieves the highest **recall** for the `title` class. This strategy ensures we minimize false negatives, capturing as many potential headings as possible for the final hierarchy-building stage.
- **Advanced Feature Engineering**: The models are trained on a powerful, multi-modal feature vector that combines layout, structural, and syntactic information. The key features include:
    -   **Layout & Style Features**: `avg_font_size`, `relative_font_size` (compared to page median), `is_bold`, `space_above`.
    -   **Textual Content Features**: `word_count`, `char_count`, `is_all_caps`, `is_title_case`, `ends_with_colon`, `starts_with_list_pattern`.
    -   **Syntactic (POS) Features**: `noun_count`, `verb_count`, `adj_count`, `cardinal_num_count`, `noun_ratio`, `verb_ratio`.
    -   **Interaction Feature**: `caps_x_font` (an interaction between `is_all_caps` and `relative_font_size` to give more weight to large, capitalized text).
-   **Post-Processing Rules**: After prediction, we apply a set of heuristic rules to correct common misclassifications. For example, a block predicted as a "title" that starts with a lowercase letter or ends with a period is automatically re-classified as "text". This significantly improves the final precision of our results.

**Model Performance**: Our best-performing classifier achieved >96% accuracy with outstanding recall scores, ensuring that virtually all document headings are correctly identified while maintaining high precision to minimize false positives.

#### Handling Data Imbalance with SMOTE
During model training, we addressed the significant class imbalance inherent in documents (far more text paragraphs than headings). We employed the **Synthetic Minority Over-sampling Technique (SMOTE)**, as detailed in related research. This technique generates synthetic examples of the minority class (headings), preventing the model from becoming biased towards the majority class and significantly improving its ability to correctly identify headings.

### Stage 4: Deterministic Hierarchy Ranking
After identifying all titles, we assign their hierarchical levels (H1, H2, H3). A simple ML model is insufficient for this task, as a title's rank is relative to its context. We instead use a **deterministic, stateful algorithm** that mimics human logic:
1.  **Style Clustering:** Titles with similar visual styles (font size, weight, indentation) are grouped using a **KMeans clustering** algorithm.
2.  **Numbering Recognition:** Regular expressions detect explicit numbering schemes (`1.`, `1.2`, `A.`, `Appendix C:`), which provide strong, unambiguous evidence of hierarchy.
3.  **Tree-Building Algorithm:** The algorithm processes titles sequentially, using a **stack** to maintain the current path in the document's hierarchy. It uses a prioritized set of rules (numbering first, then style) to determine if a title is a child, sibling, or a promotion to a higher level, correctly reconstructing the document's logical tree.

## 3. Project Directory Structure

The project is organized into a modular structure to separate concerns, making it easy to manage and extend.

```
Adobe_round_1_A/
├── app/
│   ├── extractor/      # PDF parsing and raw textline extraction logic.
│   ├── merging/        # Logic for merging textlines into textblocks.
│   ├── models/         # Stores trained ML models (.joblib files).
│   └── models_code/    # Scripts for ML model testing and hierarchy analysis.
├── input/              # Input directory for user-provided PDFs.
├── output/             # Output directory for final JSON results.
├── .dockerignore       # Specifies files to ignore in the Docker build.
├── .gitignore          # Specifies files to ignore for Git.
├── complete_pipeline.py# Main script that orchestrates the entire workflow.
├── docker_runner.py    # Entrypoint script for the Docker container.
├── Dockerfile          # Defines the Docker image for the application.
├── README.md           # This documentation file.
└── requirements.txt    # Lists all Python dependencies.
```

## 4. Tech Stack & Libraries

-   **Language:** Python 3.11
-   **Core Libraries:**
    -   `pymupdf4llm` & `PyMuPDF`: For advanced and standard PDF parsing.
    -   `scikit-learn`: For building and using our machine learning models (Random Forest, KMeans, etc.).
    -   `pandas` & `numpy`: For efficient data manipulation and numerical operations.
    -   `joblib`: For serializing and loading our trained ML models.
    -   `nltk`: For extracting syntactic features (Parts-of-Speech).
    - For other libraries see requirements.txt
-   **Deployment:** Docker

## 5. Setup and Execution

The entire solution is packaged in a Docker container for easy and consistent execution, as required by the hackathon.

### Prerequisites
-   Docker must be installed and running on your system.

### Step 1: Build the Docker Image
Navigate to the project's root directory and run the following command to build the image. This command will install all dependencies inside the container.

```bash
docker build --platform linux/amd64 -t pdf-hierarchy-extractor .
```

### Step 2: Run the Solution
Place all the PDF files you want to process into the `input` directory. Then, run the following command. It will automatically process every PDF in the `input` folder and generate a corresponding `.json` file in the `output` folder.

```bash
docker run --rm \
  -v "$(pwd)/input":/app/input \
  -v "$(pwd)/output":/app/output \
  --network none \
  pdf-hierarchy-extractor
```
-   `-v "$(pwd)/input":/app/input`: Mounts your local `input` folder into the container.
-   `-v "$(pwd)/output":/app/output`: Mounts your local `output` folder for the results.
-   `--network none`: Ensures the solution runs completely offline.

## 6. Pipeline Workflow Overview

The `docker_runner.py` script executes the following end-to-end pipeline defined in `complete_pipeline.py`:

1.  **PDF Extraction**: PDFs from `/app/input` are processed by the `extractor` to produce structured textline data.
2.  **Textline Merging**: The first ML model predicts which consecutive textlines should be merged. The `merging` script then combines them into textblocks.
3.  **Textblock Classification**: The second ML model classifies each textblock as a `title` or `text`.
4.  **Hierarchy Analysis**: The deterministic algorithm processes the identified titles to assign hierarchical levels (Title, H1, H2, H3).
5.  **JSON Output**: The final hierarchy is formatted into the required JSON structure and saved to the `/app/output` directory.

## 7. Final Output Format

For each `filename.pdf` in the input directory, the solution generates a `filename.json` in the output directory with the following structure:

```json
{
  "title": "The Main Title of the Document",
  "outline": [
    {
      "level": "H1",
      "text": "This is a Level 1 Heading",
      "page": 1
    },
    {
      "level": "H2",
      "text": "This is a subsection under H1",
      "page": 2
    },
    {
      "level": "H3",
      "text": "This is a deeper subsection",
      "page": 2
    },
    {
      "level": "H1",
      "text": "This is another Level 1 Heading",
      "page": 3
    }
  ]
}
```
## 8. References

[1] Budhiraja, S. S., & Mago, V. (2018). *A Supervised Learning Approach For Heading Detection*. [Preprint]. ResearchGate. Retrieved from https://www.researchgate.net/publication/327405802_A_Supervised_Learning_Approach_For_Heading_Detection

[2] POSOS. (n.d.). *How to extract and structure text from PDF files with Python and Machine Learning*. POSOS Blog. Retrieved July 27, 2025, from https://www.posos.co/blog-articles/how-to-extract-and-structure-text-from-pdf-files-with-python-and-machine-learning

[3] *PyMuPDF Documentation*. (n.d.). Retrieved July 27, 2025, from https://pymupdf.readthedocs.io/en/latest/