# Resume Information Extraction System

A robust resume parsing and information extraction system using spaCy NER, advanced data analysis, visualization, and data augmentation, with Rasa conversational AI for interactive correction and ranking.

---

## Features

- **Dual-phase extraction:** Combines spaCy NER models with custom regex rules.
- **Interactive correction:** Users can modify extracted information via natural language in the web UI.
- **Automatic ranking:** Scores and ranks resumes against job requirements.
- **Context optimized:** Special handling for names, companies, and locations.
- **Data analysis & visualization:** Built-in tools for entity statistics, pattern analysis, and visualizations.
- **Advanced augmentation:** Targeted and balanced augmentation pipeline to boost recall and precision.

---

## Project Structure

```
rasa_bot/
├── actions/
├── custom_components/
├── data/
├── models/
├── nlp_utils/
├── web/
├── config.yml
├── credentials.yml
├── domain.yml
├── endpoints.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run_nlp_pipeline.sh
└── test.py
```

---

## Quickstart

### 1. Install Python Dependencies (for NER/augmentation/analysis)

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_md
pip install spacy-transformers
# Do NOT install rasa here; it's handled by Docker!
```

### 2. Prepare Data

- Place your resume files (PDF, DOCX, images) in `resume_system_data/raw_data/`.
- Run the notebook (`nlp-data.ipynb`) or scripts in `nlp_utils/` to parse, clean, and preprocess resumes.

### 3. Data Analysis & Visualization

- Use the notebook or `nlp_utils/entity_analysis.py` to analyze entity statistics, visualize entity lengths, and explore entity patterns.
- Example:
```python
from nlp_utils.entity_analysis import analyze_entity_statistics, visualize_entity_lengths
stats = analyze_entity_statistics(all_data)
visualize_entity_lengths(stats)
```
- Visualizations and co-occurrence graphs are generated and saved as PNGs for easy inspection.

### 4. Data Augmentation

- Use the augmentation pipeline to balance and enrich your NER data:
```python
from nlp_utils.augmentation import create_balanced_augmentation
augmented_result = create_balanced_augmentation(result_dict)
```
- This will generate new `.spacy` files and JSONs for training.

### 5. Train the NER Model

#### Tok2Vec Model

```shell
python -m spacy train models/resume_model/config_tok2vec.cfg --output models/tok2vec --gpu-id 0
```

#### Transformer Model

```shell
python -m spacy train models/resume_model/config_transformer.cfg --output models/transformer --gpu-id 0
```

### 6. Evaluate Model

```shell
python -m spacy benchmark accuracy models/tok2vec/model-best dev.spacy --output metrics.json --gpu-id 0
```

### 7. Build and Start Rasa (Docker)

```shell
docker build -t custom-rasa-with-spacy .
docker-compose run --rm rasa train
docker-compose build web
docker-compose up
```

---

## Custom Rasa Component

To use the trained spaCy NER model in Rasa, add the custom component in your pipeline (`config.yml`):

```yaml
pipeline:
  - name: WhitespaceTokenizer
  - name: custom_components.resume_ner.ResumeNER
    model_path: "./models/model-best"
```

---

## Example Usage

```python
import spacy

nlp = spacy.load("models/model-best")
doc = nlp("Python is a great programming language used in AI by Google in California.")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

---

## Tips

- For best results, use resumes in English and ensure clear section headers.
- The system is optimized for Indian names, companies, and locations.
- Use the web interface in `web/` for interactive correction and feedback.
- All data cleaning, augmentation, and analysis functions are available in `nlp_utils/` for reuse.

---

## Visual Demo

![Screenshot 2025-04-23 200226](https://github.com/user-attachments/assets/525d0ad2-d371-432a-b404-fdd20bf37f91)


---

## References

- [spaCy Documentation](https://spacy.io/)
- [Rasa Documentation](https://rasa.com/docs/)
- [Project Paper/Blog](#) <!-- Add link if available -->

---

## License

MIT License

---
