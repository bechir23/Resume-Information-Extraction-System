# Resume Information Extraction System

A robust resume parsing and information extraction system using Rasa conversational AI and spaCy NER, with custom regex rules and advanced data augmentation for high-accuracy entity extraction.

---

## Features

- **Dual-phase extraction:** Combines spaCy NER models with custom regex rules.
- **Interactive correction:** Users can modify extracted information via natural language.
- **Automatic ranking:** Scores and ranks resumes against job requirements.
- **Indian context optimized:** Special handling for Indian names, companies, and locations.

---

## Project Structure

```
rasa_bot/
├── actions/
├── custom_components/
├── data/
├── models/
├── Resume-Information-Extraction-System/
├── web/
├── config.yml
├── credentials.yml
├── domain.yml
├── endpoints.yml
├── Dockerfile
├── docker-compose.yml
└── test.py
```

---

## Quickstart

### 1. Install Dependencies

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_md
pip install spacy-transformers
pip install rasa
```

### 2. Prepare Data

- Place your resume files (PDF, DOCX, images) in `resume_system_data/raw_data/`.
- Run the notebook or scripts to parse and preprocess resumes.

### 3. Train the NER Model

#### Tok2Vec Model

```shell
python -m spacy train config_tok2vec.cfg --output ./models/tok2vec --gpu-id 0
```

#### Transformer Model

```shell
python -m spacy train config_transformer_jupyter.cfg --output ./models/transformer --gpu-id 0
```

### 4. Evaluate Model

```shell
python -m spacy benchmark accuracy models/tok2vec/model-best dev.spacy --output metrics.json --gpu-id 0
```

---

## Custom Rasa Component

To use the trained spaCy NER model in Rasa, add the custom component in your pipeline:

```yaml
pipeline:
  - name: WhitespaceTokenizer
  - name: custom_components.ResumeNERComponent
    model_path: "./models/model-best"
```

---

## Data Augmentation

To improve recall and precision, run the augmentation pipeline:

```python
# In your notebook or script
augmented_result = create_balanced_augmentation(result_dict)
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

---

## References

- [spaCy Documentation](https://spacy.io/)
- [Rasa Documentation](https://rasa.com/docs/)
- [Project Paper/Blog](#) <!-- Add link if available -->

---

## License

MIT License

---
