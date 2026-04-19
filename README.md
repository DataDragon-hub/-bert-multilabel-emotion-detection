# BERT-based Multi-Label Emotion Detection (PyTorch, Transformers)

Fine-tuned a BERT model for multi-label emotion classification to detect multiple emotions (anger, fear, joy, sadness, surprise) from text data.

---

## 🚀 Overview

This project focuses on **multi-label emotion detection**, where each text input can express more than one emotion simultaneously.
Unlike traditional sentiment analysis, this approach captures the complexity of human emotions in text.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers (BERT)
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn

---

## 📊 Dataset

* ~2,700+ English text samples
* Each sample labeled with **5 emotions**:

  * Anger
  * Fear
  * Joy
  * Sadness
  * Surprise
* Multi-label format (each sample can have multiple emotions)

---

## 🏗️ Model Architecture

* Pre-trained **BERT (bert-base-uncased)**
* Fine-tuned using:

  * `BertForSequenceClassification`
  * `problem_type="multi_label_classification"`
* Output layer with **sigmoid activation** for multi-label prediction

---

## ⚙️ Training Details

* Optimizer: AdamW
* Learning Rate: 2e-5
* Epochs: 4
* Loss Function: Binary Cross Entropy (BCEWithLogitsLoss)
* Max Sequence Length: 128
* Train/Validation Split: 80/20

---

## 📈 Evaluation Metrics

* Micro F1-score
* Hamming Loss
* Coverage Error
* Classification Report (per emotion)

### 🔹 Sample Performance

* Micro F1 Score: ~0.70+
* Hamming Loss: ~0.16–0.18

The model performs well on common emotions like **joy and sadness**, while less frequent emotions like **anger** remain more challenging.

---

## 🔍 Features

* End-to-end NLP pipeline:

  * Data preprocessing
  * Tokenization using BERT tokenizer
  * Model training and evaluation
  * Model saving and loading
* Supports:

  * Batch prediction from CSV
  * Real-time text input prediction
* Outputs emotion probabilities and binary predictions

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install transformers datasets scikit-learn torch matplotlib seaborn
```

### 2. Train the model

* Run the training script or notebook
* Model will be saved locally using:

```python
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
```

### 3. Run inference

```python
predict_emotions("I can't believe this happened!")
```

---

## 💡 Example Output

Input:

```
"I can't believe this happened!"
```

Output:

```
Predicted emotions: ['surprise', 'fear']
Probabilities: {anger: 0.12, fear: 0.67, joy: 0.08, sadness: 0.22, surprise: 0.71}
```

---

## 📌 Key Learnings

* Handling **multi-label classification problems**
* Fine-tuning transformer models for NLP tasks
* Working with **imbalanced datasets**
* Evaluating models using appropriate multi-label metrics
* Building reusable ML pipelines for training and inference

---

## 📫 Contact

For questions or collaboration, feel free to reach out:
**[davidcr2710@yahoo.com](mailto:davidcr2710@yahoo.com)**
