# cis6930fa24-project2

# README

# Name: Sai Sri Harsha Guddati
## Assignment Description
In this project, we need to create a data pipeline to develop a system that predicts the names redacted in text contexts using machine learning. In the code, we have trained a model on a given dataset and it generates predictions for redacted names in a test dataset.

### Steps
To run the Unredactor tool, ensure your dataset files are correctly formatted and placed in the project directory.

Prepare a training file (unredactor.tsv) with labeled redacted contexts for training and validation.
Provide a test file (test.tsv) for name prediction in unseen contexts.

### File Formats
1. Training/Validation File (unredactor.tsv):
Tab-separated file with three columns:
split: Indicates whether the data is for training (training) or validation (validation).
name: The original name that was redacted.
context: The sentence or text containing the redacted name.
2. Test File (test.tsv):
Tab-separated file with two columns:
id: A unique identifier for the test context.
context: The sentence or text containing the redacted name



## How to Install
To install the required dependencies, ensure you have pipenv installed. Navigate to the project directory and run:

```bash 
pipenv install -e .
```
This will install all the necessary dependencies in an isolated environment.

## How to Run
Run the script from the command line with the following parameters:

```bash 
    pipenv run python unredactor.py
```


## Testing
To run the test files in the /tests/ folder

```bash
pipenv run python -m pytest -v
```

The test files are located in the tests/ folder, ensuring that the critical functions work as expected.

## Example Output
After running the code, the following output will be printed as an example:

```python 
    print('''Precision: 0.85
             Recall: 0.82
             F1-Score: 0.83''')
```
Submission File: A file named submission.tsv is also generated for test predictions:
```bash
id    name
1     John
2     Alice
```
## Functions Overview
### Main Functions
#### 1. __init__(self)
Initializes the Unredactor system by setting up:

NLTK resources: Downloads required data for tokenization and part-of-speech tagging.
Vectorizer: CountVectorizer to extract features from text.
Classifier: A Logistic Regressor model for predicting names.
#### 2. extract_features(self, context)
Generates feature representations of the provided context for training and prediction.
Features include:

Length of redacted text.
Surrounding words and their part-of-speech (POS) tags.
Sentence structure details (e.g., position of redaction, total length).
N-grams (bigrams and trigrams) from surrounding words.
This robust feature set helps the classifier understand redacted contexts effectively.
#### 3. prepare_training_data(self, training_file)
Prepares the model for prediction by training it with labeled data.

Input: A TSV file with split, name, and context columns.
Process:
Extracts features from the training set.
Vectorizes the features and fits the Naive Bayes classifier.
Output: A trained model ready for predictions.
#### 4. predict_names(self, validation_file)
Predicts names for redacted contexts in a validation set.

Input: A TSV file containing validation data.
Process:
Extracts and vectorizes features for the validation set.
Predicts names using the trained classifier.
Evaluates the model with precision, recall, and F1-score metrics.
Output: Predicted names and performance metrics.
#### 5. generate_submission(self, test_file, output_file='submission.tsv')
Generates predictions for a test dataset and saves the results to a file.

Input:
A test TSV file with id and context columns.
An optional output file name.
Process:
Extracts and vectorizes features from the test dataset.
Predicts names and creates a submission file with IDs and predicted names.
Output: A TSV file containing predictions.
#### 6. main()
The entry point of the program.
Workflow:

Trains the model using a provided training dataset (unredactor.tsv).
Evaluates the model with predictions on a validation dataset.
Generates predictions for the test dataset (test.tsv) and saves them to a submission file.
### Test Functions
#### Test File: tests/test_unredactor.py
This file contains unit tests to verify the functionality of the Unredactor class. The tests use the unittest framework to ensure each method behaves as expected.

1. test_extract_features()
Purpose: Verifies the extract_features method generates correct feature representations from a given context.
Test Cases:

Checks that features include expected elements like length, surrounding words, POS tags, and n-grams.
Confirms handling of edge cases (e.g., no redacted text or minimal context).
```python 
    def test_extract_features():
        pass
```
2. test_prepare_training_data()
Purpose: Ensures that training data preparation works as intended.
Test Cases:

Confirms the classifier is successfully trained on the input TSV file.
Checks that vectorized features are correctly generated from the training set.
```python 
    def test_prepare_training_data():
        pass
```
3. test_predict_names()
Purpose: Validates the predict_names method's ability to make accurate predictions.
Test Cases:

Ensures predictions match expected output for a sample validation dataset.
Verifies precision, recall, and F1-score calculations for the predictions.
```python 
    def test_predict_names():
        pass
```
4. test_generate_submission()
Purpose: Tests the generate_submission method to confirm correct output file generation.
Test Cases:

Checks the format and content of the generated submission file.
Ensures the file includes the correct IDs and predicted names.
```python 
    def test_generate_submission():
        pass
```
5. test_pipeline_workflow()
Purpose: Simulates an end-to-end test of the entire Unredactor pipeline.
Test Cases:

Executes the full workflow: training, validation, and test prediction.
Ensures the pipeline runs without errors and produces expected results.
```python 
    def test_pipeline_workflow()
        pass
```
#### Test File: tests/test_main.py
1. test_main_pipeline_execution()
Purpose: Verifies the end-to-end execution of the main() function.
Test Cases:

Ensures that the prepare_training_data, predict_names, and generate_submission methods are called in sequence.
Validates the integration of individual components (e.g., feature extraction, training, prediction).
2. test_output_files()
Purpose: Confirms that output files are created correctly after running the pipeline.
Test Cases:

Verifies that the submission.tsv file is generated in the correct format.
Checks that the file includes the appropriate columns (id and name) and the expected number of rows.
3. test_logging_and_errors()
Purpose: Ensures that errors are handled gracefully during the pipeline execution.
Test Cases:

Tests the behavior when input files are missing or improperly formatted.
Confirms that meaningful error messages are logged.
Ensures the program exits gracefully without crashing.
4. test_integration_with_sample_data()
Purpose: Runs the full pipeline with a set of sample input files to verify integration.
Test Cases:

Uses mock or small TSV files (e.g., sample_training.tsv, sample_validation.tsv, sample_test.tsv) to simulate the pipeline.
Confirms that the final predictions align with the expected results based on the sample data.


## Bugs and Assumptions
1. Context Length Matters:

The code heavily relies on context features (previous and next words, sentence length, etc.), assuming these features are sufficient to predict redacted names, which may not hold true for more complex or ambiguous sentences.
2. POS Tags Are Predictive:

Assumes part-of-speech (POS) tags for previous and next words significantly contribute to identifying redacted entities, which is not guaranteed.
3. Training Data is Balanced and Labeled:

Assumes a sufficient number of labeled examples with diverse contexts are available for training. Imbalanced or insufficient training data will lead to poor model performance.
4. Count-Based Features Are Sufficient:

Relies on simple count-based features (n-grams, word counts, sentence length) instead of more sophisticated embeddings (e.g., word vectors or contextual embeddings like BERT).
5. No Preprocessing or Cleaning:

Assumes that input text is already clean, tokenized correctly, and free from noise (e.g., typos, extra spaces, HTML entities), which is often not true in real-world datasets.
6. No Error Handling:

The code lacks robust error handling for file I/O, feature extraction, or model training.
7. No Hyperparameter Tuning:

The classifier and vectorizer parameters are hardcoded, with no provision for tuning them based on validation performance.
