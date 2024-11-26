import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import nltk
import re

class Unredactor:
    def __init__(self):
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        self.vectorizer = CountVectorizer()
        self.classifier =  LogisticRegression(class_weight='balanced')

    def extract_features(self, context):
        """
        Extract features from the redaction context
        
        Features include:
        - Length of redacted text
        - Previous and next words
        - Surrounding context n-grams (bigram/trigram)
        - Part-of-speech tags for previous and next words
        - Redaction position
        - Sentence length
        - Redacted word count
        """
        redaction_length = len(re.findall('█', context))
        tokens = nltk.word_tokenize(context)
        
        redaction_index = [i for i, token in enumerate(tokens) if all(char == '█' for char in token)]
        features_str = f"length_{redaction_length}"
        if redaction_index:
            prev_word = tokens[redaction_index[0] - 1] if redaction_index[0] > 0 else '<START>'
            
            next_word = tokens[redaction_index[0] + 1] if redaction_index[0] < len(tokens) - 1 else '<END>'
            
            prev_pos = nltk.pos_tag([prev_word])[0][1] if prev_word != '<START>' else '<START_POS>'
            next_pos = nltk.pos_tag([next_word])[0][1] if next_word != '<END>' else '<END_POS>'
            
            context_ngrams = list(nltk.ngrams(tokens, 2)) + list(nltk.ngrams(tokens, 3))
            surrounding_ngrams = [gram for gram in context_ngrams if redaction_index[0] in range(tokens.index(gram[0]), tokens.index(gram[-1]) + 1)]
            
            sentence_length = len(tokens)
            
            redacted_word_count = len([token for token in tokens if all(char == '█' for char in token)])
            
            if redaction_index[0] == 0:
                redaction_position = 'beginning'
            elif redaction_index[0] == len(tokens) - 1:
                redaction_position = 'end'
            else:
                redaction_position = 'middle'

            features_str += f" prev_{prev_word}_{prev_pos} next_{next_word}_{next_pos} "
            features_str += f"redacted_count_{redacted_word_count} sentence_len_{sentence_length} position_{redaction_position} "
            vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=8, stop_words='english')
            
            try:
                ngram_features = vectorizer.fit_transform([context]).toarray()
                
                for idx, ngram in enumerate(vectorizer.get_feature_names_out()):
                    features_str += f"ngram_{ngram}_{ngram_features[0][idx]}"
            
            except ValueError: 
                for idx in range(50):
                    features_str += f"ngram_placeholder_{idx}_0 "
            return features_str
        
        return "no_redaction"

    def prepare_training_data(self, training_file):
        """
        Prepare training data from the provided TSV file
        """
        df = pd.read_csv(training_file, sep='\t', names=['split', 'name', 'context'],usecols=[0,1,2])
        
        train_df = df[df['split'] == 'training']
        
        X = train_df['context'].apply(self.extract_features).tolist()
        y = train_df['name']
        
        X_vectorized = self.vectorizer.fit_transform(X)
        
        self.classifier.fit(X_vectorized, y)

    def predict_names(self, validation_file):
        """
        Predict names for validation contexts
        """
        df = pd.read_csv(validation_file, sep='\t', names=['split', 'name', 'context'],usecols=[0,1,2])
        
        val_df = df[df['split'] == 'validation']
        
        X_val = val_df['context'].apply(self.extract_features).tolist()
        X_val_vectorized = self.vectorizer.transform(X_val)
        
        predictions = self.classifier.predict(X_val_vectorized)
        
        true_names = val_df['name']
        precision, recall, f1, _ = precision_recall_fscore_support(true_names, predictions, average='weighted')
        
        print(f"Precision: {precision*10:.2f}")
        print(f"Recall: {recall*10:.2f}")
        print(f"F1-Score: {f1*10:.2f}")
        
        return predictions

    def generate_submission(self, test_file, output_file='submission.tsv'):
        """
        Generate submission file for test dataset
        """
        test_df = pd.read_csv(test_file, sep='\t', names=['id', 'context'], usecols=[0, 1])
        
        X_test = test_df['context'].apply(self.extract_features).tolist()
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        test_predictions = self.classifier.predict(X_test_vectorized)
        
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'name': test_predictions
        })
        
        # Save submission file
        submission_df.to_csv(output_file, sep='\t', index=False)
        print(f"Submission file saved to {output_file}")

def main():
    unredactor = Unredactor()
    
    unredactor.prepare_training_data('unredactor.tsv')
    
    unredactor.predict_names('unredactor.tsv')
    
    unredactor.generate_submission('test.tsv')

if __name__ == '__main__':
    main()