# SVM Text Classifier

Support Vector Machine method for text classification.  (Description...)

## Usage

This module provides a simple interface for constructing a text based classifier and using it for predictions:

```python
# Create model:
classifier = SVM_text_class(x_train, y_train)

# Make predictions:
predictions = classifier.pred(x_pred)
```

For a full example, see [svm_text_classifier_example.ipynb](svm_text_classifier_example.ipynb).
