import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import val_ds
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Evaluate on the dataset directly
val_loss, val_acc = model.evaluate(val_ds)
print(f'Validation Accuracy: {val_acc:.4f}')

# Collect all true labels and images from the dataset
y_true = []
X_all = []

for x_batch, y_batch in val_ds:
    X_all.append(x_batch.numpy())
    y_true.append(y_batch.numpy())

X_all = np.concatenate(X_all)
y_true = np.concatenate(y_true)

# Predict classes
y_pred = model.predict(X_all)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_true_classes, y_pred_classes)