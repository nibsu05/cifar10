from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(true_labels, pred_labels):
    print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
    print(f"Precision (macro): {precision_score(true_labels, pred_labels, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(true_labels, pred_labels, average='macro'):.4f}")
    print(f"F1 Score (macro): {f1_score(true_labels, pred_labels, average='macro'):.4f}")