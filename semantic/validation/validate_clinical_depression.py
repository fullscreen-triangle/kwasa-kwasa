# validation_clinical_depression.py
"""
Validate clinical depression diagnosis accuracy
Uses your actual depression_treatment data
"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from src.core.semantic_maxwell_demon import SemanticMaxwellDemon


def test_depression_diagnosis():
    """Test on real clinical depression data"""

    smd = SemanticMaxwellDemon()

    # Load your depression treatment data
    # (Assuming you have this from depression_treatment.trb execution)
    patient_data = load_depression_data()

    results = {
        'true_labels': [],
        'predicted_labels': [],
        'confidence_scores': [],
        'patient_ids': []
    }

    for patient in patient_data:
        # Extract clinical features
        features = {
            'plv': patient['plv'],
            'theta_power': patient['theta_power'],
            'gamma_power': patient['gamma_power'],
            'phq9_score': patient['phq9_score'],
            'symptoms': patient['symptoms']
        }

        # Semantic Maxwell Demon diagnosis
        diagnosis = smd.diagnose_depression(features)

        results['true_labels'].append(patient['expert_diagnosis'])
        results['predicted_labels'].append(diagnosis['label'])
        results['confidence_scores'].append(diagnosis['confidence'])
        results['patient_ids'].append(patient['id'])

        print(f"Patient {patient['id']}:")
        print(f"  True: {patient['expert_diagnosis']}")
        print(f"  Predicted: {diagnosis['label']}")
        print(f"  Confidence: {diagnosis['confidence']:.2f}")
        print()

    # Calculate metrics
    accuracy = accuracy_score(results['true_labels'], results['predicted_labels'])
    precision, recall, f1, _ = precision_recall_fscore_support(
        results['true_labels'],
        results['predicted_labels'],
        average='weighted'
    )

    # Confusion matrix
    cm = confusion_matrix(results['true_labels'], results['predicted_labels'])

    print("=" * 60)
    print("CLINICAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1-score: {f1:.1%}")
    print(f"\nMeets 94% claim: {accuracy >= 0.94}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # Visualize
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    # Confidence distribution
    ax2.hist(results['confidence_scores'], bins=20, edgecolor='black')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.94, color='red', linestyle='--', label='94% threshold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('clinical_validation.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'meets_94_claim': accuracy >= 0.94,
        'confusion_matrix': cm,
        'results': results
    }

def load_depression_data():
    """Load depression treatment data"""
    # Load from your depression_treatment execution outputs
    # This should match the data structure from depression_treatment.trb
    pass

if __name__ == '__main__':
    results = test_depression_diagnosis()
