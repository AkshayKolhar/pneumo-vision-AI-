import os
from pathlib import Path
import torch
from utils.grad_cam import GradCAM
from utils.inference import load_model
import matplotlib.pyplot as plt

def visualize_predictions(model_path='models/best_pneumonia_model.pth',
                         data_dir='data/chest_xray/test',
                         num_samples=10):
    """
    Visualize Grad-CAM for sample predictions
    
    Args:
        model_path: Path to trained model
        data_dir: Path to test data
        num_samples: Number of samples to visualize
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)

    # Initialize Grad-CAM
    grad_cam = GradCAM(model)
    
    # Get sample images
    test_dir = Path(data_dir)
    normal_images = list((test_dir / 'NORMAL').glob('*.jpeg'))
    pneumonia_images = list((test_dir / 'PNEUMONIA').glob('*.jpeg'))
    
    # Select samples
    samples = []
    if normal_images:
        samples.extend(normal_images[:num_samples//2])
    if pneumonia_images:
        samples.extend(pneumonia_images[:num_samples//2])
    
    print(f"Visualizing {len(samples)} samples...")
    
    # Generate visualizations
    for i, img_path in enumerate(samples):
        try:
            fig, overlay, pred_class = grad_cam.visualize(str(img_path))
            
            # Save visualization
            save_path = f'gradcam_sample_{i+1}.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Sample {i+1}: {img_path.name} -> "
                  f"{'PNEUMONIA' if pred_class == 1 else 'NORMAL'}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nVisualizations saved as 'gradcam_sample_*.png'")

if __name__ == '__main__':
    visualize_predictions()