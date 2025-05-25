import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from main import EnhancedBMCNNwHFCs, NepaliMNISTDataset, LabelSmoothingLoss
import os

def test_model(model_path, test_data_dir, batch_size=64, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate file paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist")
    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test data directory {test_data_dir} does not exist")

    # Load the model architecture
    try:
        model = EnhancedBMCNNwHFCs(num_classes=46)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # Define test transforms (consistent with training code)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Prepare test dataset and dataloader
    try:
        test_dataset = NepaliMNISTDataset(test_data_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    except Exception as e:
        raise RuntimeError(f"Failed to load test dataset: {str(e)}")

    criterion = LabelSmoothingLoss(classes=46, smoothing=0.05)  # Match training smoothing

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


# Example usage
if __name__ == "__main__":
    model_path = "../models/best_model.pth"
    test_data_dir = "../data/extracted/dhcd/test"
    try:
        test_model(model_path, test_data_dir)
    except Exception as e:
        print(f"Error during testing: {str(e)}")