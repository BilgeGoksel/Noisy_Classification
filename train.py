import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time
import os
from pathlib import Path
import random
from datetime import datetime
from models import DeepCNN, ResNet34Pretrained, ResNet34Scratch, EfficientNetB0Custom


def set_seed(seed=42):
    """Tüm random seed'leri sabitler"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """GPU/CPU device seçimi"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" GPU kullanılıyor: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        print(f"PyTorch CUDA Support: {torch.backends.cudnn.enabled}")

        # GPU'ya test tensörü gönder
        test_tensor = torch.randn(1, 1).to(device)
        print(f"Test tensor device: {test_tensor.device}")

    else:
        device = torch.device('cpu')
        print(" CUDA kullanılamıyor, CPU kullanılıyor")
        print("Nedenleri:")
        print("- CUDA yüklü değil")
        print("- PyTorch CUDA versiyonu yüklü değil")
        print("- GPU driver sorunu")
    return device


def get_data_transforms():
    """Veri dönüşüm pipeline'ları"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def create_data_loaders(dataset_path, batch_size=32, num_workers=4):
    """Data loader'ları oluşturur"""
    train_transform, val_test_transform = get_data_transforms()

    # Datasetleri yükle
    train_dataset = datasets.ImageFolder(
        root=Path(dataset_path) / 'train',
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=Path(dataset_path) / 'val',
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=Path(dataset_path) / 'test',
        transform=val_test_transform
    )

    # Data loader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Sınıf isimleri
    class_names = train_dataset.classes
    print(f"Sınıflar: {class_names}")
    print(f"Train: {len(train_dataset)} örnek")
    print(f"Val: {len(val_dataset)} örnek")
    print(f"Test: {len(test_dataset)} örnek")

    return train_loader, val_loader, test_loader, class_names


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f'    Batch {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {running_loss / (batch_idx + 1):.4f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validasyon"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def test_model(model, test_loader, device, class_names):
    """Test ve classification report"""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            _, predicted = output.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_acc = 100. * correct / total

    # Classification report
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4
    )

    return test_acc, report, all_preds, all_targets


def train_model(model_class, model_name, train_loader, val_loader, test_loader,
                class_names, device, epochs=25, lr=0.001):
    """Model eğitimi"""
    print(f"\n{'=' * 60}")
    print(f"EĞİTİM BAŞLADI: {model_name}")
    print(f"{'=' * 60}")

    # Model oluştur
    model = model_class(num_classes=len(class_names)).to(device)

    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Eğitim geçmişi
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    best_model_state = None

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Eğitim
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validasyon
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduler
        scheduler.step()

        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  *** Yeni en iyi model! Val Acc: {val_acc:.2f}% ***")

    # En iyi modeli yükle
    model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    print(f"\nEğitim tamamlandı! Süre: {training_time:.2f} saniye")
    print(f"En iyi validasyon accuracy: {best_val_acc:.2f}%")

    # Test
    print("\nTest değerlendirmesi yapılıyor...")
    test_acc, report, test_preds, test_targets = test_model(model, test_loader, device, class_names)

    print(f"Test Accuracy: {test_acc:.2f}%")

    # Model ve sonuçları kaydet
    results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_report': report,
        'training_time': training_time,
        'test_preds': test_preds,
        'test_targets': test_targets
    }

    return model, results


def save_results(results, output_dir):
    """Sonuçları kaydet"""
    os.makedirs(output_dir, exist_ok=True)

    model_name = results['model_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Classification report kaydet
    report_path = Path(output_dir) / f"{model_name}_classification_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"MODEL: {model_name}\n")
        f.write(f"TARIH: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"En İyi Validasyon Accuracy: {results['best_val_acc']:.4f}%\n")
        f.write(f"Test Accuracy: {results['test_acc']:.4f}%\n")
        f.write(f"Eğitim Süresi: {results['training_time']:.2f} saniye\n")
        f.write("\nCLASSIFICATION REPORT:\n")
        f.write("=" * 60 + "\n")
        f.write(results['test_report'])

    print(f"Classification report kaydedildi: {report_path}")

    # Model kaydet
    return report_path


def main():
    # Konfigürasyon
    SEED = 42
    DATASET_PATH = "split_dataset"
    OUTPUT_DIR = "training_results"
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.001
    NUM_WORKERS = 4

    # Seed sabitler
    set_seed(SEED)

    # Device seçimi
    device = get_device()

    # Veri yükleyicilerini oluştur
    print("Veri yükleyicileri hazırlanıyor...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        DATASET_PATH, BATCH_SIZE, NUM_WORKERS
    )

    # Modeller tanımı
    models_to_train = [
        (DeepCNN, "DeepCNN"),
        (ResNet34Pretrained, "ResNet34_Pretrained"),
        (ResNet34Scratch, "ResNet34_Scratch"),
        (EfficientNetB0Custom, "EfficientNetB0")
    ]

    # Tüm sonuçları kaydet
    all_results = []

    print(f"\nToplam {len(models_to_train)} model eğitilecek")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    for i, (model_class, model_name) in enumerate(models_to_train):
        print(f"\n Model {i + 1}/{len(models_to_train)}: {model_name}")

        try:
            model, results = train_model(
                model_class, model_name,
                train_loader, val_loader, test_loader,
                class_names, device, EPOCHS, LEARNING_RATE
            )

            # Sonuçları kaydet
            report_path = save_results(results, OUTPUT_DIR)
            results['report_path'] = report_path
            all_results.append(results)

            # Memory temizle
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f" {model_name} eğitiminde hata: {str(e)}")
            continue

    # Özet raporu
    print(f"\n{'=' * 80}")
    print("TÜM MODELLERİN EĞİTİM ÖZETİ")
    print(f"{'=' * 80}")

    for result in all_results:
        print(f"{result['model_name']:<20} - "
              f"Val: {result['best_val_acc']:.2f}% | "
              f"Test: {result['test_acc']:.2f}% | "
              f"Süre: {result['training_time']:.1f}s")

    # En iyi modeli bul
    if all_results:
        best_result = max(all_results, key=lambda x: x['test_acc'])
        print(f"\n EN İYİ MODEL: {best_result['model_name']} "
              f"(Test Acc: {best_result['test_acc']:.2f}%)")

    print(f"\nTüm sonuçlar kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()