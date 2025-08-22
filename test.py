import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import random

# Model importları
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
        print(f"GPU kullanılıyor: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CPU kullanılıyor")
    return device


def get_test_transform():
    """Test için veri dönüşüm pipeline'ı"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_test_loader(dataset_path, batch_size=32, num_workers=4):
    """Test data loader oluştur"""
    test_transform = get_test_transform()

    test_dataset = datasets.ImageFolder(
        root=Path(dataset_path) / 'test',
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = test_dataset.classes
    print(f"Test sınıfları: {class_names}")
    print(f"Test örnekleri: {len(test_dataset)}")

    return test_loader, class_names


def load_model(model_class, model_path, num_classes, device):
    """Kaydedilmiş modeli yükle"""
    model = model_class(num_classes=num_classes).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model başarıyla yüklendi: {model_path}")
        return model
    except Exception as e:
        print(f"Model yüklenirken hata: {str(e)}")
        return None


def test_model_detailed(model, test_loader, device, class_names, model_name):
    """Detaylı model test"""
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    correct = 0
    total = 0

    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}

    print(f"\n{model_name} test ediliyor...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Sınıf bazında doğruluk
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)} işlendi")

    # Genel accuracy
    overall_acc = 100. * correct / total

    # Sınıf bazında accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            class_accuracies[class_name] = acc
        else:
            class_accuracies[class_name] = 0.0

    # Classification report
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4
    )

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)

    results = {
        'model_name': model_name,
        'overall_accuracy': overall_acc,
        'class_accuracies': class_accuracies,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'class_names': class_names
    }

    return results


def plot_confusion_matrix(cm, class_names, model_name, save_path=None):
    """Confusion matrix çiz"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix kaydedildi: {save_path}")

    plt.show()
    plt.close()


def save_detailed_results(results, output_dir):
    """Detaylı sonuçları kaydet"""
    os.makedirs(output_dir, exist_ok=True)

    model_name = results['model_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ana rapor dosyası
    report_path = Path(output_dir) / f"{model_name}_detailed_test_report_{timestamp}.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"DETAYLI TEST RAPORU\n")
        f.write(f"={'=' * 60}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Accuracy: {results['overall_accuracy']:.4f}%\n")
        f.write(f"\n")

        # Sınıf bazında accuracy
        f.write(f"SINIF BAZINDA DOĞRULUK ORANLARI:\n")
        f.write(f"{'-' * 40}\n")
        for class_name, acc in results['class_accuracies'].items():
            f.write(f"{class_name:<15}: {acc:.2f}%\n")
        f.write(f"\n")

        # Classification Report
        f.write(f"CLASSIFICATION REPORT:\n")
        f.write(f"{'-' * 40}\n")
        f.write(results['classification_report'])
        f.write(f"\n")

        # Confusion Matrix
        f.write(f"CONFUSION MATRIX:\n")
        f.write(f"{'-' * 40}\n")
        cm = results['confusion_matrix']

        # Header
        f.write(f"{'':>12}")
        for class_name in results['class_names']:
            f.write(f"{class_name[:8]:>8}")
        f.write(f"\n")

        # Matrix rows
        for i, class_name in enumerate(results['class_names']):
            f.write(f"{class_name[:10]:>10}  ")
            for j in range(len(results['class_names'])):
                f.write(f"{cm[i, j]:>8}")
            f.write(f"\n")

        f.write(f"\n")

        # İstatistikler
        f.write(f"İSTATİSTİKLER:\n")
        f.write(f"{'-' * 40}\n")
        f.write(f"Toplam test örneği: {len(results['targets'])}\n")
        f.write(f"Doğru tahmin: {sum(np.array(results['predictions']) == np.array(results['targets']))}\n")
        f.write(f"Yanlış tahmin: {sum(np.array(results['predictions']) != np.array(results['targets']))}\n")

    print(f"Detaylı test raporu kaydedildi: {report_path}")

    # Confusion matrix görselini kaydet
    cm_path = Path(output_dir) / f"{model_name}_confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(results['confusion_matrix'], results['class_names'],
                          model_name, cm_path)

    return report_path, cm_path


def compare_models(all_results, output_dir):
    """Modelleri karşılaştır"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = Path(output_dir) / f"model_comparison_{timestamp}.txt"

    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(f"MODEL KARŞILAŞTIRMA RAPORU\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test edilen model sayısı: {len(all_results)}\n\n")

        # Genel performans sıralaması
        sorted_results = sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True)

        f.write(f"GENEL PERFORMANS SIRALAMASI:\n")
        f.write(f"{'-' * 40}\n")
        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i}. {result['model_name']:<20}: {result['overall_accuracy']:.2f}%\n")
        f.write(f"\n")

        # Sınıf bazında en iyi performans
        class_names = all_results[0]['class_names']
        f.write(f"SINIF BAZINDA EN İYİ PERFORMANS:\n")
        f.write(f"{'-' * 40}\n")

        for class_name in class_names:
            best_acc = 0
            best_model = ""
            for result in all_results:
                acc = result['class_accuracies'][class_name]
                if acc > best_acc:
                    best_acc = acc
                    best_model = result['model_name']
            f.write(f"{class_name:<15}: {best_model:<20} ({best_acc:.2f}%)\n")
        f.write(f"\n")

        # Detaylı karşılaştırma tablosu
        f.write(f"DETAYLI KARŞILAŞTIRMA:\n")
        f.write(f"{'-' * 80}\n")

        # Header
        f.write(f"{'Model':<20}")
        f.write(f"{'Overall':<10}")
        for class_name in class_names:
            f.write(f"{class_name[:8]:<10}")
        f.write(f"\n")
        f.write(f"{'-' * 80}\n")

        # Her model için satır
        for result in sorted_results:
            f.write(f"{result['model_name']:<20}")
            f.write(f"{result['overall_accuracy']:.2f}%{'':>4}")
            for class_name in class_names:
                acc = result['class_accuracies'][class_name]
                f.write(f"{acc:.2f}%{'':>4}")
            f.write(f"\n")

    print(f"Model karşılaştırma raporu kaydedildi: {comparison_path}")
    return comparison_path


def main():
    # Konfigürasyon
    SEED = 42
    DATASET_PATH = "split_dataset"
    OUTPUT_DIR = "test_results"
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Seed sabitler
    set_seed(SEED)

    # Device
    device = get_device()

    # Test data loader
    test_loader, class_names = create_test_loader(DATASET_PATH, BATCH_SIZE, NUM_WORKERS)

    # Test edilecek modeller (eğer .pth dosyaları varsa)
    model_configs = [
        (DeepCNN, "DeepCNN", None),  # Model path'i buraya yazılabilir
        (ResNet34Pretrained, "ResNet34_Pretrained", None),
        (ResNet34Scratch, "ResNet34_Scratch", None),
        (EfficientNetB0Custom, "EfficientNetB0", None)
    ]

    all_results = []

    print(f"Test başlıyor - {len(model_configs)} model")
    print(f"Test örnekleri: {len(test_loader.dataset)}")
    print(f"Sınıflar: {class_names}")

    for model_class, model_name, model_path in model_configs:
        try:
            # Eğer model path'i yoksa, scratch'ten model oluştur (sadece test için)
            if model_path and Path(model_path).exists():
                model = load_model(model_class, model_path, len(class_names), device)
                if model is None:
                    continue
            else:
                print(
                    f"\n⚠  {model_name} için kaydedilmiş model bulunamadı, yeni model oluşturuluyor (rastgele ağırlıklar)")
                model = model_class(num_classes=len(class_names)).to(device)
                model.eval()

            # Test
            results = test_model_detailed(model, test_loader, device, class_names, model_name)
            all_results.append(results)

            # Sonuçları kaydet
            save_detailed_results(results, OUTPUT_DIR)

            # Memory temizle
            del model
            torch.cuda.empty_cache()

            print(f" {model_name} - Test Accuracy: {results['overall_accuracy']:.2f}%")

        except Exception as e:
            print(f" {model_name} test edilirken hata: {str(e)}")
            continue

    # Model karşılaştırması
    if len(all_results) > 1:
        print(f"\nModel karşılaştırması yapılıyor...")
        compare_models(all_results, OUTPUT_DIR)

    # Özet
    print(f"\n{'=' * 60}")
    print("TEST ÖZETİ")
    print(f"{'=' * 60}")

    if all_results:
        best_result = max(all_results, key=lambda x: x['overall_accuracy'])
        print(f"En iyi model: {best_result['model_name']} ({best_result['overall_accuracy']:.2f}%)")

        print(f"\nTüm modeller:")
        for result in sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True):
            print(f"  {result['model_name']:<20}: {result['overall_accuracy']:.2f}%")
    else:
        print("Hiç model test edilemedi!")

    print(f"\nTüm sonuçlar kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()