import os
import shutil
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np


def set_seed(seed=42):
    """Tüm random seed'leri sabitler"""
    random.seed(seed)
    np.random.seed(seed)


def create_split_directories(output_path):
    """Train/Test/Val klasörlerini oluşturur"""
    splits = ['train', 'test', 'val']
    classes = ['gaussian', 'perlin', 'poisson', 'salt', 'speckle']

    for split in splits:
        for class_name in classes:
            dir_path = Path(output_path) / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Klasör yapısı oluşturuldu: {output_path}")


def get_files_per_class(dataset_path):
    """Her sınıf için dosya listelerini alır"""
    classes = ['gaussian', 'perlin', 'poisson', 'salt', 'speckle']
    class_files = {}

    for class_name in classes:
        class_path = Path(dataset_path) / class_name
        if class_path.exists():
            files = [f for f in class_path.glob('*') if
                     f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            class_files[class_name] = files
            print(f"{class_name}: {len(files)} dosya")
        else:
            print(f"Uyarı: {class_path} klasörü bulunamadı!")
            class_files[class_name] = []

    return class_files


def split_and_copy_files(class_files, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Dosyaları train/val/test olarak böler ve kopyalar
    train_ratio + val_ratio + test_ratio = 1.0 olmalı
    """
    set_seed(seed)

    split_info = {
        'train': {},
        'val': {},
        'test': {}
    }

    total_files = 0

    for class_name, files in class_files.items():
        if len(files) == 0:
            print(f"Uyarı: {class_name} sınıfında dosya yok!")
            continue

        total_files += len(files)

        # Önce train ve geçici (val+test) ayırımı
        train_files, temp_files = train_test_split(
            files,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )

        # Geçici dosyaları val ve test olarak ayır
        # val_ratio ve test_ratio'yu geçici dosyalar üzerinden hesapla
        val_size = val_ratio / (val_ratio + test_ratio)

        val_files, test_files = train_test_split(
            temp_files,
            train_size=val_size,
            random_state=seed,
            shuffle=True
        )

        # Dosya sayılarını kaydet
        split_info['train'][class_name] = len(train_files)
        split_info['val'][class_name] = len(val_files)
        split_info['test'][class_name] = len(test_files)

        # Dosyaları kopyala
        splits_data = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split_name, file_list in splits_data.items():
            dest_dir = Path(output_path) / split_name / class_name

            for i, file_path in enumerate(file_list):
                # Hedef dosya adını oluştur (çakışmaları önlemek için)
                dest_file = dest_dir / f"{class_name}_{i:04d}{file_path.suffix}"
                shutil.copy2(file_path, dest_file)

        print(f"{class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Özet bilgileri yazdır
    print("\n" + "=" * 50)
    print("DATASET BÖLÜNME ÖZETİ")
    print("=" * 50)

    for split_name in ['train', 'val', 'test']:
        total_split = sum(split_info[split_name].values())
        print(f"\n{split_name.upper()}:")
        for class_name in split_info[split_name]:
            count = split_info[split_name][class_name]
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            print(f"  {class_name}: {count} dosya ({percentage:.1f}%)")
        print(f"  TOPLAM: {total_split} dosya")

    print(f"\nTOPLAM DOSYA SAYISI: {total_files}")
    print(f"Seed kullanıldı: {seed}")

    return split_info


def save_split_info(split_info, output_path, seed):
    """Bölünme bilgilerini txt dosyasına kaydeder"""
    info_file = Path(output_path) / "dataset_split_info.txt"

    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("DATASET BÖLÜNME BİLGİLERİ\n")
        f.write("=" * 50 + "\n")
        f.write(f"Kullanılan Seed: {seed}\n")
        f.write(f"Oluşturulma Tarihi: {Path(__file__).stat().st_mtime}\n\n")

        total_files = sum(sum(class_dict.values()) for class_dict in split_info.values())

        for split_name in ['train', 'val', 'test']:
            total_split = sum(split_info[split_name].values())
            f.write(f"{split_name.upper()}:\n")
            for class_name in ['gaussian', 'perlin', 'poisson', 'salt', 'speckle']:
                count = split_info[split_name].get(class_name, 0)
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                f.write(f"  {class_name}: {count} dosya ({percentage:.1f}%)\n")
            f.write(f"  TOPLAM: {total_split} dosya\n\n")

        f.write(f"GENEL TOPLAM: {total_files} dosya\n")

    print(f"Bölünme bilgileri kaydedildi: {info_file}")


def main():
    # Konfigürasyon
    SEED = 42
    INPUT_DATASET_PATH = "noisy_dataset"  # Ana dataset klasörünüz
    OUTPUT_DATASET_PATH = "split_dataset"  # Bölünmüş dataset kayıt yeri

    # Bölünme oranları (toplamı 1.0 olmalı)
    TRAIN_RATIO = 0.7  # %70 eğitim
    VAL_RATIO = 0.15  # %15 doğrulama
    TEST_RATIO = 0.15  # %15 test

    print("Gürültü Sınıflandırması Dataset Bölme Script'i")
    print("=" * 50)
    print(f"Kaynak Dataset: {INPUT_DATASET_PATH}")
    print(f"Hedef Dataset: {OUTPUT_DATASET_PATH}")
    print(f"Train: {TRAIN_RATIO * 100}%, Val: {VAL_RATIO * 100}%, Test: {TEST_RATIO * 100}%")
    print(f"Seed: {SEED}")
    print()

    # Seed'i sabitler
    set_seed(SEED)

    # Kaynak dataset kontrolü
    if not Path(INPUT_DATASET_PATH).exists():
        print(f"Hata: {INPUT_DATASET_PATH} klasörü bulunamadı!")
        return

    # Hedef klasör yapısını oluştur
    create_split_directories(OUTPUT_DATASET_PATH)

    # Her sınıf için dosyaları al
    class_files = get_files_per_class(INPUT_DATASET_PATH)

    # Dosyaları böl ve kopyala
    split_info = split_and_copy_files(
        class_files,
        OUTPUT_DATASET_PATH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )

    # Bölünme bilgilerini kaydet
    save_split_info(split_info, OUTPUT_DATASET_PATH, SEED)

    print("\nDataset bölme işlemi tamamlandı!")


if __name__ == "__main__":
    main()