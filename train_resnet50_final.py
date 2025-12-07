import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
from PIL import Image
import random
import sys

# Проверка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# Параметры обучения
BATCH_SIZE = 32
NUM_EPOCHS = 40  # Увеличено для лучшей сходимости
LEARNING_RATE = 0.0001  # Более низкий learning rate для тонкой настройки
NUM_WORKERS = 0  # 0 для Windows
PATIENCE = 7  # Early stopping patience
MIN_IMAGES_PER_CLASS = 50

# Пути
DATASET_PATH = "data/plantvillage_dataset"
MODEL_SAVE_PATH = "models/plant_disease_resnet50_final.pth"
BEST_MODEL_PATH = "models/plant_disease_resnet50_best.pth"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

print(f"\n📂 Dataset path: {DATASET_PATH}")
print(f"💾 Best model save path: {BEST_MODEL_PATH}")

# Проверка существования датасета
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset not found at: {DATASET_PATH}")
    print("Please ensure you copied the dataset manually to the correct location!")
    exit()

# Получение списка классов
class_names = [d for d in os.listdir(DATASET_PATH)
               if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
class_names.sort()
NUM_CLASSES = len(class_names)

print(f"\n🎯 Dataset Classes ({NUM_CLASSES} total):")
for i, class_name in enumerate(class_names, 1):
    class_path = os.path.join(DATASET_PATH, class_name)
    num_images = len([f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"   {i}. {class_name.replace('_-_', ' - ')}: {num_images:,} images")

# Настройка преобразований данных с продвинутой аугментацией
print(f"\n🔄 Setting up data transformations...")

# Аугментация для обучающей выборки
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Преобразования для валидации и теста
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка датасета
print("\n📦 Loading dataset...")
full_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=train_transforms
)

# Проверка соответствия классов
print("\n🔍 Verifying class names match dataset structure...")
dataset_class_names = full_dataset.classes
for i, (expected, actual) in enumerate(zip(class_names, dataset_class_names)):
    print(f"   Class {i + 1}: Expected '{expected}', Actual '{actual}'")

# Разделение датасета: 70% train, 15% val, 15% test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_test_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

# Применение правильных преобразований для валидации и теста
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

print(f"\n📊 Dataset split:")
print(f"   Training:   {len(train_dataset):,} images ({len(train_dataset) / len(full_dataset):.1%})")
print(f"   Validation: {len(val_dataset):,} images ({len(val_dataset) / len(full_dataset):.1%})")
print(f"   Test:       {len(test_dataset):,} images ({len(test_dataset) / len(full_dataset):.1%})")

# Балансировка классов
print("\n⚖️ Setting up class balancing...")

# Подсчет изображений в каждом классе для обучающей выборки
class_counts = torch.zeros(NUM_CLASSES)
for _, label in train_dataset:
    class_counts[label] += 1

# Вычисление весов для каждого класса
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

print("\n📊 Class distribution in training set:")
for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
    weight = class_weights[i]
    print(f"   Class {i + 1}: {class_name.replace('_-_', ' - ')}")
    print(f"      Images: {int(count):,}, Weight: {weight:.4f}")

# Создание весов для каждого сэмпла
sample_weights = [class_weights[label] for _, label in train_dataset]

# Создание взвешенного сэмплера
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Создание DataLoader
print("\n⚡ Creating data loaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,  # Взвешенная выборка вместо случайной
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# === ЗАГРУЗКА ПРЕДОБУЧЕННОЙ RESNET50 ИЗ ЛОКАЛЬНОГО ФАЙЛА ===
print("\n" + "=" * 60)
print("🧠 LOADING PRE-TRAINED RESNET50 FROM LOCAL FILE")
print("=" * 60)

# Пути для поиска весов
possible_weights_paths = [
    "models/resnet50_weights/resnet50-11ad3fa6.pth",
    "models/resnet50-11ad3fa6.pth",
    "resnet50_weights/resnet50-11ad3fa6.pth",
    "resnet50-11ad3fa6.pth",
    os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "resnet50-11ad3fa6.pth")
]

weights_path = None
for path in possible_weights_paths:
    if os.path.exists(path):
        weights_path = path
        file_size = os.path.getsize(path) / (1024 * 1024)  # Размер в MB
        print(f"✅ Found weights file at: {path}")
        print(f"   File size: {file_size:.2f} MB (expected: ~97.8 MB)")
        if abs(file_size - 97.8) > 5.0:  # Допускаем погрешность 5MB
            print(f"⚠️  Warning: File size seems incorrect. Expected ~97.8 MB, got {file_size:.2f} MB")
        break

if weights_path is None:
    print("❌ ResNet50 weights file not found in any of the expected locations!")
    print("\n💡 EXPECTED LOCATIONS:")
    print("1. models/resnet50_weights/resnet50-11ad3fa6.pth (RECOMMENDED)")
    print("2. models/resnet50-11ad3fa6.pth")
    print("3. resnet50_weights/resnet50-11ad3fa6.pth")
    print("4. resnet50-11ad3fa6.pth")

    print("\n🔧 SETUP INSTRUCTIONS:")
    print("1. Download the weights file from:")
    print("   https://download.pytorch.org/models/resnet50-11ad3fa6.pth")
    print("2. Create folder structure:")
    print("   mkdir models")
    print("   mkdir models/resnet50_weights")
    print("3. Place the downloaded file in:")
    print("   models/resnet50_weights/resnet50-11ad3fa6.pth")

    exit()

# Создаем пустую модель ResNet50
print("\n🔄 Creating empty ResNet50 model...")
model = torchvision.models.resnet50(weights=None)

# Загрузка весов
try:
    print(f"🔄 Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Weights loaded successfully!")
except RuntimeError as e:
    print(f"❌ Error loading weights: {str(e)}")
    print("\n🔧 TROUBLESHOOTING:")
    print("1. Verify the file is not corrupted (should be ~97.8 MB)")
    print("2. Check if the file was downloaded completely")
    print("3. Try downloading again from the official URL")
    print("4. Ensure you have enough disk space")
    exit()
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")
    exit()

# Проверка загруженных весов
print("\n🔍 Verifying loaded weights...")
model = model.to(device)
model.eval()
with torch.no_grad():
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    sample_output = model(sample_input)
print(f"✅ Model forward pass successful! Output shape: {sample_output.shape}")

# Заморозка всех слоев кроме последних
print("🔒 Freezing base layers...")
for param in model.parameters():
    param.requires_grad = False

# Улучшенный классификатор с двумя скрытыми слоями
print(f"🔧 Modifying classifier for {NUM_CLASSES} classes...")
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, NUM_CLASSES)
)

# Перемещение модели на GPU
model = model.to(device)
model.train()  # Переводим в режим обучения
print("✅ Model moved to GPU and ready for training!")

# Настройка функции потерь и оптимизатора с балансировкой
class_weights_tensor = class_weights.to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3
)

print(f"\n⚙️ Training configuration:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Optimizer: Adam (only for classifier)")
print(f"   Scheduler: ReduceLROnPlateau")
print(f"   Early stopping patience: {PATIENCE} epochs")
print(f"   Class balancing: Enabled")
print(f"   Advanced augmentation: Enabled")


# Функция обучения одной эпохи
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass и оптимизация
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Обновление прогресс-бара
        accuracy = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f"{running_loss / (batch_idx + 1):.4f}",
            'Acc': f"{accuracy:.2f}%"
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Функция валидации
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


# Функция для постепенного размораживания слоев
def unfreeze_layers(epoch):
    """Постепенно размораживает слои модели для более тонкой настройки"""
    if epoch == 10:  # После 10 эпох
        print("\n🔓 Unfreezing layer4 (last convolutional block)...")
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        # Обновление оптимизатора для новых параметров
        optimizer.add_param_group(
            {'params': [p for n, p in model.named_parameters() if "layer4" in n and p.requires_grad],
             'lr': LEARNING_RATE * 0.1})

    if epoch == 25:  # После 25 эпох
        print("\n🔓 Unfreezing layer3...")
        for name, param in model.named_parameters():
            if "layer3" in name and param.requires_grad == False:
                param.requires_grad = True

        # Обновление оптимизатора для новых параметров
        optimizer.add_param_group(
            {'params': [p for n, p in model.named_parameters() if "layer3" in n and p.requires_grad],
             'lr': LEARNING_RATE * 0.05})


# Обучение модели
if __name__ == '__main__':
    print(f"\n🔥 Starting training for {NUM_EPOCHS} epochs...")
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    training_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Постепенное размораживание слоев
        unfreeze_layers(epoch)

        # Обучение
        train_loss, train_acc = train_one_epoch(epoch)

        # Валидация
        val_loss, val_acc = validate()

        # Обновление scheduler
        scheduler.step(val_acc)

        # Сохранение статистики
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 New best model saved! Validation accuracy: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"🚨 Early stopping triggered after {epoch + 1} epochs!")
            break

        print(f"\n📊 Epoch {epoch + 1}/{NUM_EPOCHS} results:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"   Best Val Acc so far: {best_val_acc:.2f}%")
        print(f"   Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    training_end_time = time.time()
    print(f"\n✅ Training completed in {training_end_time - training_start_time:.2f} seconds!")

    # Загрузка лучшей модели для тестирования
    print("\n🧪 Loading best model for testing...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    # Тестирование
    print("🔍 Testing on test set...")
    test_correct = 0
    test_total = 0
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            # Сохранение предсказаний для анализа
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = 100. * test_correct / test_total
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")

    # Визуализация результатов
    print("\n📊 Generating training plots...")

    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history_final.png'))
    print(f"✅ Training history saved to: {os.path.join(RESULTS_DIR, 'training_history_final.png')}")

    # Матрица ошибок (confusion matrix)
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_final.png'))
    print(f"✅ Confusion matrix saved to: {os.path.join(RESULTS_DIR, 'confusion_matrix_final.png')}")

    # Отчет классификации
    print("\n📋 Classification Report:")
    report = classification_report(test_labels, test_predictions, target_names=class_names)
    print(report)

    # Сохранение отчета
    with open(os.path.join(RESULTS_DIR, 'classification_report_final.txt'), 'w') as f:
        f.write(report)

    # Сохранение финальной модели
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 Final model saved to: {MODEL_SAVE_PATH}")


    # Функция для предсказания на новых изображениях
    def predict_image(image_path):
        """Предсказывает болезнь на изображении растения"""

        try:
            # Загрузка и преобразование изображения
            img = Image.open(image_path).convert('RGB')
            transform = val_test_transforms
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Предсказание
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            # Получение названия класса
            predicted_class = class_names[predicted_idx.item()]
            confidence_value = confidence.item() * 100

            return {
                'class': predicted_class,
                'confidence': confidence_value,
                'probabilities': probabilities.cpu().numpy()[0]
            }

        except Exception as e:
            return {'error': str(e)}


    # Тестовый пример с реальными изображениями из датасета
    print("\n🔍 Testing prediction function with real dataset images...")
    sample_images = []

    # Берем по 1 изображению из каждого класса для тестирования
    for class_name in class_names[:5]:  # Тест только для первых 5 классов
        class_path = os.path.join(DATASET_PATH, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            sample_images.append(os.path.join(class_path, random.choice(image_files)))

    for img_path in sample_images:
        print(f"\n📸 Testing image: {os.path.basename(img_path)}")
        result = predict_image(img_path)
        if 'error' not in result:
            print(f"✅ Prediction:")
            print(f"   Class: {result['class'].replace('_-_', ' - ')}")
            print(f"   Confidence: {result['confidence']:.2f}%")
        else:
            print(f"❌ Prediction error: {result['error']}")

    print(f"\n🎉 SUCCESS! Training completed successfully!")
    print(f"📍 Best model saved to: {BEST_MODEL_PATH}")
    print(f"📊 Results saved to: {RESULTS_DIR}/")
    print(f"\n🎯 Final Result: Test Accuracy = {test_acc:.2f}%")
    if test_acc >= 85.0:
        print("✅ EXCELLENT! Model accuracy exceeds 85% target!")
    elif test_acc >= 80.0:
        print("✅ GOOD! Model accuracy meets minimum requirement for production.")
    else:
        print("⚠️  Model accuracy is below 80% target. Consider further improvements.")

    print("\n🚀 Next steps - creating Flask web application:")
    print("1. Create app/__init__.py for Flask application factory")
    print("2. Create app/routes.py for image upload and prediction endpoints")
    print("3. Create templates for user interface")
    print("4. Set up Docker containerization")