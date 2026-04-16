# scripts/split_dataset.py

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

# ============================================================
# 0. Load Dataset (CROPPED oranges)
# ============================================================

# New root: use the CROPPED dataset instead of original
# Expected structure:
# D:\GAN\Data\orange_crops\ripe_oranges\*.jpg
# D:\GAN\Data\orange_crops\unripe_orange\*.jpg
data_root = r"D:\GAN\Data\orange_crops"

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # match GAN input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# Load complete dataset
full_dataset = datasets.ImageFolder(root=data_root, transform=transform)

print("====================================")
print("Classes found:", full_dataset.classes)
print("Total images:", len(full_dataset))
print("====================================")

class_to_idx = full_dataset.class_to_idx
# Folder names must match these keys:
# ['ripe_oranges', 'unripe_orange']
ripe_label = class_to_idx["ripe_oranges"]
unripe_label = class_to_idx["unripe_orange"]

# ============================================================
# 1. Build Index List + Labels List
# ============================================================

all_indices = list(range(len(full_dataset)))
all_labels = [s[1] for s in full_dataset.samples]  # label for each image index

# ============================================================
# 2. Stratified Split: Train(70%) / Temp(30%)
# ============================================================

train_idx, temp_idx, train_labels, temp_labels = train_test_split(
    all_indices,
    all_labels,
    test_size=0.30,        # 30% → val + test
    stratify=all_labels,
    random_state=42,
)

# ============================================================
# 3. Split Temp into Val(15%) + Test(15%)
# ============================================================

val_idx, test_idx, val_labels, test_labels = train_test_split(
    temp_idx,
    temp_labels,
    test_size=0.50,        # Split 30% temp into 15% val + 15% test
    stratify=temp_labels,
    random_state=42,
)

# ============================================================
# 4. Create Datasets
# ============================================================

train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)
test_dataset  = Subset(full_dataset, test_idx)

print("====================================")
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))
print("====================================")

# ============================================================
# 5. For GANs: Separate Ripe & Unripe TRAIN INDICES
# ============================================================

train_ripe_idx = [i for i in train_idx if all_labels[i] == ripe_label]
train_unripe_idx = [i for i in train_idx if all_labels[i] == unripe_label]

print("Ripe images in TRAIN:", len(train_ripe_idx))
print("Unripe images in TRAIN:", len(train_unripe_idx))
print("====================================")

ripe_train_dataset   = Subset(full_dataset, train_ripe_idx)
unripe_train_dataset = Subset(full_dataset, train_unripe_idx)

# ============================================================
# 6. DataLoaders (classification / GAN training)
# ============================================================

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

ripe_train_loader   = DataLoader(ripe_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
unripe_train_loader = DataLoader(unripe_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print("Dataloaders ready!\n")
print("You can now import ripe_train_loader & unripe_train_loader in train scripts")
