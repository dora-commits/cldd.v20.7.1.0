# Define transforms
transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
target_transform = None

# Load data
url = 'https://drive.google.com/file/d/1-agOF3uCwQWHhaa454sIoGLLxXtXr_tl/view?usp=sharing'
medical_train = MedicalDataset(root='Medical',
                                url= url,
                                transform=transform,
                                target_transform=target_transform,
                                download=True,  # Set to True if download is needed
                                )

dataset = medical_train
# Feed to data loader
data_loader = DataLoader(dataset,
                            batch_size=len(dataset),
                            shuffle=True,
                            num_workers=2,
                            pin_memory=False,
                            drop_last=True)

# Obtain batch
# for idx, (image, target) in enumerate(data_loader):
#     print(f"Batch {idx}, Target {target}")

# Delete large variables to free memory
del medical_train
del dataset

# If using GPU, free CUDA memory
torch.cuda.empty_cache()

# batch = next(iter(data_loader))
# print(batch[0].shape)
# print(batch[1].shape)
dec_x_f_b, dec_y_f_b = next(iter(data_loader))
del data_loader
print(dec_x_f_b.shape)
print(dec_y_f_b.shape)
print(collections.Counter(dec_y_f_b.numpy()))