class MedicalDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 url: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 pin_memory: bool = False,
                ):

        super(MedicalDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.pin_memory = pin_memory
        self.dataset_root = root

        if download:
            if not url:
                raise ValueError("URL for downloading dataset must be provided.")
            self.download_dataset(url)

        self.dataset_root = os.path.join(self.dataset_root, 'COVID-19_Radiography_Dataset')
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_root}. Please set download=True to download it.")

        # Set desired class-to-index mapping
        self.class_to_idx = {
            'Normal': 0,
            'Lung_Opacity': 1,
            'COVID': 2,
            'Viral Pneumonia': 3
        }

        print('class_to_idx', self.class_to_idx)

        # Get image paths and targets
        self.image_paths, self.targets = self._find_paths(self.dataset_root, self.class_to_idx)
        # print('image_paths', self.image_paths)
        # print('targets', self.targets)

        if self.__len__() == 0:
            raise FileNotFoundError(f"Found 0 files in {self.dataset_root}")

        # Pin memory (if necessary)
        if self.pin_memory:
            self.data = self._load_images('RGB', self.image_paths)

    def download_dataset(self, url: str):
        archive_filename = os.path.join(self.dataset_root, 'Medical.zip')
        download_root = self.dataset_root

        # Ensure the directory exists
        os.makedirs(download_root, exist_ok=True)

        # Convert Google Drive sharing link to direct download link
        file_id = url.split('/')[5]
        direct_url = f"https://drive.google.com/uc?id={file_id}"

        # Download the dataset using gdown
        if not os.path.exists(archive_filename):
            print(f"Downloading dataset from {direct_url}...")
            # Use gdown.download to download the file from Google Drive
            gdown.download(direct_url, archive_filename, quiet=False)
            print("Download completed.")

        # Check if the file exists and is not empty
        if not os.path.exists(archive_filename) or os.stat(archive_filename).st_size == 0:
            raise FileNotFoundError(f"Failed to download or file is empty: {archive_filename}")

        # Extract the dataset
        print(f"Extracting {archive_filename}...")
        try:
            # Open the zip file
            with zipfile.ZipFile(archive_filename, 'r') as zip_ref:
                # Extract all contents to the download_root
                zip_ref.extractall(download_root)
            print("Extraction completed.")
        except zipfile.BadZipFile:
            print(f"Could not extract {archive_filename}. The file may be corrupted or not a valid zip file.")
            # Remove the potentially corrupted downloaded file
            os.remove(archive_filename)
            raise

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        # Get image and target of idx
        if self.pin_memory:
            image = self.data[idx]
        else:
            image = self._pil_loader(self.image_paths[idx])
        target = self.targets[idx]

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.targets)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        # Return the fixed class-to-index mapping
        classes = ['Normal', 'Lung_ Opacity', 'COVID', 'Viral Pneumonia']
        class_to_idx = self.class_to_idx
        return classes, class_to_idx

    def _find_paths(self, folder: str, class_to_idx: Dict[str, int]) -> Tuple[List[str], List[int]]:
        # Initialize lists to store image paths and corresponding targets
        image_paths, targets = [], []

        for target_class in class_to_idx.keys():
            class_idx = class_to_idx[target_class]
            target_folder = os.path.join(folder, target_class, 'images')
            for root, _, fnames in sorted(os.walk(target_folder, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        image_paths.append(os.path.join(root, fname))
                        targets.append(class_idx)

        return image_paths, targets

    def _pil_loader(self, path: str) -> Image.Image:
        # Load PIL image according to path
        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    def _load_images(self, mode: str, paths: List[str]) -> List[Image.Image]:
        # Load images according to paths
        images = []
        for path in paths:
            images.append(self._pil_loader(path))
        return images