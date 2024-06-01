class ResNetClassifier_v3(nn.Module):
    def __init__(self, num_classes=4, p=0.1):
        super().__init__()
        self.model = resnet18(pretrained=True)
        # self.model = resnet18(pretrained=False, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modify the final fully connected layer for your number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.dropout = nn.Dropout(p=p)  # Add dropout layer

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)  # Apply dropout before the final layer
        return x

    def train_model(self, trainloader, testloader, num_epochs, learning_rate, data_name, fold_id, device='cuda'):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        best_test_loss = float('inf')
        best_model_weights = None
        early_stop_counter = 0
        early_stop_patience = 10

        train_loss_step = []
        train_acc_step = []
        validation_loss_step = []
        validation_acc_step = []

        for epoch in range(num_epochs):
            train_running_loss = 0.0
            train_correct = 0
            train_total = 0

            self.train()
            for (images, labels) in tqdm(trainloader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.item()
                _, predictions = torch.max(logits, 1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

            train_loss = train_running_loss / len(trainloader)
            train_acc = train_correct / train_total
            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f} Train Acc: {train_acc:.6f}')
            train_loss_step.append(train_loss)
            train_acc_step.append(train_acc)

            test_running_loss = 0.0
            test_correct = 0
            test_total = 0

            self.eval()
            predicttion_1, label_1 = [], []
            with torch.no_grad():
                for (images, labels) in tqdm(testloader):
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = self(images)
                    loss = criterion(logits, labels)

                    test_running_loss += loss.item()
                    _, predictions = torch.max(logits, 1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)
                    predicttion_1.append(predictions)
                    label_1.append(labels)

            test_loss = test_running_loss / len(testloader)
            test_acc = test_correct / test_total
            print(f'Epoch {epoch}: Test Loss: {test_loss:.6f} Test Acc: {test_acc:.6f}')
            validation_loss_step.append(test_loss)
            validation_acc_step.append(test_acc)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_weights = self.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f'Validation loss has not improved for {early_stop_patience} epochs. Stopping early...')
                    break
            lr_scheduler.step(test_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}: Learning Rate: {current_lr:.8f}')

        path = '/content/' + data_name + '/model/' + str(fold_id + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        path_model = path + '/model_medical_v20_1_6.pth'
        torch.save(best_model_weights, path_model)

        self.load_state_dict(best_model_weights)
        self.eval()

        plot_training_curves(train_loss_step, train_acc_step, validation_loss_step, validation_acc_step)

        return train_loss_step, train_acc_step, validation_loss_step, validation_acc_step

    def predict_model(self, inputs, device='cuda'):
        self.eval()
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

def plot_training_curves(train_loss_step, train_acc_step, validation_loss_step, validation_acc_step):
    epochs = range(len(train_loss_step))  # Number of epochs

    plt.figure(figsize=(14, 6))  # Adjust the figure size for side-by-side plots

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_step, 'b-', label='Training Loss')
    plt.plot(epochs, validation_loss_step, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_step, 'g-', label='Training Accuracy')
    plt.plot(epochs, validation_acc_step, 'm-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()
