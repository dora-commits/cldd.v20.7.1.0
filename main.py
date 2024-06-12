from collections import Counter
if not args['pre_trained']:
    for repeats in range(0,num_repeats):

        acs_array_smote     = np.zeros(1)
        gm_array_smote      = np.zeros(1)
        f1macro_array_smote = np.zeros(1)
        precision_smote     = np.zeros(1)
        recall_smote        = np.zeros(1)

        for i in range(0, 1):

            print('Fold {}'.format(i + 1))
            print(f"x_train type: {type(dec_x_f)}, shape: {dec_x_f.shape}")
            print(f"y_train type: {type(dec_y_f)}, shape: {dec_y_f.shape}")

            # Define the number of samples per class for the test set
            test_samples_per_class = 100

            # Separate the dataset by class
            class_indices = {label: np.where(dec_y_f == label)[0] for label in np.unique(dec_y_f)}

            # Check that each class has enough samples
            for label, indices in class_indices.items():
                if len(indices) < test_samples_per_class:
                    raise ValueError(f"Not enough samples for class {label}. Required: {test_samples_per_class}, Available: {len(indices)}")

            test_indices = []
            for label, indices in class_indices.items():
                sampled_indices = np.random.choice(indices, test_samples_per_class, replace=False)
                test_indices.extend(sampled_indices)

            test_indices = np.array(test_indices)

            # Create the test set
            x_test = dec_x_f[test_indices]
            y_test = dec_y_f[test_indices]

            # Create the training set by excluding the test set indices
            train_indices = np.setdiff1d(np.arange(len(dec_x_f)), test_indices)
            x_classification = dec_x_f[train_indices]
            y_classification = dec_y_f[train_indices]

            # Print class distributions to verify
            print("Class distribution in test set:", dict(Counter(y_test.numpy())))
            print("Class distribution in training set:", dict(Counter(y_classification.numpy())))

            testX = torch.tensor(x_test, dtype=torch.float32)
            testY = torch.tensor(y_test, dtype=torch.long)

            dataset_train = torch.utils.data.TensorDataset(x_classification, y_classification)

            t0 = time.time()
            train_AE(dataset_train, i, args['data_name'])
            t1 = time.time()

            t2 = time.time()
            combx, comby = GenerateSamples(dataset_train, i, args['data_name'])
            t3 = time.time()

            t4 = time.time()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ResNetClassifier_v3()
            model.to(device)

            combx = combx.reshape(combx.shape[0],3, 64, 64)
            # combx = combx.reshape(combx.shape[0],3, 32, 32)

            combx = torch.Tensor(combx)
            comby = torch.tensor(comby,dtype=torch.long)
            combx = combx.to(device)
            comby = comby.to(device)
            combx = combx.cpu().numpy()
            comby = comby.cpu().numpy()

            trainX_imbal = combx
            trainY_imbal = comby

            x_train, x_validation, y_train, y_validation = train_test_split(trainX_imbal, trainY_imbal, test_size=0.2, random_state=0, stratify=trainY_imbal)
            # print('x_train', x_train)
            # print('y_train', y_train)
            print(f"x_train type: {type(x_train)}, shape: {x_train.shape}")
            print(f"y_train type: {type(y_train)}, shape: {y_train.shape}")
            print(f"x_train type: {type(x_validation)}, shape: {x_validation.shape}")
            print(f"y_train type: {type(y_validation)}, shape: {y_validation.shape}")
            print("Class distribution in training set:", dict(Counter(y_train)))
            print("Class distribution in validation set:", dict(Counter(y_validation)))

            # Assuming x_train and y_train are NumPy arrays
            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)

            x_validation = torch.tensor(x_validation, dtype=torch.float32)
            y_validation = torch.tensor(y_validation, dtype=torch.long)

            print(f"x_train type after: {type(x_train)}, shape: {x_train.shape}")
            print(f"y_train type after: {type(y_train)}, shape: {y_train.shape}")

            train_set = TensorDataset(x_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'],shuffle=True,num_workers=1)

            validation_set = TensorDataset(x_validation, y_validation)
            validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args['batch_size'],shuffle=True,num_workers=1)

            train_loss_step, train_acc_step, validation_loss_step, validation_acc_step = model.train_model(train_loader, validation_loader, args['resnet_num_epochs'], args['resnet_learning_rate'], args['data_name'], i)
            t5 = time.time()

            train_loss_total.append(train_loss_step)
            train_acc_total.append(train_acc_step)
            validation_loss_total.append(validation_loss_step)
            validation_acc_total.append(validation_acc_step)

            t6 = time.time()
            resnet_outputs = model.predict_model(testX.to(device))
            torch.cuda.empty_cache()
            #
            # from PIL import ImageOps
            # import keras.utils as image
            # import matplotlib.pyplot as plt

            # class_labels = ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia']

            # combx = testX
            # comby = testY
            # combx = combx.reshape(combx.shape[0], 3, 64, 64)

            # count = 0
            # img_list = []
            # num = 0

            # for i in range(resnet_outputs.shape[0]):
            #     img = image.array_to_img((combx[i].permute(1, 2, 0)), scale=True)

            #     if torch.eq(torch.argmax(resnet_outputs[i]), testY[i]):
            #         if torch.eq(testY[i], torch.tensor(0)):
            #             img_with_border = ImageOps.expand(img, border=2, fill='blue')
            #         else:
            #             img_with_border = ImageOps.expand(img, border=2, fill='green')
            #     else:
            #         img_with_border = ImageOps.expand(img, border=2, fill='red')

            #     img_list.append([img_with_border, torch.eq(testY[i], torch.argmax(resnet_outputs[i]))])

            #     count += 1
            #     if count == 10:
            #         figure, axis = plt.subplots(1, 10, figsize=(15, 3))
            #         for j in range(10):
            #             axis[j].imshow(img_list[j][0])
            #             title = f"{class_labels[testY[i]]}: {num + j + 1}" if img_list[j][1] else f"Fail: {num + j + 1}"
            #             axis[j].set_title(title)
            #             axis[j].axis('off')
            #         plt.tight_layout()
            #         plt.show()
            #         count = 0
            #         img_list.clear()
            #         num += 10

            #
            y_predict = [torch.argmax(resnet_outputs[i]).item() for i in range(resnet_outputs.shape[0])]
            y_true = [testY[i].item() for i in range(resnet_outputs.shape[0])]
            t7 = time.time()

            train_model_time[repeats]           = np.round((t1 - t0)/60, 5)
            generate_samples_time[repeats]      = np.round((t3 - t2)/60, 5)
            train_classification_time[repeats]  = np.round((t5 - t4)/60, 5)
            prediction_time[repeats]            = np.round((t7 - t6)/60, 5)

            time_df.loc[len(time_df.index)] = [train_model_time[repeats],generate_samples_time[repeats],train_classification_time[repeats], prediction_time[repeats]]

            # fig, ax = plt.subplots()
            fig, ax = plt.subplots(figsize=(15, 15))
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=time_df.values, colLabels=time_df.columns, loc='center')
            fig.tight_layout()
            plt.show()

            acc = accuracy_score(y_true, y_predict)
            gm = geometric_mean_score(y_true, y_predict, average='macro')
            f1_macro = f1_score(y_true, y_predict, average='macro')
            precision = precision_score(y_true, y_predict, average='macro')
            recall = recall_score(y_true, y_predict, average='macro')
            cm = confusion_matrix(y_true, y_predict)

            ax= plt.subplot()
            sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

            # labels, title and ticks
            ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
            ax.set_title('Confusion Matrix');
            ax.xaxis.set_ticklabels(['0', '1', '2', '3']); ax.yaxis.set_ticklabels(['0', '1', '2', '3']);
            plt.show()

            # print(i)
            acs_array_smote[i] = acc
            gm_array_smote[i] = gm
            f1macro_array_smote[i] = f1_macro
            precision_smote[i] = precision
            recall_smote[i] = recall

        Accuracy[repeats] = acs_array_smote.mean()
        GeometricMean[repeats] = gm_array_smote.mean()
        F1Score[repeats] = f1macro_array_smote.mean()
        Precision[repeats] = precision_smote.mean()
        Recall[repeats] = recall_smote.mean()

        result_df.loc[len(result_df.index)] = [Accuracy[repeats],GeometricMean[repeats],F1Score[repeats], Precision[repeats], Recall[repeats]]

        # fig_2, ax_2 = plt.subplots()
        fig_2, ax_2 = plt.subplots(figsize=(15, 15))
        fig_2.patch.set_visible(False)
        ax_2.axis('off')
        ax_2.axis('tight')
        ax_2.table(cellText=result_df.values, colLabels=result_df.columns, loc='center')
        fig_2.tight_layout()
        plt.show()

        file_name = 'Metrics_{}_{}.csv'.format(args['data_name'], args['oversampling_method'])
        dst = '/content/' + args['data_name'] + '/' + file_name
        result_df.to_csv(file_name, mode='a',index=False, header=args['header'])
        shutil.copyfile(file_name, dst)
     