def train_AE(dataset_torch, fold_id, data_name):
    encoder = Encoder(args)
    decoder = Decoder(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)

    # DataLoader
    dl_batch_size = len(dataset_torch)
    batch_size = args['batch_size']
    num_workers = 2

    train_loader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_aux = torch.utils.data.DataLoader(dataset_torch, batch_size=dl_batch_size, shuffle=True, num_workers=num_workers)
    dec_x, dec_y = next(iter(dl_aux))
    del dl_aux

    best_loss = np.inf
    train_losses = []  # List to store the training loss for each epoch

    if args['train']:
        # weight_decay = 1e-5  # Set weight decay here
        # weight_decay = 0
        # enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'], weight_decay=weight_decay)
        # dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'], weight_decay=weight_decay)

        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])

        # Learning rate scheduler
        # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=20, gamma=0.5)
        # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=20, gamma=0.5)

        for epoch in range(args['epochs']):
            train_loss = 0.0

            encoder.train()
            decoder.train()

            for images, labs in tqdm(train_loader):
                encoder.zero_grad()
                decoder.zero_grad()

                images, labs = images.to(device), labs.to(device)

                with torch.amp.autocast(device_type='cuda'):
                    z_hat = encoder(images)  # Encode
                    x_hat = decoder(z_hat)  # Decode
                    mse = criterion(x_hat, images)  # Compute MSE loss

                    comb_loss = mse
                    comb_loss.backward()

                    enc_optim.step()
                    dec_optim.step()

                train_loss += comb_loss.item()

                # Clear GPU memory
                del images, labs, z_hat, x_hat, mse, comb_loss
                torch.cuda.empty_cache()

            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)  # Append the average training loss for this epoch

            print('Epoch: {} \tTrain Loss: {:.6f}'.format(epoch, train_loss))

            # Save the best model
            if train_loss < best_loss:
                path = '/content/' + data_name + '/model/' + str(fold_id)
                if not os.path.exists(path):
                    os.makedirs(path)

                path_enc = path + '/bst_enc_SMOTEAE_Medical_v20_1_6.pth'
                path_dec = path + '/bst_dec_SMOTEAE_Medical_v20_1_6.pth'
                torch.save(encoder.state_dict(), path_enc)
                torch.save(decoder.state_dict(), path_dec)
                best_loss = train_loss

            # Step the schedulers
            # enc_scheduler.step()
            # dec_scheduler.step()

        # Free memory after training
        del encoder, decoder, enc_optim, dec_optim, train_loader
        torch.cuda.empty_cache()

    print(path_enc)
    print(path_dec)

    del dec_x, dec_y
    torch.cuda.empty_cache()


    # Optionally track total training time
    # t1 = time.time()
    # print('Total time (min): {:.2f}'.format((t1 - t0) / 60))