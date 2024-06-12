def make_imbalanced_dataset(X, y):
  imbal = [10100, 6000, 3600, 1300]
  new_X =[]
  new_Y =[]
  for c in range(0,4):
    xclass = X[y==c]
    yclass = y[y==c]
    new_X.append(xclass[0:imbal[c]])
    new_Y.append(yclass[0:imbal[c]])
  X_imbal = torch.cat(new_X)
  Y_imbal = torch.cat(new_Y)
  del new_X
  del new_Y
  return X_imbal, Y_imbal