# Initialize dataset
dec_x_f, dec_y_f = make_imbalanced_dataset(dec_x_f_b, dec_y_f_b)
del dec_x_f_b
del dec_y_f_b
print('dec_y ',dec_y_f.shape)
print(collections.Counter(dec_y_f.numpy()))
print('dec_x',dec_x_f.shape)
print(dec_x_f.shape[0])

