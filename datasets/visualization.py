# dec_x_f = dec_x_f.reshape(dec_x_f.shape[0],3, 32, 32)
print('dec_y ',dec_y_f.shape)
print(collections.Counter(dec_y_f.numpy()))
print('train imgs after reshape ',dec_x_f.shape)
print(dec_x_f.shape[0])

classes = ['0', '1', '2', '3']
num_classes = len(classes)
samples_per_class = 10
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(dec_y_f == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        img=image.array_to_img((dec_x_f[idx].permute(1, 2, 0)), scale=True)
        plt.imshow(img)
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()