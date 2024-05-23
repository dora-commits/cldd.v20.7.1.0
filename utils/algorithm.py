def biased_get_class(c, dec_x, dec_y):

    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    return xbeg, ybeg
    #return xclass, yclass

# TODO : SMOTE Algorithm
def SMOTE(X, y,n_to_sample,cl):
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    return samples, [cl]*n_to_sample

# TODO : ADASYN Algorithm
def ADASYN(X, y,xclass, yclass,  cl, m_major, m_minor, beta=1):
  K = 5
  d= m_major/m_minor
  G = (m_major-m_minor)*beta

  # fitting the model
  clf = neighbors.KNeighborsClassifier()
  clf.fit(X, y)
  Ri = []
  Minority_per_xi = []

  for i in range(m_minor):
    # Returns indices of the closest neighbours, and return it as a list
    xi = xclass[i, :].reshape(1, -1)
    # Returns indices of the closest neighbours, and return it as a list
    neighbours = clf.kneighbors(xi, n_neighbors=K, return_distance=False)[0]
    delta=0
    for j in neighbours:
      if(y[j]!=0):
        delta+=1
    Ri.append(delta/K)

    minority = []
    for index in neighbours:
            # Shifted back 1 because indices start at 0
            if y[index]==cl:
                minority.append(index)
    Minority_per_xi.append(minority)

  Ri_norm = []
  for ri in Ri:
    ri_norm = ri / sum(Ri)
    Ri_norm.append(ri_norm)

  Gi = []
  for r in Ri_norm:
    gi = round(r * G)
    Gi.append(int(gi))
  syn_data=[]
  syn_number =0

  for i in range(m_minor):
    neighbor_indices = np.random.choice(list(range(1, K+1)),Gi[i])
    for j in range(Gi[i]):
        # If the minority list is not empty
        if Minority_per_xi[i]:
            index = np.random.choice(Minority_per_xi[i])
            xzi = X[index, :].reshape(1, -1)
            si = xi + (xzi - xi) * np.random.uniform(0, 1)
            syn_data.append(si)
            syn_number+=1
  return syn_data, [cl]*syn_number