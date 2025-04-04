import h5py

with h5py.File("./cache/libero_goal_embedding.hdf5") as f:
    print(f['embeddings'].keys())
