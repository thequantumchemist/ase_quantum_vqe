def random_positions_with_min_distance(n_atoms=3, min_dist=0.6, box_size=3.0, max_tries=10000):
    for attempt in range(max_tries):
        # Create n_atoms random positions in a box of size box_size
        positions = np.random.uniform(0, box_size, size=(n_atoms, 3))
        # Calculate all pairwise distances
        dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
        # Ignore self-distances by setting diagonal to a large value
        np.fill_diagonal(dists, np.inf)
        if np.all(dists >= min_dist):
            return positions.tolist()
    raise RuntimeError(f"Failed to place {n_atoms} atoms with min distance {min_dist} after {max_tries} attempts.")

