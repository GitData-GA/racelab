track = rl.load(path="racelab_dev/tests/test_track.npy")
track

rl.plot(track, savefig_options={'fname': 'plot.jpg'})

optimal = rl.optimize(track, 'k1999', line_iterations=100, xi_iterations=5)