import numpy as np
import matplotlib.pyplot as plt

def own_plot():
    voxels = np.load("batch_0.npy")
    print(voxels.shape)
    r = voxels[0, 0, :, :, :]
    g = voxels[0, 1, :, :, :]
    b = voxels[0, 2, :, :, :]
    a = voxels[0, 3, :, :, :]


    X, Y, Z = np.mgrid[-1.0:1.0:(2.0 / 128), -1.0:1.0:(2.0 / 128), -1.0:1.0:(2.0 / 128)]
    coords = np.array([X, Y, Z])
    # print(coords.shape)
    # print(coords[:, 3, 0, 0])
    # exit()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(r, g, b, coords)
    plt.show()

def example_plot():
    import matplotlib.pyplot as plt
    import numpy as np

    def midpoints(x):
        sl = ()
        for i in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x

    # prepare some coordinates, and attach rgb values to each
    r, g, b = np.indices((17, 17, 17)) / 16.0
    rc = midpoints(r)
    gc = midpoints(g)
    bc = midpoints(b)

    # define a sphere about [0.5, 0.5, 0.5]
    sphere = (rc - 0.5) ** 2 + (gc - 0.5) ** 2 + (bc - 0.5) ** 2 < 0.5 ** 2

    # combine the color components
    colors = np.zeros(sphere.shape + (3,))
    colors[..., 0] = rc
    colors[..., 1] = gc
    colors[..., 2] = bc

    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    print(r.shape)
    print(sphere.shape)
    print(r)
    exit()
    ax.voxels(r, g, b, sphere,
              facecolors=colors,
              edgecolors=np.clip(2 * colors - 0.5, 0, 1),  # brighter
              linewidth=0.5)
    ax.set(xlabel='r', ylabel='g', zlabel='b')
    ax.set_aspect('equal')
    plt.show()

def my_example():
    X, Y, Z = np.mgrid[-1.0:1.0:(2.0 / 128), -1.0:1.0:(2.0 / 128), -1.0:1.0:(2.0 / 128)]
    neuralVolumes = np.load("batch_0.npy")
    first_frame = neuralVolumes[0, :, :, :, :]
    first_frame = first_frame.reshape(128*128*128, 4)
    print(first_frame[0] / 255.0)
    #first_frame = first_frame[:, :, :, 0:3]
    print(first_frame.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=(first_frame / 255.0))
    plt.show()

if __name__ == "__main__":
    #example_plot()
    #my_example()
    fig = plt.figure()
    X, Y, Z = np.mgrid[-1:1:(2.0 / 128), -1:1:(2.0 / 128), -1:1:(2.0 / 128)]
    neuralVolumes = np.load("batch_0.npy")
    first_frame = neuralVolumes[0, :, :, :, :]
    #print(np.max(first_frame))
    first_frame = first_frame.reshape((128 * 128 * 128, 4)) / 255.0
    #print(first_frame.shape)
    first_frame = first_frame.clip(max=1.0, min=0.0)
    #print(first_frame.shape)
    print(np.where(first_frame > 1.0)[0].size)
    #print(first_frame[34])

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=first_frame)
    plt.show()


