import numpy as np

from fsai.objects.track import Track

# load track track into cone objects
track = Track("examples/data/tracks/laguna_seca.json")
blue_lines, yellow_lines, orange_lines = track.get_boundary()

if __name__ == "__main__":
    from scipy.interpolate import CubicSpline
    import matplotlib.pyplot as plt
    x = np.arange(10)
    y = np.sin(x)
    cs = CubicSpline(x, y)
    xs = np.arange(-0.5, 9.6, 0.1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, y, 'o', label='data')
    ax.plot(xs, cs(xs), label="S")
    ax.set_xlim(-0.5, 9.5)
    ax.legend(loc='lower left', ncol=2)
    plt.show()

# # draw and show the track
# cv2.imshow("Track Boundary Example", render(
#     lines=[
#         ((255, 0, 0), 2, blue_lines),
#         ((0, 255, 255), 2, yellow_lines),
#         ((0, 100, 255), 2, orange_lines),
#     ],
# ))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
