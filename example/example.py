from pic2func import function_from_picture, fourier_function_from_picture
import matplotlib.pyplot as plt

# Get an (x,y) function from a picture.
f = function_from_picture("test.png")  # Nx2 array: (x,y) points

fig, ax = plt.subplots()
ax.plot(f[:,0], f[:,1], c="r", lw=5)
ax.axhline(0, c="k", lw=3)
ax.axvline(0, c="k", lw=3)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.set_title("y=f(x)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
fig.tight_layout()
fig.savefig("my-test-output.png")

# Get an (x,y) function and recreate using 5 terms of the Fourier series.
# (Gibbs phenonemon is 'removed', see source.)
f, xsample, fsample = fourier_function_from_picture("test.png", n=5)

fig, ax = plt.subplots()
ax.plot(f[:,0], f[:,1], c="r", lw=5, label="f(x)")
ax.plot(xsample, fsample, c="b", lw=5, label="Fourier reconstruction")
ax.axhline(0, c="k", lw=3)
ax.axvline(0, c="k", lw=3)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.set_title("y=f(x)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize=15, loc="lower right")
fig.tight_layout()
fig.savefig("my-test-output2.png")
