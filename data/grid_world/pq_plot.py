import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import numpy as np
import scienceplots
from PIL import Image

goal_state = (4, 3)
terminal_states = [(2, 4), (4, 0)]

# Import parquet file
df_first = pd.read_parquet("./mc-epsilon_greedy-first.parquet")
df_test = pd.read_parquet("./mc-epsilon_greedy-test.parquet")
df_length = pd.read_parquet("./mc-epsilon_greedy-length.parquet")

# Prepare Data to Plot
x_first = df_first["episode_x"].to_numpy(dtype=np.int32)
y_first = df_first["episode_y"].to_numpy(dtype=np.int32)
r_first = df_first["reward"].to_numpy(dtype=np.float64)

x_test = df_test["episode_x"]
y_test = df_test["episode_y"]
r_test = df_test["reward"]

length = df_length["length"]
x_length = np.arange(1, len(length) + 1)

print(df_first)
print(df_test)

# Plot First
#
# 5 x 5 Grid (x_first as x coordinate and y_first as y coordinate)
# Make GIF
frames = []
for idx in range(1, len(x_first) + 1):
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.autoscale(tight=True)
        ax.set_title("First Visit")
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)

        for i in range(6):
            ax.axhline(i - 0.5, color="black", lw=0.5)
            ax.axvline(i - 0.5, color="black", lw=0.5)

        # Color the goal state in green
        ax.add_patch(
            Rectangle(
                (goal_state[0] - 0.5, goal_state[1] - 0.5),
                1,
                1,
                color="green",
                alpha=0.5,
            )
        )

        # Color the terminal states as red
        for term in terminal_states:
            ax.add_patch(
                Rectangle(
                    (term[0]-0.5, term[1]-0.5),
                    1,
                    1,
                    color='red',
                    alpha=0.5
                )
            )

        if idx == 1:
            ax.scatter(x_first[0], y_first[0], color="blue", s=100)
        else:
            ax.scatter(x_first[: idx - 1], y_first[: idx - 1], color="gray", s=100)
            ax.scatter(x_first[idx - 1], y_first[idx - 1], color="blue", s=100)

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer._renderer)
        frames.append(Image.fromarray(image))
        plt.close(fig)

frames[0].save(
    "episode_first.gif", save_all=True, append_images=frames[1:], duration=300, loop=0
)

# Plot test
frames = []
for idx in range(1, len(x_test) + 1):
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.autoscale(tight=True)
        ax.set_title("Test")
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)

        for i in range(6):
            ax.axhline(i - 0.5, color="black", lw=0.5)
            ax.axvline(i - 0.5, color="black", lw=0.5)

        # Color the goal state in green
        ax.add_patch(
            Rectangle(
                (goal_state[0] - 0.5, goal_state[1] - 0.5),
                1,
                1,
                color="green",
                alpha=0.5,
            )
        )

        # Color the terminal states as red
        for term in terminal_states:
            ax.add_patch(
                Rectangle(
                    (term[0]-0.5, term[1]-0.5),
                    1,
                    1,
                    color='red',
                    alpha=0.5
                )
            )

        if idx == 1:
            ax.scatter(x_test[0], y_test[0], color="blue", s=100)
        else:
            ax.scatter(x_test[: idx - 1], y_test[: idx - 1], color="gray", s=100)
            ax.scatter(x_test[idx - 1], y_test[idx - 1], color="blue", s=100)

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer._renderer)
        frames.append(Image.fromarray(image))
        plt.close(fig)

frames[0].save(
    "episode_test.gif", save_all=True, append_images=frames[1:], duration=300, loop=0
)

# Plot Episode Length
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length")
    ax.set_yscale("log")

    ax.plot(x_length, length)
    ax.grid()
    plt.savefig("episode_length.png", dpi=600, bbox_inches="tight")

