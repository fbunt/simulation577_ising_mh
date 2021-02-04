import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


UP = 1
DOWN = -1


class IsingSim:
    def __init__(self, L, energy):
        self.L = L
        self.N = L * L
        self.dem_energy_dist = np.zeros(self.N, dtype=int)
        self.target_energy = energy
        self.sys_energy = 0
        self.dem_energy = 0
        self.mcs = 0
        self.sys_energy_acc = 0
        self.dem_energy_acc = 0
        self.magnetization = 0
        self.m_acc = 0
        self.m2_acc = 0
        self.accepted_moves = 0
        self.temperature = 0

        self.lattice = np.full((L, L), UP, dtype=np.int8)

        nx = [1, -1, 0, 0]
        ny = [0, 0, 1, -1]
        # Lookup table of neighbor direction vectors
        self.dirs = np.array(list(zip(nx, ny)))
        self.irand = 0
        self.nrand = 1024 * 10
        self.rand_pts = np.random.randint(self.L, size=(self.nrand, 2))

        self.init()

    def reset_acc(self):
        self.sys_energy_acc = 0
        self.dem_energy_acc = 0
        self.m_acc = 0
        self.m2_acc = 0
        self.mcs = 0

    def init(self):
        tries = 0
        energy = -self.N
        mag = self.N
        max_tries = 10 * self.N
        while energy < self.target_energy and tries < max_tries:
            pt = self.get_random_pt()
            de = self.get_delta(pt)
            if de > 0:
                energy += de
                spin = -self.lattice[pt]
                self.lattice[pt] = spin
                mag += 2 * spin
            tries += 1
        self.sys_energy = energy
        self.magnetization = mag

    def step(self):
        for i in range(self.N):
            pt = self.get_random_pt()
            de = self.get_delta(pt)
            if de <= self.dem_energy:
                spin = -self.lattice[pt]
                self.lattice[pt] = spin
                self.accepted_moves += 1
                self.sys_energy += de
                self.dem_energy -= de
                self.magnetization += 2 * spin
            self.sys_energy_acc += self.sys_energy
            self.dem_energy_acc += self.dem_energy
            self.m_acc += self.magnetization
            self.m2_acc += self.magnetization * self.magnetization
        self.mcs += 1
        self.temperature = 4.0 / np.log(
            1 + (4 * self.mcs * self.N / self.dem_energy_acc)
        )

    def get_delta(self, pt):
        # (-1, 0) and (0, -1) allow periodic wrapping from the left and top
        # edges to the right and bottom edges. (1, 0) and (0, 1) don't.
        nn = pt + self.dirs
        # Enforce periodic condition for (1, 0) and (0, 1)
        nn[nn == self.L] = 0
        # Use indexing tricks to get neighbors and sum them
        de = 2 * self.lattice[pt] * self.lattice[nn.T[0], nn.T[1]].sum()
        return de

    def get_random_pt(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.rand_pts = np.random.randint(self.L, size=(self.nrand, 2))
        pt = self.rand_pts[self.irand]
        self.irand += 1
        return tuple(pt)


class SimPlotter:
    def __init__(self, sim, fig):
        self.sim = sim
        self.fig = fig
        self.im = None
        self.steps = []
        self.data = {
            "sys_energy": [],
            "dem_energy": [],
            "mag": [],
            "tmp": [],
        }
        self.min_max = {
            "sys_energy": [self.sim.sys_energy, self.sim.sys_energy],
            "dem_energy": [self.sim.dem_energy, self.sim.dem_energy],
            "mag": [self.sim.magnetization, self.sim.magnetization],
            "tmp": [self.sim.temperature, self.sim.temperature],
        }
        gs = GridSpec(2, 2)
        self.im_ax = self.fig.add_subplot(gs[0, 0])
        self.mag_ax = self.fig.add_subplot(gs[1, 0])
        self.sys_ax = self.fig.add_subplot(gs[0, 1])
        self.tmp_ax = self.fig.add_subplot(gs[1, 1])
        self.im = self.im_ax.imshow(
            np.full_like(self.sim.lattice, UP),
            interpolation="none",
            animated=True,
            cmap="gray",
            vmin=DOWN,
            vmax=UP,
        )
        (self.mag_line,) = self.mag_ax.plot(self.steps, self.data["mag"])
        self.mag_ax.set_title("Magnetization")
        self.mag_ax.set_xlabel("Steps")
        (self.sys_line,) = self.sys_ax.plot(
            self.steps, self.data["dem_energy"]
        )
        self.sys_ax.set_title("System Energy")
        self.sys_ax.set_xlabel("Steps")
        (self.tmp_line,) = self.tmp_ax.plot(self.steps, self.data["tmp"])
        self.tmp_ax.set_title("Temperature")
        self.tmp_ax.set_xlabel("Steps")
        self.axs = (self.mag_ax, self.sys_ax, self.tmp_ax)
        for ax in self.axs:
            ax.grid()
        self.artists = [self.im, self.mag_line, self.sys_line, self.tmp_line]
        plt.tight_layout()

    def _update_data(self, k, v):
        self.data[k].append(v)
        mm = self.min_max[k]
        if v < mm[0]:
            mm[0] = v
        if v > mm[1]:
            mm[1] = v

    def init(self):
        return self.artists

    def update(self, step):
        self.steps.append(step)
        self.sim.step()
        self._update_data("sys_energy", self.sim.sys_energy)
        self._update_data("dem_energy", self.sim.dem_energy)
        self._update_data("mag", self.sim.magnetization)
        self._update_data("tmp", self.sim.temperature)
        if step == 1:
            # Drop initial zero values
            tmp = self.data["tmp"]
            self.min_max["tmp"] = [min(tmp), max(tmp)]

        self.im.set_data(self.sim.lattice)
        self.mag_line.set_data(self.steps, self.data["mag"])
        self.sys_line.set_data(self.steps, self.data["sys_energy"])
        self.tmp_line.set_data(self.steps, self.data["tmp"])
        self.mag_ax.set_ylim(*self.min_max["mag"])
        # Add buffer so that data isn't obscured by axes lines
        mm = self.min_max["sys_energy"]
        self.sys_ax.set_ylim(mm[0] - 1, mm[1] + 1)
        self.tmp_ax.set_ylim(*self.min_max["tmp"])
        self.mag_ax.set_xlim(-1, step)
        self.sys_ax.set_xlim(-1, step)
        self.tmp_ax.set_xlim(-1, step)
        return self.artists


class SimAnimation:
    def __init__(
        self,
        sim,
        interval,
        artist_class,
        max_iter=None,
        figsize=None,
        notebook=False,
    ):
        self.notebook = notebook
        self.max_iter = max_iter
        self.sim = sim
        self.im = None
        self.ani = None
        self.fig = plt.figure(figsize=figsize)
        self.interval = interval
        self.paused = False
        self.artist = artist_class(sim, self.fig)

    def init(self):
        return self.artist.init()

    def update(self, step):
        return self.artist.update(step)

    def on_click(self, event):
        """Toggle play/pause with space bar. Handy for non-jupyter runs."""
        if event.key != " ":
            return
        if self.paused:
            self.ani.event_source.start()
            self.paused = False
        else:
            self.ani.event_source.stop()
            self.paused = True

    def run(self):
        self.fig.canvas.mpl_connect("key_press_event", self.on_click)
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.max_iter,
            init_func=self.init,
            interval=self.interval,
        )
        if not self.notebook:
            plt.show()
        else:
            return HTML(self.ani.to_html5_video())


def run_sim(sim, iters):
    for _ in range(iters):
        sim.step()


if __name__ == "__main__":
    sim = IsingSim(50, 100)
    ani = SimAnimation(sim, 100, SimPlotter)
    ani.run()
