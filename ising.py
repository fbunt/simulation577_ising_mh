import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


UP = 1
DOWN = -1

CRIT_TEMP = 2 / np.log(1 + np.sqrt(2))


class RandomPointPool:
    def __init__(self, low, hi):
        assert low < hi, "Low must be less than hi"
        self.low = low
        self.hi = hi
        self.irand = 0
        self.nrand = 1024 * 100
        self.shape = (self.nrand, 2)
        self.vals = np.random.randint(self.low, self.hi, size=self.shape)

    def __call__(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.vals = np.random.randint(self.low, self.hi, size=self.shape)
        pt = self.vals[self.irand]
        self.irand += 1
        return tuple(pt)


class RandomPool:
    def __init__(self):
        self.irand = 0
        self.nrand = 1024 * 10
        self.vals = np.random.rand(self.nrand)

    def __call__(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.vals = np.random.rand(self.nrand)
        v = self.vals[self.irand]
        self.irand += 1
        return v


class IsingSim:
    def __init__(self, L, temp=CRIT_TEMP):
        self.L = L
        self.N = L * L
        self.temp = temp
        self.energy = 0
        self.energy_acc = 0
        self.e2_acc = 0
        self.mag = 0
        self.mag_acc = 0
        self.m2_acc = 0
        self.accepted_moves = 0
        self.p_accepted = 0
        self.p_accepted_acc = 0
        self.mcs = 0

        self.lattice = np.full((L, L), UP, dtype=np.int8)

        nx = [1, -1, 0, 0]
        ny = [0, 0, 1, -1]
        # Lookup table of neighbor direction vectors
        self.dirs = np.array(list(zip(nx, ny)))
        self.rand_pt = RandomPointPool(0, self.L)
        self.rand = RandomPool()
        self.w = np.zeros(9)
        self.init()

    def reset_acc(self):
        self.energy_acc = 0
        self.e2_acc = 0
        self.mag_acc = 0
        self.m2_acc = 0
        self.mcs = 0
        self.accepted_moves = 0
        self.p_accepted = 0
        self.p_accepted_acc = 0

    def init(self):
        self.energy = -2 * self.N
        self.mag = self.N
        self.reset_acc()
        self.w[8] = np.exp(-8 / self.temp)
        self.w[4] = np.exp(-4 / self.temp)

    def heat_capacity(self):
        n = self.mcs or 1
        e2_avg = self.e2_acc / n
        e_avg = self.energy_acc / n
        hcap = (e2_avg - (e_avg * e_avg)) / (self.temp * self.temp)
        return hcap

    def specific_heat(self):
        return self.heat_capacity() / self.N

    def susceptibility(self):
        n = self.mcs or 1
        m2_avg = self.m2_acc / n
        m_avg = self.mag_acc / n
        sus = (m2_avg - (m_avg * m_avg)) / (self.temp * self.N)
        return sus

    def step(self):
        accepted = 0
        for i in range(self.N):
            pt = self.rand_pt()
            de = self.get_delta(pt)
            if de <= 0 or self.w[de] > self.rand():
                spin = -self.lattice[pt]
                self.lattice[pt] = spin
                accepted += 1
                self.energy += de
                self.mag += 2 * spin
            self.energy_acc += self.energy
            self.e2_acc += self.energy * self.energy
            self.mag_acc += self.mag
            self.m2_acc += self.mag * self.mag
        self.accepted_moves += accepted
        self.p_accepted = accepted / self.N
        self.p_accepted_acc += self.p_accepted
        self.mcs += 1

    def get_delta(self, pt):
        # (-1, 0) and (0, -1) allow periodic wrapping from the left and top
        # edges to the right and bottom edges. (1, 0) and (0, 1) don't.
        nn = pt + self.dirs
        # Enforce periodic condition for (1, 0) and (0, 1)
        nn[nn == self.L] = 0
        # Use indexing tricks to get neighbors and sum them
        de = 2 * self.lattice[pt] * self.lattice[nn.T[0], nn.T[1]].sum()
        return de


class SimPlotter:
    def __init__(self, sim, fig):
        self.sim = sim
        self.fig = fig
        self.im = None
        self.steps = []
        self.data = {"energy": [], "mag": [], "sheat": []}
        self.min_max = {
            "energy": [self.sim.energy, self.sim.energy],
            "mag": [self.sim.mag, self.sim.mag],
            "sheat": [self.sim.specific_heat(), self.sim.specific_heat()],
        }
        gs = GridSpec(2, 2)
        self.im_ax = self.fig.add_subplot(gs[0, 0])
        self.mag_ax = self.fig.add_subplot(gs[1, 0])
        self.energy_ax = self.fig.add_subplot(gs[0, 1])
        self.sheat_ax = self.fig.add_subplot(gs[1, 1])
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
        (self.energy_line,) = self.energy_ax.plot(
            self.steps, self.data["energy"]
        )
        self.energy_ax.set_title("System Energy")
        self.energy_ax.set_xlabel("Steps")
        (self.sheat_line,) = self.sheat_ax.plot(self.steps, self.data["sheat"])
        self.sheat_ax.set_title("Specific Heat")
        self.sheat_ax.set_xlabel("Steps")
        self.axs = (self.mag_ax, self.energy_ax, self.sheat_ax)
        for ax in self.axs:
            ax.grid()
        self.artists = [
            self.im,
            self.mag_line,
            self.energy_line,
            self.sheat_line,
        ]
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
        self._update_data("energy", self.sim.energy)
        self._update_data("mag", self.sim.mag)
        self._update_data("sheat", self.sim.specific_heat())
        if step == 1:
            for k in self.data:
                self.min_max[k] = list(self.data[k])

        self.im.set_data(self.sim.lattice)
        self.mag_line.set_data(self.steps, self.data["mag"])
        self.energy_line.set_data(self.steps, self.data["energy"])
        self.sheat_line.set_data(self.steps, self.data["sheat"])

        self.mag_ax.set_ylim(*self.min_max["mag"])
        self.energy_ax.set_ylim(*self.min_max["energy"])
        self.sheat_ax.set_ylim(*self.min_max["sheat"])

        self.mag_ax.set_xlim(-1, step)
        self.energy_ax.set_xlim(-1, step)
        self.sheat_ax.set_xlim(-1, step)
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
    sim = IsingSim(20, CRIT_TEMP)
    sim.step()
    ani = SimAnimation(sim, 100, SimPlotter)
    ani.run()
