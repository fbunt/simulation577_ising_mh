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


ENERGY = "Energy"
ENERGY_AVG = "<E> / N"
E2_AVG = "<E^2> / N"
MAG = "Magnetization"
MAG_AVG = "<M>"
M2_AVG = "<M^2>"
HEAT_CAP = "Heat Capacity"
SPEC_HEAT = "Specific Heat"
SUS = "$\\chi / N$"
P_ACC = "P_acc"
P_ACC_AVG = "<P_acc>"


class SimPlotter:
    def __init__(self, sim, fig, keys):
        self.sim = sim
        self.fig = fig
        self.keys = keys
        self.im = None
        self.steps = []
        self.fetchers = {
            ENERGY: lambda: self.sim.energy,
            ENERGY_AVG: lambda: self.sim.energy_acc
            / (self.sim.N * self.sim.mcs),
            E2_AVG: lambda: self.sim.e2_acc / (self.sim.N * self.sim.mcs),
            MAG: lambda: self.sim.mag,
            MAG_AVG: lambda: self.sim.mag_acc / self.sim.mcs,
            M2_AVG: lambda: self.sim.m2_acc / self.sim.mcs,
            HEAT_CAP: lambda: self.sim.heat_capacity(),
            SPEC_HEAT: lambda: self.sim.specific_heat(),
            SUS: lambda: self.sim.susceptibility(),
            P_ACC: lambda: self.sim.p_accepted,
            P_ACC_AVG: lambda: self.sim.p_accepted_acc / self.sim.mcs,
        }
        self.data = {k: [] for k in self.fetchers}
        self.min_max = {k: [f(), f()] for (k, f) in self.fetchers.items()}
        n = len(self.keys) + 1
        if n <= 2:
            nr = 1
            nc = 2
        elif n <= 4:
            nr = 2
            nc = 2
        elif n <= 6:
            nr = 3
            nc = 2
        elif n <= 9:
            nr = 3
            nc = 3
        gs = GridSpec(nr, nc)
        self.im_ax = self.fig.add_subplot(gs[0, 0])
        self.axes = {}
        k = -1
        for i in range(nr):
            for j in range(nc):
                if k < 0:
                    k += 1
                    continue
                if k >= len(self.keys):
                    break
                key = self.keys[k]
                self.axes[key] = self.fig.add_subplot(gs[i, j])
                k += 1
        self.im = self.im_ax.imshow(
            np.full_like(self.sim.lattice, UP),
            interpolation="none",
            animated=True,
            cmap="gray",
            vmin=DOWN,
            vmax=UP,
        )
        self.lines = {}
        for key in self.keys:
            (line,) = self.axes[key].plot(self.steps, self.data[key])
            self.axes[key].set_ylabel(key)
            self.axes[key].set_xlabel("Steps")
            self.lines[key] = line
        for ax in self.axes.values():
            ax.grid()
        self.artists = [self.im]
        self.artists.extend(self.lines.values())
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
        for k in self.keys:
            self._update_data(k, self.fetchers[k]())
        if step == 1:
            # Reset min/max to actual data
            for k in self.keys:
                self.min_max[k] = list(self.data[k])

        self.im.set_data(self.sim.lattice)
        for k, line in self.lines.items():
            line.set_data(self.steps, self.data[k])
            self.axes[k].set_ylim(*self.min_max[k])
            self.axes[k].set_xlim(-1, step)
        return self.artists


class SimAnimation:
    def __init__(
        self,
        sim,
        interval,
        artist_class,
        keys,
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
        self.artist = artist_class(sim, self.fig, keys)

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
    sim = IsingSim(32, 2)
    sim.step()
    ani = SimAnimation(
        sim,
        100,
        SimPlotter,
        [
            MAG_AVG,
            ENERGY_AVG,
            SPEC_HEAT,
            SUS,
            P_ACC_AVG,
        ],
    )
    ani.run()
