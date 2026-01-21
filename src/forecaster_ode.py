# src/forecaster_ode.py
import torch
import torch.nn as nn

from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        return self.net(z)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-3, atol=1e-4, method="rk4"):
        super().__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def forward(self, z0, t0: torch.Tensor, t1: torch.Tensor):
        t = torch.stack([t0, t1], dim=0).reshape(-1)
        z_traj = odeint(self.odefunc, z0, t, rtol=self.rtol, atol=self.atol, method=self.method)
        return z_traj[-1]

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x, t):
        dt = t.clone()
        dt[1:] = t[1:] - t[:-1]
        dt[0] = 0.0
        xt = torch.cat((x, dt), dim=-1)
        _, hT = self.rnn(xt.flip(0))
        stats = self.hid2lat(hT[0])
        z_mean = stats[:, :self.latent_dim]
        z_logvar = stats[:, self.latent_dim:]
        return z_mean, z_logvar

class NeuralODEForecast(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, latent_dim=6, out_dim=2,
                 ode_hidden=128, solver="rk4", rtol=1e-3, atol=1e-4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = RNNEncoder(input_dim, hidden_dim, latent_dim)
        self.odefunc = ODEFunc(latent_dim, hidden_dim=ode_hidden)
        self.odeblock = ODEBlock(self.odefunc, method=solver, rtol=rtol, atol=atol)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x_history, t_history, delta_next: float = 1.0):
        z_mean, _ = self.encoder(x_history, t_history)
        z0 = z_mean
        t_T = t_history[-1, 0, 0]
        t0 = t_T
        t1 = t_T + x_history.new_tensor(float(delta_next))
        zT = self.odeblock(z0, t0, t1)
        return self.decoder(zT)
