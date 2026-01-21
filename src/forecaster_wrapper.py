# src/forecaster_wrapper.py
import numpy as np
import torch
import torch.nn as nn

class ForecastingWrapper(nn.Module):
    def __init__(
        self,
        base_model,
        norm_params,
        feature_cols,
        device,
        history_len=20,
        pred_steps=20,
        roll_with_predictions=False,
        trend_extrap="linear",
        trend_tail=3,
        delta_next=1.0,
    ):
        super().__init__()
        self.net = base_model.to(device).eval()
        self.norm_params = norm_params
        self.fcols = feature_cols
        self.device = device
        self.H = int(history_len)
        self.K = int(pred_steps)
        self.roll_with_predictions = bool(roll_with_predictions)
        self.trend_extrap = trend_extrap
        self.trend_tail = max(2, int(trend_tail))
        self.delta_next = float(delta_next)

    def _key(self, col: str) -> str:
        return col[:-5] if isinstance(col, str) and col.endswith("_norm") else col

    def _z(self, col, x):
        k = self._key(col)
        mu = self.norm_params[k]["mean"]
        sd = self.norm_params[k]["std"]
        return (x - mu) / (sd + 1e-12)

    def _invz_bg(self, z):
        mu = self.norm_params["bg"]["mean"]
        sd = self.norm_params["bg"]["std"]
        return z * sd + mu

    def _call_net(self, x_hist, t_hist):
        try:
            return self.net(x_hist, t_hist, delta_next=self.delta_next)
        except TypeError:
            return self.net(x_hist, t_hist)

    @torch.no_grad()
    def forward(self, raw_window: np.ndarray, h=None):
        if isinstance(raw_window, torch.Tensor):
            raw_window = raw_window.detach().cpu().numpy()
        B = raw_window.shape[0]

        win_obs = raw_window.copy()
        for j, col in enumerate(self.fcols):
            win_obs[:, :, j] = self._z(col, win_obs[:, :, j])

        x_hist = torch.tensor(win_obs, dtype=torch.float32, device=self.device).permute(1, 0, 2)
        H = x_hist.shape[0]
        t_hist = torch.arange(H, device=self.device).unsqueeze(1).expand(-1, B).unsqueeze(-1).float()

        pred_z = self._call_net(x_hist, t_hist)  # (B,2)
        mean_next = pred_z[:, 0]
        bg_next = self._invz_bg(mean_next).unsqueeze(-1)  # (B,1)

        # Non-AR horizon build (default)
        if not self.roll_with_predictions:
            preds = torch.zeros(B, self.K, 1, device=self.device)
            bg_hist_obs = torch.tensor(raw_window[:, :, 0], dtype=torch.float32, device=self.device)

            if self.trend_extrap == "linear":
                T = min(self.trend_tail, bg_hist_obs.shape[1])
                y = bg_hist_obs[:, -T:]
                x = torch.arange(T, device=self.device).float()
                x = x - x.mean()
                y_centered = y - y.mean(dim=1, keepdim=True)
                denom = (x**2).sum().clamp_min(1e-9)
                slope = (y_centered @ x) / denom
            else:
                slope = torch.zeros(B, device=self.device)

            preds[:, 0, 0] = bg_next[:, 0]
            if self.K > 1:
                ks = torch.arange(1, self.K, device=self.device).float()
                base = bg_next[:, 0].unsqueeze(-1)
                preds[:, 1:, 0] = base + slope.unsqueeze(-1) * ks.unsqueeze(0)
            return preds, None

        # AR loop (optional)
        preds = torch.zeros(B, self.K, 1, device=self.device)
        preds[:, 0, 0] = bg_next[:, 0]

        raw_roll = raw_window.copy()
        raw_roll[:, -1, 0] = bg_next[:, 0].cpu().numpy()
        win_z = win_obs.copy()
        win_z[:, -1, 0] = self._z("bg", raw_roll[:, -1, 0])
        win_z = torch.tensor(win_z, dtype=torch.float32, device=self.device)

        for step in range(1, self.K):
            x_net = win_z.permute(1, 0, 2)
            t_net = torch.arange(x_net.shape[0], device=self.device).unsqueeze(1).expand(-1, B).unsqueeze(-1).float()
            pred_z = self._call_net(x_net, t_net)
            mean_z = pred_z[:, 0]
            bg_next = self._invz_bg(mean_z).squeeze(-1)
            preds[:, step, 0] = bg_next

            raw_roll[:, -1, 0] = bg_next.cpu().numpy()
            new_z_last = win_z[:, -1, :].clone()
            new_z_last[:, 0] = self._z("bg", raw_roll[:, -1, 0])
            win_z = torch.cat([win_z[:, 1:, :], new_z_last.unsqueeze(1)], dim=1)

        return preds, None
