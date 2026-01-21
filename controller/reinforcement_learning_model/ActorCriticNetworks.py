import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional 


class ActorNetwork(nn.Module):
    """
    PyTorch replica of ScaledController (TensorFlow/Keras).
    - It preserves the same comments and steps.
    - It maintains a "stateful" LSTM: the states (c, h) are saved
      as attributes and are manually reset with `reset_states()`.
    """
    def __init__(self,
                 scaler_params: dict,
                 units: int = 128,
                 alpha: float = 250.0):
        super().__init__()

        # 1) Parámetros del normalizador
        # En __init__ de ActorNetwork y CriticNetwork
        dmin = torch.tensor(scaler_params["data_min"], dtype=torch.float32)
        dmax = torch.tensor(scaler_params["data_max"], dtype=torch.float32)
        scale = torch.tensor(scaler_params["scale"], dtype=torch.float32)
        min_ = torch.tensor(scaler_params["min"], dtype=torch.float32)

        self.register_buffer("_min", min_)
        self.register_buffer("_scale", scale)

        # 2) Umbral para la compuerta
        self.register_buffer("tau_norm", 900.0 * self._scale[4] + self._min[4])
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        # 3) Capas: Usamos nn.LSTM para mayor eficiencia y flexibilidad.
        #    batch_first=True es crucial para trabajar con formas (N, S, D).
        self.lstm = nn.LSTM(input_size=5, hidden_size=units, batch_first=True)
        self.layernorm = nn.LayerNorm(units)
        self.dense = nn.Linear(units, units // 2)
        self.delta = nn.Linear(units // 2, 3)

        # 4) Estado oculto persistente para el modo inferencia.
        self.hidden_state = None


    def _to_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Converts physical variables to the [-1, 1] range."""
        return x * self._scale + self._min

    @torch.no_grad()
    def reset_states(self):
        """Resets the persistent hidden state. It is called at the end of an episode."""
        self.hidden_state = None

    def forward(self,
                x: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes the input, adapting to the operation mode.

        Args:
            x (torch.Tensor): Observation tensor.
                - Inference: (N, 5)
                - Training: (N, S, 5) where S > 1
            hidden_state (tuple, optional): Initial hidden state for TBPTT.
        
        Returns:
            A tuple with (final_actions, new_hidden_state).
        """
        # Normaliza la entrada. La forma de salida coincide con la de entrada.
        x_norm = self._to_norm(x)

        # Distingue entre secuencia de entrenamiento (con hidden_state) y paso de inferencia (stateful).
        is_inference_step = x.dim() == 2

        if is_inference_step:
            # --- MODO INFERENCIA ---
            # x_norm: (N, 5) -> x_step: (N, 1, 5)
            x_step = x_norm.unsqueeze(1)
            
            # Usa y actualiza el estado interno
            lstm_out, self.hidden_state = self.lstm(x_step, self.hidden_state)
            
            # lstm_out: (N, 1, units) -> lo hacemos (N, units)
            lstm_out = lstm_out.squeeze(1)
            new_hidden_state = self.hidden_state
        else:
            # --- MODO ENTRENAMIENTO (SECUENCIA) ---
            # x_norm: (N, S, 5)
            # Usa el estado oculto proporcionado para el TBPTT
            lstm_out, new_hidden_state = self.lstm(x_norm, hidden_state)
            # lstm_out: (N, S, units)

        # A partir de aquí, las operaciones son generales para (..., Features)
        # donde '...' puede ser (N) o (N, S).

        # Capas densas para calcular los deltas de acción.
        h = self.layernorm(lstm_out)    # h: (..., units)
        h = self.dense(h)               # h: (..., units/2)
        deltas = torch.tanh(self.delta(h)) # deltas: (..., 3)
#         print(f"!!!!!!!! deltas: {deltas} !!!!!!!!")

        # Suma residual: se suma el delta a las acciones originales del estado.
        # Las acciones originales son los 3 últimos componentes del estado: mf, brk, ice_sp
        orig = x_norm[..., 2:]          # orig: (..., 3)
        abs_raw = orig + deltas         # abs_raw: (..., 3)
        bounded = torch.tanh(abs_raw)   # bounded: (..., 3)

        # Compuerta para mf (main fuel): si ice_sp es alto, mf debe ser bajo.
        ice_norm = bounded[..., 2:3]    # ice_sp_norm: (..., 1)
        gate = torch.sigmoid((ice_norm - self.tau_norm) * self.alpha) # gate: (..., 1)

        valor_minimo_mf = -1.0
        mf_norm = gate * bounded[..., 0:1] + (1. - gate) * valor_minimo_mf # mf_norm: (..., 1)

        # Salida final: se combinan las acciones calculadas.
        # brk_norm (brake), ice_sp_norm (ice setpoint)
        rest_norm = bounded[..., 1:]    # [brk_norm, ice_sp_norm]: (..., 2)

        # Concatena [mf_norm, brk_norm, ice_sp_norm]
        final_actions = torch.cat([mf_norm, rest_norm], dim=-1) # final_actions: (..., 3)

        return final_actions, new_hidden_state


class CriticNetwork(nn.Module):
    """
    Recurrent critic for RTD3.
    - It processes the observation with its own stateful LSTM.
    - It combines the hidden state with the current action and produces Q1.
    """
    def __init__(self,
                 scaler_params: dict,
                 units: int = 128):
        super().__init__()

        # --- Normalizador idéntico al del actor ---
        scale = torch.tensor(scaler_params["scale"], dtype=torch.float32)
        min_ = torch.tensor(scaler_params["min"], dtype=torch.float32)
        self.register_buffer("_min", min_)
        self.register_buffer("_scale", scale)

        # --- RNN: nn.LSTMCell -> nn.LSTM ---
        self.lstm = nn.LSTM(input_size=5, hidden_size=units, batch_first=True)
        self.layernorm = nn.LayerNorm(units)

        # --- Cabeza de Evaluación Q (salida única) ---
        concat_dim = units + 3  # h_t ⊕ a_t
        hidden_dim = units // 2

        # Definimos una única red MLP (Perceptrón Multicapa) para calcular el valor Q
        self.q_network = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # --- Estado interno persistente para inferencia ---
        self.hidden_state = None

    def _to_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Physical scale -> [-1,1] range."""
        return x * self._scale + self._min

    @torch.no_grad()
    def reset_states(self):
        """Resets the persistent hidden state."""
        self.hidden_state = None


    def forward(self,
                obs: torch.Tensor,
                act: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes the observation and action to produce the Q values.

        Args:
            obs: Observation tensor.
                - Inference: (N, 5)
                - Training: (N, S, 5)
            act: Action tensor.
                - Inference: (N, 3)
                - Training: (N, S, 3)
            hidden_state (tuple, optional): Initial hidden state for TBPTT.

        Returns:
            A tuple with (q_value, new_hidden_state).
        """
        # Normaliza las observaciones.
        obs_norm = self._to_norm(obs)

        # Distingue entre secuencia de entrenamiento y paso de inferencia.
        is_inference_step = obs.dim() == 2

        if is_inference_step:
            # --- MODO INFERENCIA ---
            # obs_norm: (N, 5) -> obs_step: (N, 1, 5)
            obs_step = obs_norm.unsqueeze(1)
            
            # Usa y actualiza el estado interno
            h_out, self.hidden_state = self.lstm(obs_step, self.hidden_state)
            
            # h_out: (N, 1, units) -> lo hacemos (N, units)
            h_out = h_out.squeeze(1)
            new_hidden_state = self.hidden_state
        else:
            # --- MODO ENTRENAMIENTO (SECUENCIA) ---
            # obs_norm: (N, S, 5)
            # Usa el estado oculto proporcionado para el TBPTT
            h_out, new_hidden_state = self.lstm(obs_norm, hidden_state)
            # h_out: (N, S, units)

        # Normaliza la salida de la LSTM
        h_norm = self.layernorm(h_out) # h_norm: (..., units)

        # Concatena el estado oculto con la acción.
        # Funciona para ambos modos gracias a la operación sobre la última dimensión.
        # (..., units) + (..., 3) -> (..., units + 3)
        xu = torch.cat([h_norm, act], dim=-1)

        # --- Cabeza Q ---
        # Pasa la entrada combinada a través de la red Q para obtener el valor
        q_value = self.q_network(xu)

        return q_value, new_hidden_state

    
def copy_target(target: nn.Module, source: nn.Module):
    """
    Copies the parameters from a source network to a target one.
    It is used for the initial "hard update" of the target networks.
    (The implementation you proposed yourself is correct).

    Args:
        target (nn.Module): The target network to which the parameters are copied.
        source (nn.Module): The source network from which the parameters are copied.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)
        
def soft_update(target_model, local_model,tau):
      for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
          target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)