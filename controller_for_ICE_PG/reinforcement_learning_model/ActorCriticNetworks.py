import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional 
class ActorNetwork(nn.Module):
    def __init__(self, scaler_params: dict, units: int = 128, alpha: float = 0.0): 
        # Nota: alpha ya no se usa, pero lo dejo en args por compatibilidad
        super().__init__()
        # --- Normalización (Igual que antes) ---
        dmin = torch.tensor(scaler_params["data_min"], dtype=torch.float32)
        dmax = torch.tensor(scaler_params["data_max"], dtype=torch.float32)
        scale = torch.tensor(scaler_params["scale"], dtype=torch.float32)
        min_ = torch.tensor(scaler_params["min"], dtype=torch.float32)
        
        self.register_buffer("_min", min_)
        self.register_buffer("_scale", scale)
        # --- Arquitectura LSTM ---
        self.lstm = nn.LSTM(input_size=6, hidden_size=units, batch_first=True)  # 6 dims: vel_target, vel, mf, brk, ice_sp, error
        self.layernorm = nn.LayerNorm(units)
        
        # --- Cabezal de Acción Directa ---
        self.dense = nn.Linear(units, units) 
        self.activation = nn.ReLU()
        # Salida directa a 3 acciones (MF, BRK, ICE_SP)
        self.output_layer = nn.Linear(units, 3) 
        
        # CORRECCIÓN CRÍTICA: Inicialización conservadora para prevenir saturación
        # Valores pequeños → pre-activaciones cerca de 0 → tanh(0) ≈ 0
        # Esto previene que el Actor empiece saturado en ±1
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)
        self.hidden_state = None
    def _to_norm(self, x: torch.Tensor) -> torch.Tensor:
        # Solo normalizar las primeras 5 dimensiones (el error ya está normalizado)
        x_first5 = x[..., :5] * self._scale + self._min
        x_error = x[..., 5:6]  # Error ya normalizado, pasar directamente
        return torch.cat([x_first5, x_error], dim=-1)
    @torch.no_grad()
    def reset_states(self):
        self.hidden_state = None
    def forward(self, x: torch.Tensor, hidden_state=None):
        # 1. Normalizar entrada (primeras 5 dims + error ya normalizado)
        x_norm = self._to_norm(x)
        # 2. Gestión LSTM (Igual que antes)
        is_inference_step = x.dim() == 2
        if is_inference_step:
            x_step = x_norm.unsqueeze(1)
            lstm_out, self.hidden_state = self.lstm(x_step, self.hidden_state)
            lstm_out = lstm_out.squeeze(1)
            new_hidden_state = self.hidden_state
        else:
            lstm_out, new_hidden_state = self.lstm(x_norm, hidden_state)
        # 3. Procesamiento
        h = self.layernorm(lstm_out)
        h = self.activation(self.dense(h))
        
        # 4. Salida Directa (Sin residuales, sin gates)
        # Tanh fuerza la salida al rango [-1, 1] que es lo que queremos
        actions = torch.tanh(self.output_layer(h))
        return actions, new_hidden_state
    
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
class CriticNetwork(nn.Module):
    """
    Crítico recurrente para RTD3.
    """
    def __init__(self,
                 scaler_params: dict,
                 units: int = 128):
        super().__init__()
        # --- Normalizador (Sin cambios) ---
        scale = torch.tensor(scaler_params["scale"], dtype=torch.float32)
        min_ = torch.tensor(scaler_params["min"], dtype=torch.float32)
        self.register_buffer("_min", min_)
        self.register_buffer("_scale", scale)
        # --- RNN: nn.LSTM ---
        # CAMBIO: La LSTM espera (obs=6 + act=3) = 9 entradas
        self.lstm = nn.LSTM(input_size=6 + 3, hidden_size=units, batch_first=True)
        self.layernorm = nn.LayerNorm(units)
        # --- Cabeza de Evaluación Q ---
        # CAMBIO 2: La red Q ahora solo recibe la salida de la LSTM (units)
        #           Ya no concatenamos la acción al final.
        hidden_dim = units // 2
        self.q_network = nn.Sequential(
            nn.Linear(units, hidden_dim),  # <-- ANTES: nn.Linear(units + 3, ...)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # --- Estado interno (Sin cambios) ---
        self.hidden_state = None
    def _to_norm(self, x: torch.Tensor) -> torch.Tensor:
        # Solo normalizar las primeras 5 dimensiones (el error ya está normalizado)
        x_first5 = x[..., :5] * self._scale + self._min
        x_error = x[..., 5:6]  # Error ya normalizado, pasar directamente
        return torch.cat([x_first5, x_error], dim=-1)
    @torch.no_grad()
    def reset_states(self):
        self.hidden_state = None
    def forward(self,
                obs: torch.Tensor,
                act: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # Normaliza las observaciones (primeras 5 dims + error ya normalizado)
        obs_norm = self._to_norm(obs)
        # --- DEBUG: VERIFICAR NORMALIZACIÓN DE ACCIONES ---
        # Solo imprimir 1% de las veces durante entrenamiento para no saturar logs
        if self.training and torch.rand(1).item() < 0.00075:
            print("\n=== DEBUG: Verificación de Normalización en Critic ===")
            print(f"  Dimensiones del scaler:")
            print(f"    _scale tiene {len(self._scale)} valores")
            print(f"    _min tiene {len(self._min)} valores")
            print(f"  Scaler params para OBSERVACIONES (dim 0-4):")
            print(f"    scale: {self._scale[:5]}")
            print(f"    min:   {self._min[:5]}")
            
#             # Verificar si hay parámetros para acciones
#             if len(self._scale) >= 8:
#                 print(f"  Scaler params para ACCIONES (dim 5-7):")
#                 print(f"    scale: {self._scale[5:8]}")
#                 print(f"    min:   {self._min[5:8]}")
#             else:
#                 print(f"  ⚠️  El scaler NO tiene parámetros para acciones (solo {len(self._scale)} dims)")
#                 print(f"     Las acciones NO se normalizan (ya están en rango correcto)")
            
#             print(f"\n  Observaciones RAW:")
            print(f"    min={obs.min():.3f}, max={obs.max():.3f}, mean={obs.mean():.3f}")
            print(f"  Observaciones NORMALIZADAS:")
            print(f"    min={obs_norm.min():.3f}, max={obs_norm.max():.3f}, mean={obs_norm.mean():.3f}")
            print(f"\n  Acciones (entrada al Critic):")
            print(f"    min={act.min():.3f}, max={act.max():.3f}, mean={act.mean():.3f}")
            print("=" * 55 + "\n")
        # --- FIN DEBUG ---
        # CAMBIO 3: Concatenar obs y act *ANTES* de la LSTM
        # (..., 5) + (..., 3) -> (..., 8)
        lstm_input = torch.cat([obs_norm, act], dim=-1)
        is_inference_step = obs.dim() == 2
        if is_inference_step:
            # --- MODO INFERENCIA ---
            # (N, 8) -> (N, 1, 8)
            lstm_input_step = lstm_input.unsqueeze(1)
            
            # CAMBIO 4: Pasar la entrada combinada (de 8) a la LSTM
            h_out, self.hidden_state = self.lstm(lstm_input_step, self.hidden_state)
            
            h_out = h_out.squeeze(1) # (N, units)
            new_hidden_state = self.hidden_state
        else:
            # --- MODO ENTRENAMIENTO (SECUENCIA) ---
            # (N, S, 8)
            
            # CAMBIO 5: Pasar la entrada combinada (de 8) a la LSTM
            h_out, new_hidden_state = self.lstm(lstm_input, hidden_state)
            # h_out: (N, S, units)
        # Normaliza la salida de la LSTM (sin cambios)
        h_norm = self.layernorm(h_out) 
        # CAMBIO 6: Eliminar la concatenación post-LSTM
        # xu = torch.cat([h_norm, act], dim=-1) # <-- LÍNEA BORRADA
        # --- Cabeza Q ---
        # CAMBIO 7: La red Q ahora solo procesa la salida de la LSTM
        q_value = self.q_network(h_norm) # <-- ANTES: self.q_network(xu)
        return q_value, new_hidden_state