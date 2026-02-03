import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import keras
import joblib
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from graphviz import Digraph
from datetime import datetime
import shutil
keras.config.enable_unsafe_deserialization()

# Hilfsfunktion zum Laden der Skalierer
def load_scaler(directory, scaler_name):
    scaler_lib_path = os.path.join(directory, f"{scaler_name}.lib")
    scaler_p_path = os.path.join(directory, f"{scaler_name}.p")

    if os.path.exists(scaler_lib_path):
        scaler = joblib.load(scaler_lib_path)
    elif os.path.exists(scaler_p_path):
        with open(scaler_p_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"Scaler file not found: {scaler_lib_path} or {scaler_p_path}")

    return scaler

def set_states (model_main, states):
    for layer in model_main.layers:
        if hasattr(layer, 'reset_states') and layer.stateful:
            name = layer.name
            layer.states[0].assign(states[name.replace('m_', 'out_h_')])
            layer.states[1].assign(states[name.replace('m_', 'out_c_')])

# Hilfsfunktion zum Laden eines Netzwerks
def load_network(directory, input_scaler_name='input_scaler', output_scaler_name='output_scaler'):
    model_inf_keras = os.path.join(directory, 'model_inf.keras')
    model_init_keras = os.path.join(directory, 'model_init.keras')

    model_main = keras.models.load_model(model_inf_keras, compile=False)
    model_init = keras.models.load_model(model_init_keras, compile=False)

    print(f"Input_Shape Main:", model_main.input_shape)
    for i, layer in enumerate(model_main.layers):
        print(f"{i:02d}: {layer.name} ({layer.__class__.__name__})")

    print(f"Input_Shape Init:", model_init.input_shape)
    for i, layer in enumerate(model_init.layers):
        print(f"{i:02d}: {layer.name} ({layer.__class__.__name__})")

    input_scaler = load_scaler(directory, input_scaler_name)
    output_scaler = load_scaler(directory, output_scaler_name)

    @tf.function(jit_compile=False)
    def predict_main(input_tensor):
        return model_main(input_tensor, training=False)

    @tf.function(jit_compile=False)
    def predict_init(input_tensor):
        return model_init(input_tensor, training=False)

    return model_main, model_init, input_scaler, output_scaler, predict_main, predict_init

def predict_with_lstm_network(model_main, model_init, input_scaler, output_scaler, data, aux_initial_values,
                              manual_initialization=None, reset_state=True, timestep_global=None,
                              predict_main=None, predict_init=None):

    input_data_scaled = input_scaler.transform(data)
    input_data_scaled = input_data_scaled.reshape((1, input_data_scaled.shape[0], input_data_scaled.shape[1]))

    if reset_state:
        for layer in model_main.layers:
            if hasattr(layer, "reset_states") and layer.stateful:
                layer.reset_states()

    if aux_initial_values is not None and timestep_global == 0:
        aux_input_data = output_scaler.transform(aux_initial_values).reshape((1, 1, len(aux_initial_values[0])))
        state_tensors = predict_init(aux_input_data)
        states = dict(zip(model_init.output_names, state_tensors))
        set_states(model_main, states)

    if manual_initialization is not None:
        for index, value in manual_initialization.items():
            input_data_scaled[0, 0, index] = value

    num_timesteps = input_data_scaled.shape[1]
    all_predictions = []

    for t in range(num_timesteps):
        current_input = input_data_scaled[:, t:t + 1, :]

        predictions = predict_main([current_input])
        predictions = predictions.numpy()

        predictions_rescaled = output_scaler.inverse_transform(predictions[0])
        all_predictions.append(predictions_rescaled)

    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions

# Funktion zum Parsen der Konfigurationsdatei
def parse_config(config_path):
    config = {}
    global_settings = {}
    current_network = None

    def to_bool(val: str) -> bool:
        return str(val).strip().lower() in ("true", "1", "yes", "y")

    with open(config_path, 'r') as file:
        lines = file.readlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            # 1) Netzwerk-Header: nur Zeilen OHNE '='
            if '=' not in line:
                current_network = line
                config[current_network] = {}
                continue

            # Key-Value-Zeilen
            key, value = map(str.strip, line.split('=', 1))

            # 2) Globale Settings (wenn noch kein Netzwerk begonnen hat)
            if current_network is None:
                global_settings[key] = value
                continue

            # 3) Netzwerk-Details
            if key == 'directory':
                config[current_network]['directory'] = value
            elif key == 'inputs':
                config[current_network]['inputs'] = [x.strip() for x in value.split(',')]
            elif key == 'outputs':
                config[current_network]['outputs'] = [x.strip() for x in value.split(',')]
            elif key == 'stateful':
                config[current_network]['stateful'] = to_bool(value)
            elif key == 'initial_outputs':
                config[current_network]['initial_outputs'] = [float(x.strip()) for x in value.split(',')]
            elif key == 'manual_initialization':
                manual_init = {}
                for pair in value.split(','):
                    idx, val = map(str.strip, pair.split(':'))
                    manual_init[int(idx)] = float(val)
                config[current_network]['manual_initialization'] = manual_init
            elif key == 'skip_if_zero':
                raw = value.strip()
                low = raw.lower()
                if low in ("true", "1", "yes", "y"):
                    config[current_network]['skip_if_zero'] = True
                elif low in ("false", "0", "no", "n"):
                    config[current_network]['skip_if_zero'] = False
                else:
                    # Liste parsen: "ICE_Speed", "fuel"  oder  ["ICE_Speed","fuel"]
                    raw = raw.strip().strip('[]')
                    items = [s.strip().strip('"').strip("'") for s in raw.split(',') if s.strip()]
                    config[current_network]['skip_if_zero'] = items  # Liste von Strings
            else:
                # Unbekannte Keys stillschweigend ignorieren oder per print warnen
                # print(f"Ignoring unknown key '{key}' in network '{current_network}'")
                pass

    return config, global_settings


# Hilfsfunktion zum Laden der Eingaben und Ausgaben aus CSV
def load_inputs_outputs_from_csv(csv_path, config, delimiter=[';', ',']):
    last_exception = None
    for delim in delimiter:
        try:
            df = pd.read_csv(csv_path, delimiter=delim, encoding='latin1')

            if df.shape[1] <= 1 or not any("time" in col.lower() for col in df.columns):
                raise ValueError(
                    f"Nur {df.shape[1]} Spalte erkannt oder keine Zeitspalte – evtl. falscher Delimiter '{delim}'")

            break

        except Exception as e:
            print(f"Fehler beim Laden mit Delimiter '{delim}': {e}")
            last_exception = e
            df = None

    if df is None:
        raise ValueError(f"CSV konnte mit keinem der Delimiter {delimiter} geladen werden.") from last_exception

    clean_columns = {col.split('(')[0].strip(): col for col in df.columns}
    inputs = {}
    outputs = {}

    for col in clean_columns.values():
        clean_col = col.split(' ')[0].strip()
        if any(clean_col in details['inputs'] for details in config.values()):
            inputs[clean_col] = df[[col]].values
        for details in config.values():
            if clean_col in details['outputs']:
                if clean_col not in outputs:
                    outputs[clean_col] = df[[col]].values


    # Zeitspalte finden
    time_col = None
    for timecol in df.columns:
        if "time" in timecol.lower():
            time_col = timecol
            break

    if time_col is None:
        raise ValueError("Keine Zeitspalte gefunden")

    time_y = pd.to_numeric(df[time_col], errors='coerce').values.reshape(-1, 1)
    if np.isnan(time_y).any():
        raise ValueError("Zeitspalte enthält ungültige Werte")

    frequency = 1 / (time_y[2] - time_y[1])
    max_time = time_y[-1, 0]

    return df, inputs, outputs, frequency, max_time, time_y


# Hauptfunktion zur Ausführung der Netzwerke
def run_chained_networks(config, initial_inputs, true_outputs):

    base_path = os.path.abspath(os.path.dirname(__file__))
    networks = {
        name: load_network(os.path.join(base_path, details['directory']))
        for name, details in config.items()
    }
    network_outputs = {name: {} for name in networks.keys()}

    num_time_steps = list(initial_inputs.values())[0].shape[0]
    total_time_start = time.time()
    time_one_step = {}
    all_initial_outputs_logged = {}

    for t in range(num_time_steps):
        current_inputs = {key: value[t:t + 1] for key, value in initial_inputs.items()}

        for name, details in config.items():
            output_name_init = details['outputs']
            if t == 0:
                time_one_step[name] = []

            input_names = details['inputs']
            formatted_inputs = []

            for input_name in input_names:
                if ':' in input_name:
                    continue  # verschaltete Inputs aus anderen Netzwerken ignorieren
                value = current_inputs.get(input_name)
                if value is not None:
                    formatted_inputs.append(f"{input_name}={value.flatten()[0]:.6f}")

            print(f"Timestep {t} | Network: {name} | CSV inputs: " + ", ".join(formatted_inputs))
            inputs = []
            is_stateful = bool(details.get('stateful', False))

            for input_index, input_source in enumerate(details['inputs']):
                if ':' in input_source:
                    network_name, output_name = [x.strip() for x in input_source.split(':')]

                    if output_name in network_outputs[network_name] and \
                            len(network_outputs[network_name][output_name]) > t:
                        # Output[t] ist schon berechnet → direkt verwenden
                        print(f"[t={t}] Using current output from {network_name}:{output_name}")
                        inputs.append(network_outputs[network_name][output_name][t:t + 1])
                    elif t == 0 and 'manual_initialization' in details and input_index in details[
                        'manual_initialization']:
                        # Initialwert bei t=0
                        manual_value = details['manual_initialization'][input_index]
                        print(f"[t={t}] Using manual initialization for {network_name}:{output_name} → {manual_value}")
                        inputs.append(np.array([[manual_value]]))
                    elif output_name in network_outputs[network_name]:
                        # Fallback: letztes verfügbares t-1
                        print(f"[t={t}] Using fallback (t-1) from {network_name}:{output_name}")
                        inputs.append(network_outputs[network_name][output_name][t - 1:t])
                    else:
                        raise ValueError(f"Missing input '{output_name}' from network '{network_name}' at timestep {t}")
                else:
                    inputs.append(current_inputs[input_source])

            current_input = np.concatenate(inputs, axis=1)
            model_main, model_init, input_scaler, output_scaler, predict_main, predict_init = networks[name]

            # Skip nur, wenn in config angegeben
            skip_cfg = details.get('skip_if_zero', False)

            # Optional: Zeitmessungsliste initialisieren
            if t == 0 and name not in time_one_step:
                time_one_step[name] = []

            # 1) Konfigurierbares Skippen bei all-zeros
            did_skip = False
            if isinstance(skip_cfg, (list, tuple)):
                # Prüfe nur die angegebenen Variablen
                idxs = [details['inputs'].index(v) for v in skip_cfg if v in details['inputs']]
                vals = [float(current_input[0, i]) for i in idxs]  # Spaltenindex!
                if idxs and all(np.isclose(v, 0.0) for v in vals):
                    print(f"[t={t}] Skipping {name} because {skip_cfg} are zero (per config)")
                    predictions = np.zeros((1, len(details['outputs'])))
                    time_one_step[name].append(0.0)
                    did_skip = True
            elif skip_cfg is True:
                # Legacy: alle Inputs == 0 → skip
                if np.allclose(current_input, 0.0):
                    print(f"[t={t}] Skipping {name} because ALL inputs are zero (per config)")
                    predictions = np.zeros((1, len(details['outputs'])))
                    time_one_step[name].append(0.0)
                    did_skip = True

            if not did_skip:
                time_stateful_start = time.time()
                if is_stateful:
                    aux_initial_values = None
                    if t == 0:
                        aux_initial_values, outputs_for_logging = aux_initial_true_values(details, true_outputs, output_name_init)
                        all_initial_outputs_logged.update(outputs_for_logging)

                    predictions = predict_with_lstm_network(model_main, model_init, input_scaler, output_scaler, current_input, aux_initial_values, reset_state=(t==0), timestep_global=t, predict_main=predict_main, predict_init=predict_init)

                else:
                    print(f"Non-stateful networks not yet implemented.")
                    predictions = np.zeros((1, len(details['outputs'])))

                time_stateful_end = time.time()
                time_one_step[name].append(time_stateful_end - time_stateful_start)

                # 2) Robustheit: Wenn das Modell-Call fehlgeschlagen ist
                if predictions is None:
                    print(f"[t={t}] Prediction failed in {name}; filling zeros for this step.")
                    predictions = np.zeros((1, len(details['outputs'])))

            for idx, output_name in enumerate(details['outputs']):
                if output_name not in network_outputs[name]:
                    network_outputs[name][output_name] = predictions[:, idx:idx + 1]
                else:
                    network_outputs[name][output_name] = np.append(
                        network_outputs[name][output_name], predictions[:, idx:idx + 1], axis=0
                    )

    avg_prediction_time = {name: np.mean(times[1:]) for name, times in time_one_step.items()}

    total_time_end = time.time()
    total_time = total_time_end - total_time_start

    return network_outputs, avg_prediction_time, total_time, all_initial_outputs_logged

def create_network_graph(config, output_dir='Output'):
    dot = Digraph(comment='Network Graph')

    # Hinzufügen der Knoten
    for name in config.keys():
        dot.node(name, name)

    # Hinzufügen der Kanten basierend auf den Inputs
    for name, details in config.items():
        for input_source in details['inputs']:
            if ': ' in input_source:
                network_name, output_name = input_source.split(': ')
                dot.edge(network_name, name, label=output_name)
            else:
                dot.edge(input_source, name)

    # Speichern der Grafik
    graph_path = os.path.join(output_dir, 'network_graph')
    dot.render(graph_path, format='png')
    print(f"Network graph saved to {graph_path}.png")

def calculate_mae(predictions, true_values):
    # Nur endliche Werte verwenden
    mask = np.isfinite(predictions) & np.isfinite(true_values)
    return np.mean(np.abs(predictions[mask] - true_values[mask]))

def calculate_mape(predictions, true_values):
    # Nur endliche Werte verwenden und den Fall vermeiden, dass true_values Null ist
    mask = np.isfinite(predictions) & np.isfinite(true_values) & (true_values != 0)
    return np.mean(np.abs((true_values[mask] - predictions[mask]) / true_values[mask])) * 100

def calculate_max_ae(predictions, true_values):
    # Nur endliche Werte verwenden
    mask = np.isfinite(predictions) & np.isfinite(true_values)
    return np.max(np.abs(predictions[mask] - true_values[mask]))

def calculate_max_re(predictions, true_values):
    # Nur endliche Werte verwenden und den Fall vermeiden, dass true_values Null ist
    mask = np.isfinite(predictions) & np.isfinite(true_values) & (true_values != 0)
    return np.max(np.abs((true_values[mask] - predictions[mask]) / true_values[mask])) * 100

def cumulative_absolute_error(predictions, true_values, frequency):

    dt = 1/frequency

    cum_true = np.cumsum(0.5 * (true_values[:-1] + true_values[1:]) * dt)
    cum_pred = np.cumsum(0.5 * (predictions[:-1] + predictions[1:]) * dt)

    cum_true = np.insert(cum_true, 0, 0)
    cum_pred = np.insert(cum_pred, 0, 0)

    cum_error = np.abs(cum_true[-1] - cum_pred[-1])
    cum_rel_error = (cum_error/cum_true[-1]) * 100

    return cum_true, cum_pred, cum_error, cum_rel_error


def add_predictions_to_dataset(df, config, final_outputs, maes, mapes, max_aes, max_res, cum_true, cum_pred):
    new_cols = {}
    for network_name, details in config.items():
        for output_name in details['outputs']:
            base = f"{output_name}"
            if network_name in final_outputs and output_name in final_outputs[network_name] \
               and final_outputs[network_name][output_name].size > 0:
                new_cols[f"{base}_pred"] = final_outputs[network_name][output_name].ravel()
            # Fehlerkennzahlen (Skalare) sauber broadcasten:
            if network_name in maes and output_name in maes[network_name]:
                new_cols[f"{base}_mae"]      = maes[network_name][output_name]
                new_cols[f"{base}_mape"]     = mapes[network_name][output_name]
                new_cols[f"{base}_max_ae"]   = max_aes[network_name][output_name]
                new_cols[f"{base}_max_re"]   = max_res[network_name][output_name]
                new_cols[f"{base}_cum_true"] = cum_true[network_name][output_name].ravel()
            if network_name in cum_pred and output_name in cum_pred[network_name]:
                new_cols[f"{base}_cum_pred"] = cum_pred[network_name][output_name].ravel()
    if new_cols:
        df = df.join(pd.DataFrame(new_cols, index=df.index)).copy()
    return df


def aux_initial_true_values(details, true_outputs, output_names):
    aux_initial_values = []
    outputs_for_logging = {}

    if 'initial_outputs' in details and 'outputs' in details:
        aux_initial_values = [details['initial_outputs']]
        outputs_for_logging = dict(zip(details['outputs'], details['initial_outputs']))
    else:
        # echte True-Initialwerte ausgeben, falls vorhanden
        row = []
        for name in output_names:
            if name in true_outputs:
                val = float(true_outputs[name][0, 0])
            else:
                val = 0.0
            row.append(val)
            outputs_for_logging[name] = val
        aux_initial_values = [row]

    return aux_initial_values, outputs_for_logging


def main(config, input_base, output_dir='Output'):

    csv_path = os.path.join('input_data', input_base)
    df, initial_inputs, true_outputs, frequency, max_time, time_y = load_inputs_outputs_from_csv(csv_path, config)
    final_outputs, avg_prediction_times, total_time, outputs_for_logging = run_chained_networks(config, initial_inputs, true_outputs)

    # Neuen Unterordner mit Zeitstempel für die Ergebnisse erstellen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f'{timestamp}_results_{os.path.splitext(input_base)[0]}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Konfigurationsdatei in den Ergebnisordner kopieren
    shutil.copy(config_path, os.path.join(results_dir, os.path.basename(config_path)))

    maes = {}
    mapes = {}
    max_aes = {}
    max_res = {}
    cum_true = {}
    cum_pred = {}
    cum_error = {}
    cum_rel_error = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, details in config.items():

        input_cols = details['inputs']
        output_cols = details['outputs']

        n_in = len(input_cols)
        n_out = len(output_cols)
        n_tot = n_in + n_out

        fig, axes = plt.subplots(n_tot, 1, sharex=True, figsize=(12, 4 * n_tot))
        for ax in axes:
            ax.grid(True, alpha=0.7)
            ax.margins (x=0)
        fig.suptitle(f"Overview for {name}", fontsize=16)

        for i, input_col in enumerate(input_cols):
            if ':' in input_col:
                net, in_name = input_col.split(":", 1)
                net = net.strip()
                in_name = in_name.strip()
                if in_name in final_outputs[net]:
                    input_values = final_outputs[net][in_name]
                    axes[i].plot(time_y, input_values)
                    axes[i].set_title(f"Input: {input_col}")
                    axes[i].set_ylabel(input_col)
                else:
                    print(f'{in_name} not contained.')
            else:
                input_values = initial_inputs[input_col]
                axes[i].plot(time_y, input_values)
                axes[i].set_title(f"Input: {input_col}")
                axes[i].set_ylabel(input_col)

        for j, output_col in enumerate(output_cols):
            if output_col not in true_outputs:
                predictions = final_outputs[name][output_col]

                idx = n_in + j  # Position im Gesamt-Plot
                axes[idx].plot(time_y, predictions, label="Predicted")
                axes[idx].set_title(f"Output: {output_col} (Pred)")
                axes[idx].set_ylabel(output_col)
                axes[idx].set_xlabel('Time [s]')
                axes[idx].legend()
            else:
                predictions = final_outputs[name][output_col]
                true_values = true_outputs[output_col]

                idx = n_in + j  # Position im Gesamt-Plot
                axes[idx].plot(time_y, true_values, label="True")
                axes[idx].plot(time_y, predictions, label="Predicted")
                axes[idx].set_title(f"Output: {output_col} (True vs Pred)")
                axes[idx].set_ylabel(output_col)
                axes[idx].set_xlabel('Time [s]')
                axes[idx].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plot_path = os.path.join(results_dir, f'{name}_all.png')
        plt.savefig(plot_path)
        plt.close()

        maes[name] = {}
        mapes[name] = {}
        max_aes[name] = {}
        max_res[name] = {}
        cum_true[name] = {}
        cum_pred[name] = {}
        cum_error[name] = {}
        cum_rel_error[name] = {}
        for output_column in details['outputs']:
            if output_column in true_outputs:
                predictions = final_outputs[name][output_column]
                true_values = true_outputs[output_column]

                mae = calculate_mae(predictions, true_values)
                mape = calculate_mape(predictions, true_values)
                max_ae = calculate_max_ae(predictions, true_values)
                max_re = calculate_max_re(predictions, true_values)
                cum_tru, cum_pre, cum_err, cum_rel_err = cumulative_absolute_error(predictions, true_values, frequency)

                '''
                print(f'MAE of {name} for {output_column}: {mae}')
                print(f'MAPE of {name} for {output_column}: {mape}')
                print(f'Max AE of {name} for {output_column}: {max_ae}')
                print(f'Max RE of {name} for {output_column}: {max_re}')
                print(f'Cumulative Error of {name} for {output_column}: {cum_err}')
                print(f'Cumulative Relative Error of {name} for {output_column}: {cum_rel_err}')
                '''

                maes[name][output_column] = mae
                mapes[name][output_column] = mape
                max_aes[name][output_column] = max_ae
                max_res[name][output_column] = max_re
                cum_true[name][output_column] = cum_tru
                cum_pred[name][output_column] = cum_pre
                cum_error[name][output_column] = cum_err
                cum_rel_error[name][output_column] = cum_rel_err

                '''
                # Plot-Erstellung und Speicherung
                # plt.figure(figsize=((len(initial_inputs)*2), 10))
                plt.figure(figsize=(7, 5))
                plt.plot(time_y, true_values, label='True Values')
                plt.plot(time_y, predictions, label='Predicted Values')
                plt.title(f'{name} - {output_column}')
                plt.xlabel('Time [s]')
                plt.ylabel(output_column)
                plt.legend()
                plt.grid()
                plot_path = os.path.join(results_dir, f'{name}_{output_column}.png')
                plt.savefig(plot_path)
                plt.close()
                '''

                # Daten in NumPy-Arrays umwandeln und ungültige Werte filtern
                true_values = np.array(true_values, dtype=np.float64)
                predictions = np.array(predictions, dtype=np.float64)

                # NaN- oder Inf-Werte entfernen
                valid_indices = ~np.isnan(true_values) & ~np.isinf(true_values) & ~np.isnan(predictions) & ~np.isinf(
                    predictions)
                true_values = true_values[valid_indices]
                predictions = predictions[valid_indices]
                x_values = np.arange(len(true_values))

                # Plot erstellen
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=x_values/frequency, y=true_values, mode='lines', name='True Values', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=x_values/frequency, y=predictions, mode='lines', name='Predicted Values',
                                         line=dict(color='red')))

                # Layout mit automatischer Skalierung
                fig.update_layout(
                    title=f'{name} - {output_column}',
                    xaxis_title='Time [s]',
                    yaxis_title=output_column,
                    hovermode="closest",
                    xaxis=dict(autorange=True, showgrid=True, zeroline=True),
                    yaxis=dict(autorange=True, showgrid=True, zeroline=True)
                )

                # Interaktiven Plot speichern
                fig.write_html(os.path.join(results_dir, f'{name}_{output_column}_interactive.html'))

                plt.figure(figsize=(6, 4))
                plt.scatter(true_values, predictions, label=name, s=100)
                plt.xlabel(f'{output_column}_true', fontsize=14)
                plt.ylabel(f'{output_column}_pred', fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'k-', linewidth=2,
                         label='y=x')
                # plt.legend()
                plt.grid(True)
                plot_path = os.path.join(results_dir, f'{name}_{output_column}_scatter.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
            else:
                predictions = final_outputs[name][output_column]
                '''
                # Plot-Erstellung und Speicherung
                # plt.figure(figsize=((len(initial_inputs)*2), 10))
                plt.figure(figsize=(7, 5))
                plt.plot(time_y, predictions, label='Predicted Values')
                plt.title(f'{name} - {output_column}')
                plt.xlabel('Time [s]')
                plt.ylabel(output_column)
                plt.legend()
                plt.grid()
                plot_path = os.path.join(results_dir, f'{name}_{output_column}.png')
                plt.savefig(plot_path)
                plt.close()
                '''

                # Daten in NumPy-Arrays umwandeln und ungültige Werte filtern
                predictions = np.array(predictions, dtype=np.float64)

                # NaN- oder Inf-Werte entfernen
                valid_indices = ~np.isnan(predictions) & ~np.isinf(predictions)
                predictions = predictions[valid_indices]
                x_values = np.arange(len(predictions))

                # Plot erstellen
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_values/frequency, y=predictions, mode='lines', name='Predicted Values',
                                         line=dict(color='red')))

                # Layout mit automatischer Skalierung
                fig.update_layout(
                    title=f'{name} - {output_column}',
                    xaxis_title='Time [s]',
                    yaxis_title=output_column,
                    hovermode="closest",
                    xaxis=dict(autorange=True, showgrid=True, zeroline=True),
                    yaxis=dict(autorange=True, showgrid=True, zeroline=True)
                )

                # Interaktiven Plot speichern
                fig.write_html(os.path.join(results_dir, f'{name}_{output_column}_interactive.html'))

    create_network_graph(config, results_dir)

    realtime = round(total_time/max_time, 5)

    with open(f"{results_dir}/Auswertung.txt", "w") as file:
        file.write(f"Evaluation of Case {timestamp}\n\n")
        file.write(f"Inputs used for Initialization: \n\n")
        for name, value in outputs_for_logging.items():
            file.write(f"{name}: {value}\n")
        file.write(f"\n")
        file.write(f"Frequency of timeseries: {frequency[0]} Hz\n\n")
        for name, avg_time in avg_prediction_times.items():
            file.write(f"Average prediction time for {name}: {round(avg_time, 5)} seconds\n")
        file.write(f"\n")
        file.write(f"Total prediction time for the entire process: {round(total_time, 5)} seconds\n\n")
        file.write(f"Realtime Factor: {realtime}\n\n")
        file.write(f"Metrics:\n\n")
        for outer_key, inner_dict in cum_error.items():
            file.write(f"CumError of Net {outer_key} for {max_time} s:\n")
            for inner_key, value in inner_dict.items():
                file.write(f"{inner_key}: {value}\n")
            file.write(f"\n")
        for outer_key, inner_dict in cum_rel_error.items():
            file.write(f"CumRelError of Net {outer_key} for {max_time} s:\n")
            for inner_key, value in inner_dict.items():
                file.write(f"{inner_key}: {value}\n")
            file.write(f"\n")
        for outer_key, inner_dict in maes.items():
            file.write(f"MAE of Net {outer_key}:\n")
            for inner_key, value in inner_dict.items():
                file.write(f"{inner_key}: {value}\n")
            file.write(f"\n")
        for outer_key, inner_dict in mapes.items():
            file.write(f"MAPE of Net {outer_key}:\n")
            for inner_key, value in inner_dict.items():
                file.write(f"{inner_key}: {value} %\n")
            file.write(f"\n")
        for outer_key, inner_dict in max_aes.items():
            file.write(f"Max. AE of Net {outer_key}:\n")
            for inner_key, value in inner_dict.items():
                file.write(f"{inner_key}: {value}\n")
            file.write(f"\n")
        for outer_key, inner_dict in max_res.items():
            file.write(f"Max. RE of Net {outer_key}:\n")
            for inner_key, value in inner_dict.items():
                file.write(f"{inner_key}: {value} %\n")
            file.write(f"\n")

    df_with_predictions = add_predictions_to_dataset(df, config, final_outputs, maes, mapes, max_aes, max_res, cum_true, cum_pred)
    output_csv_path = os.path.join(results_dir, 'data_with_predictions.csv')
    df_with_predictions.to_csv(output_csv_path, index=False, sep=';')
    print(f"Dataset with predictions, MAE, MAPE, Max AE, and Max RE saved to {output_csv_path}")

    return final_outputs, df_with_predictions, cum_error, cum_rel_error, maes, mapes

def all_eval(all_evu, datei, cum_error, cum_rel_error, maes, mapes):

    error = {
        "cum_abs_error": cum_error,
        "cum_rel_error": cum_rel_error,
        "maes": maes,
        "mapes": mapes
    }

    for net_name in cum_error.keys():
        for sort, err in error.items():
            for out_para, wert in err[net_name].items():
                all_evu.setdefault(datei, {}).setdefault(net_name, {}).setdefault(sort, {})[out_para] = wert

    return all_evu

def all_eval_in_csv(all_evu, output_dir='Output'):

    rows = []

    for datei, netze in all_evu.items():
        for net_name, error in netze.items():
            for sort, out in error.items():
                for out_name, wert in out.items():
                    rows.append({
                        "Datei": datei,
                        "NN": net_name,
                        "Metrics": sort,
                        "Parameter": out_name,
                        "Wert": wert
                    })

    df = pd.DataFrame(rows)

    mean_matrix = df.groupby(["NN", "Metrics", "Parameter"])["Wert"].mean().reset_index()


    out_safe = os.path.join(output_dir, f'all_files_eval.xlsx')
    with pd.ExcelWriter(out_safe) as writer:
        df.to_excel(writer, sheet_name="all_values", index=False)
        mean_matrix.to_excel(writer, sheet_name="mean_values", index=False)
    return

if __name__ == "__main__":
    config_path = 'config.txt'
    config, global_settings = parse_config(config_path)
    input_base = global_settings.get('input_data', 'input_data')

    if input_base.endswith(".csv"):
        main(config, input_base)
    elif input_base == "all":
        data_path = 'input_data'
        all_evu = {}
        for datei in os.listdir(data_path):
            if datei.endswith(".csv"):
                final_outputs, df_with_predictions, cum_error, cum_rel_error, maes, mapes = main(config, datei)
                all_evu = all_eval(all_evu, datei, cum_error, cum_rel_error, maes, mapes)
        all_eval_in_csv(all_evu)
