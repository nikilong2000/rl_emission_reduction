import os
import random
import pandas as pd
import tensorflow as tf


def sample_init_state():
    """
    Generates a random initial state tensor.

    This function returns a `tf.Tensor` of shape (5,) with component values
    sampled from predefined uniform distributions.

    The components are generated within the following ranges:
      - vel:    Constant 70.0
      - mf:     [3.0,   70.0]
      - vtg:    [0.23,  0.9]
      - brk:    [0.0,   100.0]
      - ice_sp: [0.0,   4500.0]

    Returns
    -------
    tf.Tensor
        A tensor of shape (5,) containing the randomly sampled initial state.
    """
    vel = tf.constant(70, dtype=tf.float32)
    mf = tf.random.uniform([], minval=3.0, maxval=70.0, dtype=tf.float32)
    brk = tf.random.uniform([], minval=0.0, maxval=100.0, dtype=tf.float32)
    ice_sp = tf.random.uniform([], minval=900.0, maxval=4500.0, dtype=tf.float32)

    state = tf.stack([vel, mf, brk, ice_sp], axis=0)

    return state


def sample_init_state_folder(folder_path="../src/data"):
    """
    Samples an initial state from a random row in a random CSV file.

    This function selects a random CSV file from the specified folder, reads a
    single random row from it efficiently, and maps the relevant columns to
    a TensorFlow tensor representing the initial state of the system.

    Parameters
    ----------
    folder_path : str, optional
        The path to the directory containing the CSV data files,
        by default "../src/data".

    Returns
    -------
    tf.Tensor
        A tensor of shape (5,) with the initial state values for
        [fuel, Brake, ICE_Speed_soll, EM2_Torque, ICE_Torque_pred].

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in the specified folder.
    ValueError
        If the chosen CSV file is empty or contains only a header.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    chosen_file = os.path.join(folder_path, random.choice(csv_files))

    # Count how many rows the CSV has
    with open(chosen_file, "r") as f:
        n_lines = sum(1 for _ in f) - 1  # -1 to ignore the header
    if n_lines <= 0:
        raise ValueError(f"The file {chosen_file} is empty")

    # Choose a random index (0-indexed after the header)
    random_row_idx = random.randint(0, n_lines - 1)

    # Read only that row
    df_iter = pd.read_csv(
        chosen_file, skiprows=lambda x: x != 0 and x - 1 != random_row_idx
    )
    row = df_iter.iloc[0]

    # Extraer columnas y convertir a tensores
    mf = tf.constant(row["fuel"], dtype=tf.float32)
    brk = tf.constant(row["Brake"], dtype=tf.float32)
    ice_sp = tf.constant(row["ICE_Speed_soll"], dtype=tf.float32)
    EM2 = tf.constant(row["EM2_Torque"], dtype=tf.float32)
    torque = tf.constant(row["ICE_Torque_pred"], dtype=tf.float32)

    state = tf.stack([mf, brk, ice_sp, EM2, torque], axis=0)
    return state
