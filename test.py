import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_data(n):
    df = pd.DataFrame({
        'x': range(n),
        'y': range(n),
        'color': ['red'] * n,
        'size': range(n),
        'shape': ['circle'] * n
    })
    fig = plt.figure(figsize=(8,8))
    plt.scatter(df['x'], df['y'])
    return fig

def main():
    gr.Interface(
        fn=create_data,
        inputs="number",
        outputs=['plot']
    ).launch(server_name="0.0.0.0")
    
if __name__ == '__main__':
	main()