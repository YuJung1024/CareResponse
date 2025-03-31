import os
import json
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import cv2
from collections import defaultdict
from collections import deque

# Define body part connections and colors
BODY_PART_GROUPS = {
    'head': [(0, 1)],
    'right_arm': [(1, 2), (2, 3), (3, 4)],
    'left_arm': [(1, 5), (5, 6), (6, 7)],
    'right_torso': [(1, 8), (8, 9), (9, 10)],
    'left_torso': [(1, 11), (11, 12), (12, 13)],
    'hip': [(8, 11)],
    'face_right': [(0, 14), (14, 16)],
    'face_left': [(0, 15), (15, 17)]
}

BODY_PART_COLORS = {
    'head': 'blue',
    'right_arm': 'green',
    'left_arm': 'purple',
    'right_torso': 'red',
    'left_torso': 'orange',
    'hip': 'brown',
    'face_right': 'cyan',
    'face_left': 'magenta'
}

SKELETON_GRAPH = {
    1: [2, 15, 16],  # Nose
    2: [1, 3, 6, 9, 12],  # Neck
    3: [2, 4],  # Right Shoulder
    4: [3, 5],  # Right Elbow
    5: [4],  # Right Wrist
    6: [2, 7],  # Left Shoulder
    7: [6, 8],  # Left Elbow
    8: [7],  # Left Wrist
    9: [2, 10],  # Right Hip
    10: [9, 11],  # Right Knee
    11: [10],  # Right Ankle
    12: [2, 13],  # Left Hip
    13: [12, 14],  # Left Knee
    14: [13],  # Left Ankle
    15: [1, 17],  # Right Eye
    16: [1, 18],  # Left Eye
    17: [15],  # Right Ear
    18: [16]  # Left Ear
}


def bfs_fill(skeleton_graph, points):
    """
    Fills zero values in keypoints using BFS based on skeleton graph connectivity.

    :param skeleton_graph: Dictionary representing keypoint connections.
    :param keypoints: Dictionary {keypoint_id: value}, where 0 means missing data.
    """
    keypoints = {i + 1: (points[i * 3], points[i * 3 + 1])
                 for i in range(len(points) // 3)}

    queue = deque([k for k, v in keypoints.items() if v != (0, 0)])

    while queue:
        node = queue.popleft()  # Get the current node
        value = keypoints[node]  # Get its value

        # Iterate over its neighbors
        for neighbor in skeleton_graph.get(node, []):
            # If neighbor has zero value, update it
            if keypoints[neighbor] == (0, 0):
                keypoints[neighbor] = value
                queue.append(neighbor)  # Add to queue for further processing

    return [(keypoints[i][0], keypoints[i][1]) for i in range(1, len(keypoints) + 1)]


# ====================== MODEL DEFINITION ======================

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length=60):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerGenerator(torch.nn.Module):
    def __init__(self, input_dim=36, condition_dim=4, hidden_dim=256, num_layers=4, nhead=8):
        super().__init__()

        self.keypoint_embedding = torch.nn.Linear(input_dim, hidden_dim)
        self.condition_embedding = torch.nn.Linear(
            condition_dim, hidden_dim // 2)

        self.combined_embedding = torch.nn.Linear(
            hidden_dim + hidden_dim // 2, hidden_dim)

        self.positional_encoding = PositionalEncoding(hidden_dim)

        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                          dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)

        self.output_layer = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, input_seq, condition):
        batch_size, seq_len, keypoint_dim = input_seq.shape

        x = self.keypoint_embedding(input_seq)

        condition_embedded = self.condition_embedding(condition)
        condition_expanded = condition_embedded.unsqueeze(
            1).expand(-1, seq_len, -1)

        combined = torch.cat([x, condition_expanded], dim=2)
        combined = self.combined_embedding(combined)

        encoded = self.positional_encoding(combined)
        transformed = self.transformer_encoder(encoded)
        output = self.output_layer(transformed)

        return output

# ====================== DATA LOADING FUNCTIONS ======================


def load_inference_data(input_dir):

    keypoints_data = []

    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.json'):
                full_path = os.path.join(root, file)
                match = re.search(r'_(\d+)_', file)
                if match:
                    frame_number = int(match.group(1))
                    json_files.append((full_path, frame_number))

    json_files.sort(key=lambda x: x[1])

    # Process keypoints
    all_keypoints = []
    padding_frame = []
    for file_path, _ in json_files:
        with open(file_path, "r") as f:
            try:
                json_data = json.load(f)
                if "people" in json_data and len(json_data["people"]) > 0:
                    keypoints = json_data["people"][0]["pose_keypoints"]

                    grouped_keypoints = bfs_fill(SKELETON_GRAPH, keypoints)
                    padding_frame = grouped_keypoints
                    keypoints_tensor = torch.tensor(
                        grouped_keypoints, dtype=torch.float)
                    all_keypoints.append(keypoints_tensor)
                else:
                    if padding_frame:
                        grouped_keypoints = padding_frame
                    else:
                        continue
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                all_keypoints.append(torch.zeros((18, 2), dtype=torch.float))

    # Create chunks of 30 frames with 15-frame overlap
    chunked_keypoints = []
    chunk_starts = []
    current_frame = 0

    while current_frame + 15 < len(all_keypoints):
        end_idx = min(current_frame + 30, len(all_keypoints))
        chunk = all_keypoints[current_frame:end_idx]

        padding_needed = 30 - len(chunk)
        if padding_needed > 0:
            last_frame = chunk[-1] if chunk else torch.zeros(
                (18, 2), dtype=torch.float)
            for _ in range(padding_needed):
                chunk.append(last_frame)  # Use the last frame to fill

        chunk_tensor = torch.stack(chunk)
        chunked_keypoints.append(chunk_tensor)
        chunk_starts.append(current_frame)

        current_frame += 15

    return {
        'original_sequence': all_keypoints,
        'chunks': chunked_keypoints,
        'chunk_starts': chunk_starts
    }


def get_bounding_box_condition(seq):
    x_coords = seq[:, :, 0].flatten()
    y_coords = seq[:, :, 1].flatten()

    valid_x = x_coords[x_coords > 0]
    valid_y = y_coords[y_coords > 0]

    center_x, center_y, width, height = 0.0, 0.0, 1.0, 1.0

    if len(valid_x) > 0 and len(valid_y) > 0:
        min_x = valid_x.min().item()
        max_x = valid_x.max().item()
        min_y = valid_y.min().item()
        max_y = valid_y.max().item()

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max(abs(max_x - min_x), 1.0)  # Avoid zero width
        height = max(abs(max_y - min_y), 1.0)  # Avoid zero height

    # Condition vector: [center_x, center_y, width, height]
    return torch.tensor([center_x, center_y, width, height], dtype=torch.float)

# ====================== INFERENCE AND VISUALIZATION FUNCTIONS ======================


def run_inference(model, input_data, device='cpu'):
    model.eval()

    chunks = input_data['chunks']
    chunk_starts = input_data['chunk_starts']

    output_frames = [None] * len(input_data['original_sequence'])

    with torch.no_grad():
        for idx, chunk in enumerate(chunks):
            flattened_chunk = chunk.view(
                1, chunk.shape[0], -1)  # Shape: (1, 30, 36)
            condition = get_bounding_box_condition(
                chunk).unsqueeze(0).to(device)

            generated_output = model(flattened_chunk.to(device), condition)

            # Reshape output to (30, 18, 2)
            output_poses = generated_output.view(
                generated_output.shape[1], 18, 2)
            output_poses = output_poses.cpu().numpy()

            chunk_start = chunk_starts[idx]
            for i in range(min(len(output_poses), 30)):
                frame_idx = chunk_start + i
                if frame_idx < len(output_frames):
                    output_frames[frame_idx] = output_poses[i]

    for i in range(len(output_frames)):
        if output_frames[i] is None:
            output_frames[i] = np.zeros((18, 2))

    return output_frames


def draw_skeleton(keypoints, ax, frame_idx=None, title=None):
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.invert_yaxis()

    if title:
        ax.set_title(title)
    elif frame_idx is not None:
        ax.set_title(f"Frame {frame_idx}")

    valid_keypoints = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)

    for part_name, connections in BODY_PART_GROUPS.items():
        color = BODY_PART_COLORS[part_name]
        for connection in connections:
            i, j = connection
            if i < len(keypoints) and j < len(keypoints):
                # Only draw connection if both keypoints are valid
                if valid_keypoints[i] and valid_keypoints[j]:
                    ax.plot([keypoints[i, 0], keypoints[j, 0]],
                            [keypoints[i, 1], keypoints[j, 1]],
                            color=color, linewidth=2)

    ax.scatter(keypoints[valid_keypoints, 0],
               keypoints[valid_keypoints, 1], color='black', s=20)


def create_visualization(input_frames, output_frames, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    frames = []

    def init():
        ax1.clear()
        ax2.clear()
        ax1.set_xlim(0, 1000)
        ax1.set_ylim(0, 1000)
        ax1.invert_yaxis()
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0, 1000)
        ax2.invert_yaxis()
        return []

    def update(frame_idx):
        ax1.clear()
        ax2.clear()

        if frame_idx < len(input_frames):
            draw_skeleton(input_frames[frame_idx].numpy(
            ), ax1, title=f"Input - Frame {frame_idx}")

        if frame_idx < len(output_frames):
            draw_skeleton(output_frames[frame_idx], ax2,
                          title=f"Generated - Frame {frame_idx}")

        return []

    num_frames = min(len(input_frames), len(output_frames))
    ani = FuncAnimation(fig, update, frames=range(
        num_frames), init_func=init, blit=True)

    writer = animation.FFMpegWriter(
        fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_path, writer=writer)
    plt.close()

    print(f"Visualization saved to {output_path}")


def create_concatenated_video(input_frames, output_frames, output_path, fps=15, delay_frames=15):
    width, height = 1000, 1000
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

    num_frames = min(len(input_frames), len(output_frames) + delay_frames)
    for frame_idx in tqdm(range(num_frames), desc="Creating video"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        draw_skeleton(input_frames[frame_idx].numpy(),
                      ax1, title=f"Input - Frame {frame_idx}")

        if frame_idx >= delay_frames:
            output_idx = frame_idx - delay_frames
            if output_idx < len(output_frames):
                draw_skeleton(
                    output_frames[output_idx], ax2, title=f"Generated - Frame {output_idx}")
        else:
            ax2.clear()
            ax2.set_xlim(0, 1000)
            ax2.set_ylim(0, 1000)
            ax2.invert_yaxis()
            ax2.set_title("Generated (waiting for input frames)")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (width*2, height))

        video.write(img)
        plt.close(fig)

    video.release()
    print(f"Video saved to {output_path}")


def save_keypoints_to_csv(keypoints, output_path):
    with open(output_path, 'w') as f:
        f.write("frame,keypoint,x,y\n")
        for frame_idx, frame in enumerate(keypoints):
            for kp_idx, (x, y) in enumerate(frame):
                f.write(f"{frame_idx},{kp_idx},{x},{y}\n")
    print(f"Keypoints saved to {output_path}")

# ====================== MAIN FUNCTION ======================


def main():
    model_path = "generator_final.pt"  # Path to your trained model
    # Directory with input JSON files
    input_dir = "./valdata/Coffee_room_01_video (43)/keypoints"
    output_video_path = "./inference_pre_test/output_visualizationh2v57_n0_pre.mp4"
    output_csv_input = "./inference_pre_test/input_keypointsh2v57_n0_pre.csv"
    output_csv_output = "./inference_pre_test/output_keypointsh2v57_n0_pre.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    input_dim = 36  # 18 keypoints, 2 coordinates each
    condition_dim = 4  # center_x, center_y, width, height

    model = TransformerGenerator(
        input_dim=input_dim, condition_dim=condition_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print("Loading data...")
    input_data = load_inference_data(input_dir)

    print("Running inference...")
    output_frames = run_inference(model, input_data, device)

    print("Creating visualization...")
    create_concatenated_video(
        input_data['original_sequence'], output_frames, output_video_path, delay_frames=15)

    print("Saving keypoints data...")
    save_keypoints_to_csv(
        [kp.numpy() for kp in input_data['original_sequence']], output_csv_input)
    save_keypoints_to_csv(output_frames, output_csv_output)

    print("Done!")


if __name__ == "__main__":
    main()
