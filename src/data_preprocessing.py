import matplotlib.pyplot as plt
import networkx as nx
import pytesseract
import cv2
import pandas as pd
import seaborn as sns
import numpy as np


class DataPreprocessing:
    def __init__(self, train_a_path, train_b_path):
        """
        Initialize the class and load the datasets.
        """
        try:
            self.train_a = pd.read_csv(train_a_path)
            self.train_b = pd.read_csv(train_b_path)
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def print_basic_statistics(self):
        """
        Print basic statistics for the datasets.
        """
        print("\n--- Basic Statistics for TRAIN_A ---")
        print(self.train_a.describe())
        print("\n--- Basic Statistics for TRAIN_B ---")
        print(self.train_b.describe())

    def check_null_values(self):
        """
        Print the count of null values for each dataset.
        """
        print("\n--- Null Values in TRAIN_A ---")
        print(self.train_a.isnull().sum())
        print("\n--- Null Values in TRAIN_B ---")
        print(self.train_b.isnull().sum())

    def print_anomaly_distribution(self):
        """
        Print the distribution of anomalies in the datasets.
        """
        print("\n--- Anomaly Distribution in TRAIN_A ---")
        print(self.train_a['anomaly'].value_counts(normalize=True))
        print("\n--- Anomaly Distribution in TRAIN_B ---")
        print(self.train_b['anomaly'].value_counts(normalize=True))


if __name__ == "__main__":
    # Provide file paths
    train_a_path = "open/train/TRAIN_A.csv"
    train_b_path = "open/train/TRAIN_B.csv"

    # Initialize the preprocessing class
    dp = DataPreprocessing(train_a_path, train_b_path)

    # Call preprocessing methods
    dp.print_basic_statistics()
    dp.check_null_values()
    dp.print_anomaly_distribution()

# 이미지 경로
image_path = "open/meta_관망구조이미지/train/관망구조_A.jpg"

# 이미지 로드
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이미지 이진화
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# OCR을 사용해 텍스트 추출
# Tesseract 경로 설정 (필요 시 수정)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
custom_config = r'--oem 3 --psm 6'  # OCR 설정
detected_text = pytesseract.image_to_data(
    binary, config=custom_config, output_type=pytesseract.Output.DICT)

# 노드(압력계, 펌프) 추출
nodes = []
for i in range(len(detected_text['text'])):
    text = detected_text['text'][i].strip()
    if text.startswith("P") or text.startswith("Q"):  # 노드 이름(P, Q 탐지)
        nodes.append(
            (text, (detected_text['left'][i], detected_text['top'][i])))

# 엣지(선) 추출 - Canny Edge Detection
edges = cv2.Canny(binary, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                        threshold=50, minLineLength=50, maxLineGap=10)

# 네트워크 그래프 생성
G = nx.Graph()
for node, _ in nodes:
    G.add_node(node)

# 엣지 연결 (임의로 탐지된 텍스트 위치와 연결을 매핑)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        for node1, pos1 in nodes:
            for node2, pos2 in nodes:
                if node1 != node2:
                    if abs(pos1[0] - x1) < 20 and abs(pos1[1] - y1) < 20:  # 근접한 노드 연결
                        G.add_edge(node1, node2)

# 그래프 시각화
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color="lightblue", font_weight="bold")
plt.title("Network Graph Extracted from Image")
plt.show()


# Visualize distributions for specific columns
def visualize_corrected_distributions(data):
    # Visualize Q (Flow rate) for Q1 to Q5
    q_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    data_melted_q = data.melt(id_vars=[
                              'dataset'], value_vars=q_columns, var_name='Flow Rate (Q)', value_name='Value')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data_melted_q, x='Flow Rate (Q)',
                y='Value', hue='dataset')
    plt.title('Distribution of Flow Rate (Q)')
    plt.xlabel('Flow Rate (Q)')
    plt.ylabel('Value')
    plt.show()

    # Visualize M (Pump operation status) for M1 to M5
    m_columns = [col for col in data.columns if col.startswith('M')]
    data_melted_m = data.melt(
        id_vars=['dataset'], value_vars=m_columns, var_name='Pump (M)', value_name='Value')

    plt.figure(figsize=(12, 6))
    sns.countplot(data=data_melted_m, x='Value', hue='dataset')
    plt.title('Distribution of Pump Operation (M)')
    plt.xlabel('Pump Operation Status')
    plt.ylabel('Count')
    plt.show()

    # Visualize P (Pressure) for P1 to P10
    p_columns = [col for col in data.columns if col.startswith(
        'P') and not col.endswith('_flag')]
    data_melted_p = data.melt(id_vars=[
                              'dataset'], value_vars=p_columns, var_name='Pressure (P)', value_name='Value')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data_melted_p, x='Pressure (P)', y='Value', hue='dataset')
    plt.title('Distribution of Pressure (P)')
    plt.xlabel('Pressure (P)')
    plt.ylabel('Value')
    plt.show()

    # Visualize anomaly
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='anomaly', hue='dataset')
    plt.title('Distribution of Anomaly')
    plt.xlabel('Anomaly (0: Normal, 1: Abnormal)')
    plt.ylabel('Count')
    plt.show()

    # Visualize P_flag
    p_flag_columns = [col for col in data.columns if col.endswith('_flag')]
    data_melted_p_flag = data.melt(id_vars=[
                                   'dataset'], value_vars=p_flag_columns, var_name='Pressure Flag (P_flag)', value_name='Value')

    plt.figure(figsize=(12, 6))
    sns.countplot(data=data_melted_p_flag, x='Value', hue='dataset')
    plt.title('Distribution of Pressure Flags (P_flag)')
    plt.xlabel('P_flag (0: Normal, 1: Abnormal)')
    plt.ylabel('Count')
    plt.show()


# Execute the updated visualization
visualize_corrected_distributions(combined_data)
