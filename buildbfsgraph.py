import numpy as np
import pandas as pd


def buildBfsGraph(img_dict, r):
    """
    :param img_dict: {img_name: {subimg_name: {y_m: 1, x_m: 1}}}
    :param r:
    """
    edge_list = []
    for img, subimg_dict in img_dict.items():
        edges = {"from": [], "to": []}
        nodes = [k for k, _ in subimg_dict.items()]
        n_nodes = len(nodes)
        current_nodes = [nodes[0]]
        next_nodes = []
        nodes_visited = [nodes[0]]
        # 直到遍历完所有节点
        if len(subimg_dict) == 1:
            edges['from'].append(nodes[0])
            edges['to'].append(nodes[0])
        else:
            while len(nodes_visited) < len(nodes):
                # print("当前节点列表", ','.join([j.split('_')[2].split('^')[0] for j in current_nodes]))
                closest_list = []  # 若current_nodes中节点都没法找到距离r*max(h,w)以内的节点，则记录离这些最近的节点对
                for node_1 in current_nodes:
                    count = 0  # 记录从node_1中拓展出去的节点数量
                    closest_img = None  # 记录距离该节点最近的节点
                    # 遍历其他节点
                    for node_2 in nodes:
                        # 去除已经遍历过的节点
                        if node_1 == node_2 or node_2 in edges['from']:
                            continue
                        # 获取两张子图的位置信息词典
                        v1, v2 = subimg_dict[node_1], subimg_dict[node_2]
                        d_min = 1e6  # 设置初始最短距离
                        d = np.sqrt((v1['x_m'] - v2['x_m']) ** 2 + (v1['y_m'] - v2['y_m']) ** 2)  # 计算距离
                        h, w = v1['h'], v1['w']
                        if d < r * max(h, w) and node_2 not in edges['from']:
                            # 添加边
                            edges['from'].append(node_1)
                            edges['to'].append(node_2)
                            # 记录遍历到的节点
                            nodes_visited.append(node_2)
                            # 加入下一轮BFS应该遍历的节点列表
                            if node_2 not in current_nodes:
                                next_nodes.append(node_2)
                                count += 1
                        # 更新距离最近的节点
                        if d < d_min and node_2 not in nodes_visited:
                            d_min = d
                            closest_img = node_2
                    # 若node_1没有相邻节点，则记录下离它最近的节点
                    if count == 0:
                        closest_list.append([node_1, closest_img])
                # print(closest_list)
                # 若不能从current_nodes的节点能找到下一批节点，则使用最近节点代替
                if not next_nodes:
                    for pair in closest_list:
                        edges['from'].append(pair[0])
                        edges['to'].append(pair[1])
                        next_nodes.append(pair[1])
                        nodes_visited.append(pair[1])
                nodes_visited = list(set(nodes_visited))
                # 更新BFS中节点
                current_nodes, next_nodes = list(set(next_nodes)), []
                # print("扩展了", ','.join([j.split('_')[2].split('^')[0] for j in current_nodes]))
                # print("当前遍历过的节点集合为", ','.join([j.split('_')[2].split('^')[0] for j in nodes_visited]))
        edges = pd.DataFrame(edges)
        edge_list.append(edges)
    edges = pd.concat(edge_list, axis=0)
    edges = pd.DataFrame(edges)
    edges.index = range(edges.shape[0])
    return edges


if __name__ == '__main__':
    img_dict = np.load('./data/dicts/subimg_center_point_dict.npy', allow_pickle=True).item()
    edges = buildBfsGraph(img_dict, r=0.15)
    edges.to_csv("./data/dicts/edges.csv", index=False)
