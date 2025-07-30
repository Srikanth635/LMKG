"""
Knowledge graph visualization tools
"""

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import networkx as nx
    import matplotlib.pyplot as plt

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from .soma_kg import SOMAKnowledgeGraph


class KnowledgeGraphVisualizer:
    """Visualizer for SOMA knowledge graphs"""

    def __init__(self, knowledge_graph: SOMAKnowledgeGraph):
        self.kg = knowledge_graph
        self.node_colors = self._setup_node_colors()

    def _setup_node_colors(self) -> Dict[str, str]:
        """Setup color scheme for different node types"""
        return {
            'instance': '#3498db',  # Blue
            'class': '#e74c3c',  # Red
            'property': '#2ecc71',  # Green
            'literal': '#f39c12',  # Orange
            'affordance': '#9b59b6',  # Purple
            'quality': '#1abc9c',  # Teal
            'material': '#34495e',  # Dark gray
            'shape': '#e67e22'  # Orange-red
        }

    def create_html_visualization(self, filename: str = "knowledge_graph.html",
                                  title: str = "SOMA Knowledge Graph") -> str:
        """Create interactive HTML visualization using vis.js"""

        if not VISUALIZATION_AVAILABLE:
            raise ImportError("NetworkX and matplotlib required for visualization")

        # Convert to NetworkX graph
        G = self._to_networkx()

        # Prepare data for vis.js
        nodes = []
        edges = []

        for node in G.nodes():
            node_type = self._get_node_type(node)
            nodes.append({
                "id": node,
                "label": self._clean_uri(node),
                "color": self.node_colors.get(node_type, '#95a5a6'),
                "title": f"Type: {node_type}\\nURI: {node}",
                "shape": self._get_node_shape(node_type)
            })

        for source, target, data in G.edges(data=True):
            edges.append({
                "from": source,
                "to": target,
                "label": self._clean_uri(data.get('label', '')),
                "arrows": "to",
                "title": data.get('label', '')
            })

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #knowledge-graph {{
            width: 100%;
            height: 600px;
            border: 1px solid lightgray;
        }}
        .controls {{
            margin: 10px 0;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .info {{
            background: #f0f8ff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #007acc;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 10px 0;
        }}
        .stat-item {{
            text-align: center;
            padding: 10px;
        }}
        button {{
            margin: 5px;
            padding: 8px 15px;
            background: #007acc;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }}
        button:hover {{
            background: #005a99;
        }}
    </style>
</head>
<body>
    <h1>🧠 {title}</h1>

    <div class="info">
        <strong>SOMA Knowledge Graph Visualization</strong><br>
        This interactive graph shows the relationships between objects, properties, and concepts
        in the SOMA ontology. Hover over nodes and edges for more information.
    </div>

    <div class="stats">
        <div class="stat-item">
            <strong>{len(nodes)}</strong><br>Nodes
        </div>
        <div class="stat-item">
            <strong>{len(edges)}</strong><br>Edges
        </div>
        <div class="stat-item">
            <strong>{len(self.kg.triples)}</strong><br>Triples
        </div>
        <div class="stat-item">
            <strong>{len(self.kg.instances)}</strong><br>Instances
        </div>
    </div>

    <div class="controls">
        <button onclick="network.fit()">🔍 Fit to Screen</button>
        <button onclick="togglePhysics()">⚡ Toggle Physics</button>
        <button onclick="toggleHierarchy()">📊 Toggle Hierarchy</button>
        <button onclick="exportImage()">📷 Export Image</button>
        <button onclick="showNodeInfo()">ℹ️ Node Info</button>
    </div>

    <div id="knowledge-graph"></div>

    <div class="info">
        <strong>🎨 Legend:</strong><br>
        <span style="color: {self.node_colors['instance']};">●</span> Instances |
        <span style="color: {self.node_colors['class']};">●</span> Classes |
        <span style="color: {self.node_colors['property']};">●</span> Properties |
        <span style="color: {self.node_colors['literal']};">●</span> Literals |
        <span style="color: {self.node_colors['affordance']};">●</span> Affordances
    </div>

    <div id="node-info" style="margin-top: 20px; padding: 10px; background: #f9f9f9; border-radius: 5px; display: none;">
        <h3>Node Information</h3>
        <div id="node-details"></div>
    </div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('knowledge-graph');
        var data = {{
            nodes: nodes,
            edges: edges
        }};

        var options = {{
            nodes: {{
                font: {{
                    size: 12,
                    color: '#000000'
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                width: 2,
                color: {{inherit: 'from'}},
                smooth: {{
                    type: 'continuous'
                }},
                font: {{
                    size: 10,
                    align: 'middle'
                }},
                arrows: {{
                    to: {{scaleFactor: 1.2}}
                }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -80000,
                    springConstant: 0.001,
                    springLength: 200
                }}
            }},
            interaction: {{
                tooltipDelay: 200,
                hover: true
            }},
            layout: {{
                improvedLayout: true
            }}
        }};

        var network = new vis.Network(container, data, options);
        var physicsEnabled = true;
        var hierarchyEnabled = false;
        var selectedNode = null;

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{physics: physicsEnabled}});
        }}

        function toggleHierarchy() {{
            hierarchyEnabled = !hierarchyEnabled;
            if (hierarchyEnabled) {{
                network.setOptions({{
                    layout: {{
                        hierarchical: {{
                            direction: 'UD',
                            sortMethod: 'directed'
                        }}
                    }}
                }});
            }} else {{
                network.setOptions({{
                    layout: {{
                        hierarchical: false
                    }}
                }});
            }}
        }}

        function exportImage() {{
            var canvas = network.body.container.getElementsByTagName('canvas')[0];
            var dataURL = canvas.toDataURL();
            var link = document.createElement('a');
            link.download = 'soma_knowledge_graph.png';
            link.href = dataURL;
            link.click();
        }}

        function showNodeInfo() {{
            if (selectedNode) {{
                var nodeData = nodes.get(selectedNode);
                var nodeInfo = document.getElementById('node-info');
                var nodeDetails = document.getElementById('node-details');

                nodeDetails.innerHTML = `
                    <strong>Label:</strong> ${{nodeData.label}}<br>
                    <strong>ID:</strong> ${{nodeData.id}}<br>
                    <strong>Type:</strong> ${{nodeData.title}}<br>
                `;
                nodeInfo.style.display = 'block';
            }} else {{
                alert('Please select a node first by clicking on it.');
            }}
        }}

        // Event listeners
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                selectedNode = params.nodes[0];
                console.log('Selected node:', selectedNode);
            }} else {{
                selectedNode = null;
                document.getElementById('node-info').style.display = 'none';
            }}
        }});

        network.on("doubleClick", function (params) {{
            if (params.nodes.length > 0) {{
                showNodeInfo();
            }}
        }});

        network.on("stabilizationIterationsDone", function () {{
            network.setOptions({{physics: false}});
        }});

        // Initialize
        network.fit();
    </script>
</body>
</html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)

        return filename

    def create_matplotlib_plot(self, filename: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> Optional[str]:
        """Create static matplotlib visualization"""

        if not VISUALIZATION_AVAILABLE:
            raise ImportError("NetworkX and matplotlib required for visualization")

        G = self._to_networkx()

        plt.figure(figsize=figsize)

        # Use hierarchical layout if possible
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)

        # Draw nodes with different colors and sizes based on type
        node_colors = []
        node_sizes = []

        for node in G.nodes():
            node_type = self._get_node_type(node)
            node_colors.append(self.node_colors.get(node_type, '#95a5a6'))

            # Different sizes for different types
            if node_type == 'instance':
                node_sizes.append(1000)
            elif node_type == 'class':
                node_sizes.append(800)
            elif node_type == 'affordance':
                node_sizes.append(600)
            else:
                node_sizes.append(400)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               arrows=True, arrowsize=20, alpha=0.6,
                               arrowstyle='->')

        # Add labels
        labels = {node: self._clean_uri(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

        # Add edge labels for important relations
        edge_labels = {}
        for source, target, data in G.edges(data=True):
            label = self._clean_uri(data.get('label', ''))
            if len(label) < 15:  # Only show short labels to avoid clutter
                edge_labels[(source, target)] = label

        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

        plt.title("SOMA Knowledge Graph Visualization", size=16, weight='bold')
        plt.axis('off')
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close()
            return filename
        else:
            plt.show()
            return None

    def export_cytoscape_json(self, filename: str = "cytoscape_graph.json") -> str:
        """Export for Cytoscape visualization"""
        elements = []

        # Add nodes
        nodes = set()
        for triple in self.kg.triples:
            if not triple.subject.startswith('"'):
                nodes.add(triple.subject)
            if not triple.object.startswith('"'):
                nodes.add(triple.object)

        for node in nodes:
            node_type = self._get_node_type(node)
            elements.append({
                "data": {
                    "id": node,
                    "label": self._clean_uri(node),
                    "type": node_type,
                    "color": self.node_colors.get(node_type, '#95a5a6')
                }
            })

        # Add edges
        for i, triple in enumerate(self.kg.triples):
            if not triple.object.startswith('"'):  # Skip literal objects
                elements.append({
                    "data": {
                        "id": f"edge_{i}",
                        "source": triple.subject,
                        "target": triple.object,
                        "label": self._clean_uri(triple.predicate),
                        "relation": triple.predicate
                    }
                })

        cytoscape_data = {
            "elements": elements,
            "style": [
                {
                    "selector": "node",
                    "style": {
                        "label": "data(label)",
                        "text-valign": "center",
                        "color": "black",
                        "background-color": "data(color)",
                        "font-size": "10px",
                        "width": "30px",
                        "height": "30px"
                    }
                },
                {
                    "selector": "edge",
                    "style": {
                        "label": "data(label)",
                        "curve-style": "bezier",
                        "target-arrow-shape": "triangle",
                        "font-size": "8px",
                        "text-rotation": "autorotate"
                    }
                }
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cytoscape_data, f, indent=2)

        return filename

    def _to_networkx(self) -> 'nx.DiGraph':
        """Convert knowledge graph to NetworkX graph"""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("NetworkX required for graph conversion")

        G = nx.DiGraph()

        for triple in self.kg.triples:
            if not triple.object.startswith('"'):  # Skip literals for visualization
                G.add_edge(triple.subject, triple.object,
                           label=triple.predicate,
                           relation=self._clean_uri(triple.predicate))

        return G

    def _get_node_type(self, node: str) -> str:
        """Determine node type for visualization"""
        if 'instance' in node:
            # Check if it's a specific type of instance
            instance_id = node.split('#')[-1] if '#' in node else node.split('/')[-1]
            instance_info = self.kg.get_soma_instance_info(instance_id)

            if instance_info:
                class_type = instance_info['class_type'].lower()
                if 'affordance' in class_type:
                    return 'affordance'
                elif any(qual in class_type for qual in ['color', 'shape', 'temperature']):
                    return 'quality'
                elif 'material' in class_type:
                    return 'material'
                elif 'shape' in class_type:
                    return 'shape'

            return 'instance'
        elif 'soma' in node:
            return 'class'
        elif node.startswith('"'):
            return 'literal'
        else:
            return 'property'

    def _get_node_shape(self, node_type: str) -> str:
        """Get node shape for vis.js visualization"""
        shape_mapping = {
            'instance': 'dot',
            'class': 'box',
            'property': 'diamond',
            'literal': 'square',
            'affordance': 'star',
            'quality': 'triangle',
            'material': 'hexagon',
            'shape': 'ellipse'
        }
        return shape_mapping.get(node_type, 'dot')

    def _clean_uri(self, uri: str) -> str:
        """Clean URI for display"""
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.split('/')[-1]
        return uri.strip('"').split('^^')[0]

    def create_summary_report(self) -> Dict[str, any]:
        """Create a summary report of the knowledge graph"""
        stats = self.kg.get_soma_statistics()

        # Analyze the graph structure
        if VISUALIZATION_AVAILABLE:
            G = self._to_networkx()
            network_stats = {
                "is_connected": nx.is_weakly_connected(G),
                "number_of_components": nx.number_weakly_connected_components(G),
                "average_degree": sum(dict(G.degree()).values()) / len(G.nodes()) if G.nodes() else 0,
                "density": nx.density(G)
            }
        else:
            network_stats = {"message": "NetworkX not available for network analysis"}

        # Find most connected nodes
        node_connections = {}
        for triple in self.kg.triples:
            # Count subject connections
            if triple.subject not in node_connections:
                node_connections[triple.subject] = 0
            node_connections[triple.subject] += 1

            # Count object connections (if not literal)
            if not triple.object.startswith('"'):
                if triple.object not in node_connections:
                    node_connections[triple.object] = 0
                node_connections[triple.object] += 1

        # Get top connected nodes
        top_connected = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:5]
        top_connected_clean = [(self._clean_uri(node), count) for node, count in top_connected]

        return {
            "basic_statistics": stats,
            "network_analysis": network_stats,
            "top_connected_nodes": top_connected_clean,
            "visualization_available": VISUALIZATION_AVAILABLE
        }