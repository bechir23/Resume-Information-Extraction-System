import matplotlib.pyplot as plt
from wordcloud import WordCloud
def visualize_entity_lengths(stats, entity_type=None):
    """Create histograms of entity lengths"""
    
    
    if entity_type and entity_type in stats:
        # Create histogram for a specific entity type
        plt.figure(figsize=(10, 5))
        plt.hist(stats[entity_type]["lengths"], bins=20)
        plt.title(f"Length Distribution for {entity_type} entities")
        plt.xlabel("Length (chars)")
        plt.ylabel("Count")
        plt.show()
    else:
        # Create histograms for all entity types
        entity_types = list(stats.keys())
        num_types = len(entity_types)
        
        if num_types > 0:
            # Arrange in a grid
            cols = 3
            rows = (num_types + cols - 1) // cols
            
            plt.figure(figsize=(16, 4*rows))
            
            for i, entity_type in enumerate(entity_types):
                plt.subplot(rows, cols, i+1)
                plt.hist(stats[entity_type]["lengths"], bins=20)
                plt.title(f"{entity_type} (n={stats[entity_type]['count']})")
                plt.xlabel("Length (chars)")
                plt.ylabel("Count")
                
            plt.tight_layout()
            plt.show()

def create_pattern_visualizations(data, entity_type):
    """Create visualizations of text patterns for a given entity type"""
    
    
    
    # Extract all entities of the specified type
    entities = []
    for text, annot in data:
        for start, end, label in annot["entities"]:
            if label == entity_type:
                entities.append(text[start:end].strip())
    
    if not entities:
        print(f"No entities found of type: {entity_type}")
        return
    
    print(f"Found {len(entities)} {entity_type} entities")
    
    # Create a wordcloud
    text = " ".join(entities)
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white', 
                         collocations=False,
                         max_words=100).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud for {entity_type}")
    plt.axis('off')
    plt.show()
    
    # Show some examples
    print(f"Sample {entity_type} entities:")
    for i, entity in enumerate(entities[:10]):
        print(f"{i+1}. {entity}")

def create_entity_relationship_visualizations(data):
    """Visualize relationships between entity types"""
    import matplotlib.pyplot as plt
    import networkx as nx
    from collections import defaultdict
    
    # Track which entity types appear together in documents
    entity_cooccurrence = defaultdict(int)
    entity_counts = defaultdict(int)
    
    for text, annot in data:
        # Get unique entity types in this document
        types_in_doc = set()
        for start, end, label in annot["entities"]:
            types_in_doc.add(label)
            entity_counts[label] += 1
        
        # Record co-occurrences (for each pair of entity types)
        entity_types = list(types_in_doc)
        for i in range(len(entity_types)):
            for j in range(i+1, len(entity_types)):
                pair = tuple(sorted([entity_types[i], entity_types[j]]))
                entity_cooccurrence[pair] += 1
    
    # Create a network graph
    G = nx.Graph()
    
    # Add nodes with size based on entity count
    max_count = max(entity_counts.values()) if entity_counts else 1
    for entity_type, count in entity_counts.items():
        G.add_node(entity_type, size=(count / max_count) * 1500 + 500)
    
    # Add edges with weight based on co-occurrence count
    for pair, count in entity_cooccurrence.items():
        G.add_edge(pair[0], pair[1], weight=count)
    
    # Draw the network
    plt.figure(figsize=(12, 10))
    
    # Get node sizes
    node_sizes = [G.nodes[n]['size'] for n in G]
    
    # Get edge weights for width
    edge_widths = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    
    # Spring layout positions nodes by simulating a force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title("Entity Type Co-occurrence Network")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Most common entity types:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")
    
    print("\nMost common entity type co-occurrences:")
    for pair, count in sorted(entity_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pair[0]} + {pair[1]}: {count}")
