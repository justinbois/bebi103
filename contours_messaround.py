# converts a filled isocontours to a set of polygons for plotting in bokeh

import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
import networkx as nx

from scipy.spatial import Delaunay
def rgb_to_hex(rgb_frac):
    """
    Convert fractional RGB values to hexidecimal color string.

    Parameters
    ----------
    rgb_frac : array_like, shape (3,)
        Fractional RGB values; each entry is between 0 and 1.

    Returns
    -------
    str
        Hexidecimal string for the given RGB color.

    Examples
    --------
    >>> rgb_frac_to_hex((0.65, 0.23, 1.0))
    '#a53aff'

    >>> rgb_frac_to_hex((1.0, 1.0, 1.0))
    '#ffffff'
    """

    if len(rgb_frac) != 3:
        raise RuntimeError('`rgb_frac` must have exactly three entries.')

    if (np.array(rgb_frac) < 0).any() or (np.array(rgb_frac) > 1).any():
        raise RuntimeError('RGB values must be between 0 and 1.')

    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb_frac[0] * 255),
                                           int(rgb_frac[1] * 255),
                                           int(rgb_frac[2] * 255))


def triangulated_graph(polygons):
	tri = Delaunay(np.vstack(polygons))

	#create ring identifiers and indices into ring vertices
	ring_index = []
	local_dex = []
	for i,poly in enumerate(polygons):
		ring_index.append(i+np.zeros(len(poly), dtype=np.int))
		local_dex.append(np.arange(len(poly)))
	ring_index = np.hstack(ring_index)
	local_dex = np.hstack(local_dex)

	edges = set()
	for simplex in tri.simplices:
		edges.add(tuple(sorted([simplex[0], simplex[1]])))
		edges.add(tuple(sorted([simplex[0], simplex[2]])))
		edges.add(tuple(sorted([simplex[1], simplex[2]])))

	#put undirected edges in graph
	triangle_graph = nx.Graph()
	for e0,e1 in edges:
		triangle_graph.add_edge(e0, e1, weight=np.linalg.norm(tri.points[e0]-tri.points[e1]))

	# put node data in graph
	for i,p in enumerate(tri.points):
		triangle_graph.add_node(i,ring=ring_index[i], local = local_dex[i])

	return triangle_graph


""" Create a graph of ring islands. some islands will have two bridges joining them """
def create_islands(graph):
	#create minimum spanning tree from undirected edges
	mst_edges = sorted(list(nx.minimum_spanning_edges(graph,data=True)))
	islands = nx.Graph()
	for e0, e1, w in mst_edges:
		ring0 = graph.node[e0]['ring']
		ring1 = graph.node[e1]['ring']
		local0, local1 = graph.node[e0]['local'], graph.node[e1]['local']
		if  ring0 != ring1:
			islands.add_edge(ring0, ring1, weight = w,
							connection = [e0, e1, local0, local1],
							)
	return islands

""" Inserts degenerate edge, replacing nodes with new ones
 0 -> 1a 1b -> 2      0 -> 1a -> 3b -> 4 -> 5 -> 3a -> 1b -> 2
      |   |
 4 <- 3b 3a <- 5 <- 4
"""
def insert_branch(graph, edge):
	for e in edge:
		prev, next = graph.predecessors(e), graph.successors(e)
		if len(prev) > 0:
			graph.add_edge(prev[0],e-.1)
			graph.node[e-.1] = graph.node[e]

		if len(next) > 0:
			graph.add_edge(e+.1,next[0])
			graph.node[e+.1] = graph.node[e]

		graph.remove_node(e)

	# link new nodes. Order won't matter when doing shortest path query
	graph.add_edge(edge[0]+.1, edge[1]-.1)
	graph.add_edge(edge[0]-.1, edge[1]+.1)
	return graph

# create a directed graph from polygons
def polygon_graph(polygons):
	g = nx.DiGraph()
	shift = 0
	for i,poly in enumerate(polygons):
		g.add_cycle(range(shift,shift+len(poly)))
		shift += len(poly)
	return g

"""Get the shortest path that traverses the edges """
def get_merged_path(poly_graph):
	#select an edge, remove it and find path that circles back
	e1, e0 = poly_graph.edges()[0]
	ug = poly_graph.to_undirected()
	ug.remove_edge(e0,e1)
	# round to get the index of the original vertices
	return [int(round(x)) for x in nx.shortest_path(ug,e0)[e1]]

"""Connect each island through one bridge"""
def merge_islands(islands, polygons):
	poly_graph = polygon_graph(polygons)

	mst_islands = nx.minimum_spanning_tree(islands, 0)
	for r_0, r_1 in mst_islands.edges():
		e0, e1, local0, local1 = mst_islands[r_0][r_1]['connection']
		poly_graph = insert_branch(poly_graph,(e0,e1))

	return poly_graph

def filled_contours(p,cn,simplify_threshold = .01):
	"""Creates a bokeh plot of filled contours

    Args:
    	p (bokeh.plotting.Figure): Bokeh plot instance
        cn (contours): Contours generated from plt.contourf()
        simplify_threshold (Optional[float]): Resolution of the output contours in screenspace. Defaults to .01

    Returns:
        None

    """
	for cc in cn.collections:
		face_color = np.array(cc.get_facecolor()[0])
		color = rgb_to_hex(tuple((face_color[:-1]).round().astype(int)))
		alpha = face_color[-1]

		for path in cc.get_paths():
			path.simplify_threshold = simplify_threshold
			polygons = path.to_polygons()
			if len(polygons) == 1:
				p.patch(polygons[0][:,0], polygons[0][:,1], line_alpha = alpha, line_color = color, fill_alpha = alpha, fill_color = color)
			else:
				vertices = np.vstack(polygons)
				graph = triangulated_graph(polygons)
				islands = create_islands(graph)
				poly_graph = merge_islands(islands,polygons)
				merged_path = get_merged_path(poly_graph)

				p.patch(vertices[merged_path][:,0],vertices[merged_path][:,1], line_alpha = alpha, line_color = color, fill_alpha = alpha, fill_color = color)


def main(argv):
	fig = plt.figure()
	ax = fig.add_subplot(121)
	N = 200

	x = np.linspace(0, 10, N)
	y = np.linspace(0, 10, N)
	xx, yy = np.meshgrid(x, y)
	d = np.sin(xx)*np.cos(2*yy)+.1*yy
	d = d + .015*np.random.rand(*xx.shape)

	cn = plt.contourf(x,y,d,9, extend = "both")

	ax2 = fig.add_subplot(122)
	ax2.xlim = [0,10]
	ax2.ylim = [0,10]

	html_size_without_plots = 1638397 # bytes
	html_size_with_p0_only = 1901692 # bytes
	html_size_with_p1_only = 2463969 # bytes

	output_file('filled_contours.html', title='filled contours')
	p0 = figure(x_range=[0, 10], y_range=[0, 10])
	filled_contours(p0, cn,.02)

	p1 = figure(x_range=[0, 10], y_range=[0, 10])
	p1.image(image = [d], x=[0], y=[0], dw=10, dh=10, palette="Spectral11")
	show(p0)

if __name__ == '__main__':
	import sys
	main(sys.argv[1:])
