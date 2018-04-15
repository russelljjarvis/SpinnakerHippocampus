library(igraph)

#take the betweeness central. take the top 10. graph that network. for all
#ei quiets
#ii excites
#ee excites
#ie quiets
#neuron number Cv of spike. PCA with neuron number axis
#denise's analys on small world placed next to chord diagram
#slider with threshold
#improving visual of network


col_names <- c(1:122)


inhib_mat <- read.csv(file="conn_mat0.csv", header = FALSE, row.names=NULL)
colnames(inhib_mat) <- col_names
row.names(inhib_mat) = c(colnames(inhib_mat))

inhib_graph <- graph_from_adjacency_matrix(as.matrix(inhib_mat), mode = "directed")
between_inhib <- betweenness(inhib_graph)

excit_mat <- read.csv(file="conn_mat1.csv", header = FALSE, row.names=NULL)
colnames(excit_mat) <- col_names
row.names(excit_mat) = c(colnames(excit_mat))

excit_graph <- graph_from_adjacency_matrix(as.matrix(excit_mat), mode = "directed")
between_excit <- betweenness(excit_graph)
